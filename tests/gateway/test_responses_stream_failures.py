"""Streaming Responses outcome regression for #63894, through POST SSE and stored GET.

Only the agent result and its callbacks are faked; routing, batching, SSE serialization,
terminal snapshots, and retrieval use the real adapter.
"""
import json
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


USAGE = {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}
SECRET = "sk-fake-secret-responses-63894"
DIAGNOSTIC = f"provider unavailable OPENAI_API_KEY={SECRET}"
PRIOR = "Safe text already delivered."


async def _exchange(result, *, deltas=(), disable_global_redaction=False):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    app = web.Application()
    app.router.add_post("/v1/responses", adapter._handle_responses)
    app.router.add_get("/v1/responses/{response_id}", adapter._handle_get_response)

    async def fake_run(**kwargs):
        for delta in deltas:
            kwargs["stream_delta_callback"](delta)
        return result, USAGE

    with patch.object(adapter, "_run_agent", side_effect=fake_run):
        # API error redaction must force masking even when the global toggle is off.
        with patch("agent.redact._REDACT_ENABLED", False) if disable_global_redaction else patch(
                "agent.redact._REDACT_ENABLED", True):
            async with TestClient(TestServer(app)) as client:
                response = await client.post(
                    "/v1/responses", json={"model": "hermes-agent", "input": "Hi", "stream": True})
                assert response.status == 200
                wire = await response.text()
                events = [json.loads(line[6:]) for line in wire.splitlines()
                          if line.startswith("data: ") and line != "data: [DONE]"]
                created = next(e["response"] for e in events if e["type"] == "response.created")
                fetched_response = await client.get(f"/v1/responses/{created['id']}")
                assert fetched_response.status == 200
                fetched = await fetched_response.json()
    return events, fetched, wire


def _deltas(events):
    return [e["delta"] for e in events if e["type"] == "response.output_text.delta"]


def _terminal(events):
    return [e for e in events if e["type"] in ("response.failed", "response.completed")]


def _output_text(env):
    return [part["text"] for item in env["output"] if item["type"] == "message"
            for part in item["content"] if part["type"] == "output_text"]


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [
    {"failed": True, "completed": True, "error": DIAGNOSTIC, "final_response": DIAGNOSTIC},
    {"failed": True, "completed": False, "error": DIAGNOSTIC, "final_response": DIAGNOSTIC},
    {"completed": False, "error": DIAGNOSTIC, "final_response": DIAGNOSTIC},
    {"failed": True, "final_response": DIAGNOSTIC},  # error fallback is final_response
    {"failed": True, "error": DIAGNOSTIC},  # error fallback without final_response
], ids=["failed-even-if-completed", "failed-incomplete", "incomplete-with-error",
        "failed-final-fallback", "failed-error-only"])
@pytest.mark.parametrize("prior", [False, True], ids=["no-prior-text", "prior-text"])
async def test_flagged_failure_never_streams_final_as_delta(result, prior):
    events, fetched, wire = await _exchange(
        result, deltas=(PRIOR,) if prior else (), disable_global_redaction=True)
    terminals = _terminal(events)
    assert [e["type"] for e in terminals] == ["response.failed"]
    env = terminals[0]["response"]
    assert env["status"] == fetched["status"] == "failed"
    assert env["id"] == fetched["id"]
    assert env["usage"] == fetched["usage"] == USAGE
    assert _deltas(events) == ([PRIOR] if prior else [])
    if prior:
        assert _output_text(env) == _output_text(fetched) == [PRIOR]
    else:
        # Legacy failed terminal envelope includes a redacted diagnostic as assistant output.
        assert _output_text(env) == _output_text(fetched)
    assert SECRET not in wire and SECRET not in json.dumps(fetched)
    assert DIAGNOSTIC not in wire and DIAGNOSTIC not in json.dumps(fetched)
    if result.get("error"):
        assert "OPENAI_API_KEY=" in env["error"]["message"]
    elif "final_response" not in result:
        assert env["error"]["message"] == "agent run failed"


@pytest.mark.asyncio
@pytest.mark.parametrize("prior", [False, True], ids=["no-prior-text", "prior-text"])
@pytest.mark.parametrize("rewrite", [False, True], ids=["append", "rewrite"])
async def test_failed_transformed_final_is_not_appended_or_replaced(prior, rewrite):
    final = f"{PRIOR} {DIAGNOSTIC}" if prior and not rewrite else DIAGNOSTIC
    result = {"failed": True, "error": DIAGNOSTIC,
              "final_response": final,
              "response_transformed": True, "pre_transform_response": PRIOR if prior else ""}
    events, fetched, wire = await _exchange(result, deltas=(PRIOR,) if prior else ())
    assert [e["type"] for e in _terminal(events)] == ["response.failed"]
    assert _deltas(events) == ([PRIOR] if prior else [])
    assert SECRET not in wire and SECRET not in json.dumps(fetched)
    if prior:
        assert _output_text(fetched) == [PRIOR]


@pytest.mark.asyncio
async def test_error_only_without_final_remains_a_failed_terminal():
    events, fetched, _ = await _exchange({"error": DIAGNOSTIC, "final_response": ""})
    assert [e["type"] for e in _terminal(events)] == ["response.failed"]
    assert _deltas(events) == []
    assert fetched["status"] == "failed"
    assert SECRET not in json.dumps(fetched)


@pytest.mark.asyncio
@pytest.mark.parametrize("result,deltas,expected_deltas,expected_text", [
    ({"final_response": "Only at end"}, (), ["Only at end"], "Only at end"),
    ({"completed": False, "final_response": "Still output"}, (), ["Still output"], "Still output"),
    ({"failed": False, "completed": False, "error": "", "final_response": "Still output"},
     (), ["Still output"], "Still output"),
    ({"final_response": "Raw footer", "response_transformed": True,
      "pre_transform_response": "Raw"}, ("Raw",), ["Raw", " footer"], "Raw footer"),
    ({"final_response": "Rewritten", "response_transformed": True,
      "pre_transform_response": "Raw"}, ("Raw",), ["Raw"], "Rewritten"),
], ids=["provider-batch-final", "incomplete-no-error", "falsy-error",
        "success-append-transform", "success-rewrite-transform"])
async def test_non_failure_terminal_contract_unchanged(result, deltas, expected_deltas, expected_text):
    events, fetched, _ = await _exchange(result, deltas=deltas)
    assert [e["type"] for e in _terminal(events)] == ["response.completed"]
    assert _deltas(events) == expected_deltas
    assert _output_text(fetched) == [expected_text]
    assert fetched["status"] == "completed" and fetched["usage"] == USAGE
    assert [e["text"] for e in events if e["type"] == "response.output_text.done"] == [expected_text]
