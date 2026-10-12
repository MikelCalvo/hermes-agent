"""HTTP/SSE regressions for the legacy Responses streaming failure contract.

Only the agent boundary is faked; request parsing, SSE writing, storage, and
GET retrieval run through the real aiohttp handlers. Generic failed envelopes
may contain diagnostic assistant output/history: this is not a public-speech
privacy guarantee for content already emitted as text deltas.
"""

import json
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


USAGE = {"input_tokens": 13, "output_tokens": 5, "total_tokens": 18}
SECRET = "sk-1234567890abcdefghijklmnopqrstuv"


@pytest.fixture
def adapter():
    instance = APIServerAdapter(PlatformConfig(enabled=True))
    yield instance
    instance._response_store.close()


async def _stream_and_retrieve(adapter, result, *, delta=None):
    app = web.Application()
    app.router.add_post("/v1/responses", adapter._handle_responses)
    app.router.add_get("/v1/responses/{response_id}", adapter._handle_get_response)

    async def fake_agent(**kwargs):
        if delta is not None:
            kwargs["stream_delta_callback"](delta)
        return result, USAGE

    async with TestClient(TestServer(app)) as client:
        with patch.object(adapter, "_run_agent", side_effect=fake_agent) as run:
            response = await client.post(
                "/v1/responses", json={"model": "hermes-agent", "input": "hello", "stream": True},
            )
            assert response.status == 200, await response.text()
            assert response.headers["Content-Type"].startswith("text/event-stream")
            wire = await response.text()
            run.assert_awaited_once()
        events = []
        for frame in wire.split("\n\n"):
            if frame.startswith("event: "):
                event_line, data_line = frame.split("\n", 1)
                payload = json.loads(data_line.removeprefix("data: "))
                assert event_line.removeprefix("event: ") == payload["type"]
                events.append(payload)
        assert events[0]["type"] == "response.created"
        terminal = [e for e in events if e["type"] in {"response.failed", "response.completed"}]
        assert len(terminal) == 1
        envelope = terminal[0]["response"]
        retrieved = await client.get(f"/v1/responses/{envelope['id']}")
        assert retrieved.status == 200, await retrieved.text()
        stored = await retrieved.json()
        assert stored == envelope
    return events, envelope, stored, wire


def _types(events):
    return [e["type"] for e in events]


def _text(envelope):
    return envelope["output"][-1]["content"][0]["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize("delta", [None, "safe earlier text"])
async def test_flagged_failure_with_error_bearing_final_is_terminal_failure(adapter, delta):
    result = {"failed": True, "completed": False, "error": "provider stalled",
              "final_response": "provider stalled"}
    events, envelope, _, _ = await _stream_and_retrieve(adapter, result, delta=delta)
    assert _types(events)[-1] == "response.failed"
    assert "response.completed" not in _types(events)
    assert envelope["status"] == "failed"
    assert envelope["error"]["message"] == "provider stalled"
    assert envelope["usage"] == USAGE
    assert _text(envelope) == (delta or "provider stalled")
    if delta:
        assert [e["delta"] for e in events if e["type"] == "response.output_text.delta"] == [delta]
        assert [e["text"] for e in events if e["type"] == "response.output_text.done"] == [delta]
    else:
        assert "response.output_text.delta" not in _types(events)
        assert "response.output_text.done" not in _types(events)


@pytest.mark.asyncio
async def test_failed_flag_without_error_uses_final_as_error_without_emitting_it_as_delta(adapter):
    events, envelope, _, _ = await _stream_and_retrieve(
        adapter, {"failed": True, "final_response": "provider exhausted"},
    )
    assert _types(events)[-1] == "response.failed"
    assert envelope["error"]["message"] == "provider exhausted"
    assert "response.output_text.done" not in _types(events)
    assert "response.output_text.delta" not in _types(events)


@pytest.mark.asyncio
async def test_failed_flag_with_no_final_uses_default_error(adapter):
    events, envelope, _, _ = await _stream_and_retrieve(adapter, {"failed": True})
    assert _types(events)[-1] == "response.failed"
    assert envelope["error"]["message"] == "agent run failed"
    assert "response.output_text.done" not in _types(events)


@pytest.mark.asyncio
async def test_incomplete_with_error_but_no_failed_flag_is_failure(adapter):
    events, envelope, _, _ = await _stream_and_retrieve(
        adapter, {"completed": False, "error": "network stopped", "final_response": "network stopped"},
        delta="partial delivery",
    )
    assert _types(events)[-1] == "response.failed"
    assert envelope["error"]["message"] == "network stopped"
    assert envelope["usage"] == USAGE
    assert _text(envelope) == "partial delivery"


@pytest.mark.asyncio
async def test_legacy_error_only_with_empty_final_is_failure(adapter):
    # Causal regression: this has no failed/completed flags. The original
    # error-only check must survive inside the success/fallback branch.
    events, envelope, _, _ = await _stream_and_retrieve(
        adapter, {"error": "legacy run error", "final_response": ""},
    )
    assert _types(events)[-1] == "response.failed"
    assert envelope["status"] == "failed"
    assert envelope["error"]["message"] == "legacy run error"
    assert envelope["usage"] == USAGE
    assert "response.output_text.done" not in _types(events)


@pytest.mark.asyncio
async def test_successful_stream_preserves_delivered_text_and_usage(adapter):
    events, envelope, _, _ = await _stream_and_retrieve(
        adapter, {"final_response": "full text", "completed": True}, delta="live text",
    )
    assert _types(events)[-1] == "response.completed"
    assert envelope["status"] == "completed"
    assert envelope["usage"] == USAGE
    assert _text(envelope) == "live text"
    assert [e["delta"] for e in events if e["type"] == "response.output_text.delta"] == ["live text"]


@pytest.mark.asyncio
async def test_successful_final_only_fallback_emits_text(adapter):
    events, envelope, _, _ = await _stream_and_retrieve(adapter, {"final_response": "final only"})
    assert _types(events)[-1] == "response.completed"
    assert _text(envelope) == "final only"
    assert [e["delta"] for e in events if e["type"] == "response.output_text.delta"] == ["final only"]
    assert [e["text"] for e in events if e["type"] == "response.output_text.done"] == ["final only"]


@pytest.mark.asyncio
async def test_completed_false_without_error_does_not_expand_failure_predicate(adapter):
    events, envelope, _, _ = await _stream_and_retrieve(
        adapter, {"completed": False, "final_response": "fallback"},
    )
    assert _types(events)[-1] == "response.completed"
    assert envelope["status"] == "completed"
    assert _text(envelope) == "fallback"


@pytest.mark.asyncio
async def test_error_metadata_redacted_even_when_global_redaction_disabled(adapter):
    # No prior delta and no final: the generic failure envelope's assistant
    # fallback is also sourced from the forcibly redacted error metadata.
    with patch("agent.redact._REDACT_ENABLED", False):
        events, envelope, stored, wire = await _stream_and_retrieve(
            adapter, {"error": f"OPENAI_API_KEY={SECRET}", "final_response": ""},
        )
    assert _types(events)[-1] == "response.failed"
    assert envelope["status"] == stored["status"] == "failed"
    assert SECRET not in wire
    assert SECRET not in json.dumps(stored)
    assert "OPENAI_API_KEY=" in envelope["error"]["message"]
    assert "response.output_text.done" not in _types(events)
