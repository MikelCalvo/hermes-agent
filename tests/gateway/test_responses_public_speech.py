"""Opt-in public speech on the real /v1/responses aiohttp/SSE boundary.

The model is synthetic; callbacks, route, writer, and response store are production code.
"""
import asyncio
import json
from unittest.mock import patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from agent.stream_delivery import StreamDeliveryMixin
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from tests.gateway.test_api_server import _create_app


@pytest.fixture
def adapter():
    return APIServerAdapter(PlatformConfig(enabled=True, extra={}))


class _StreamAgent(StreamDeliveryMixin):
    """Exercise the real delta/interim classifier without calling a provider."""

    def __init__(self, kw):
        self.stream_delta_callback = kw["stream_delta_callback"]
        self.interim_assistant_callback = kw["interim_assistant_callback"]
        self._stream_callback = None
        self.show_commentary = True
        self._strip_think_blocks = lambda text: text

    def _enqueue_stream_hook(self, *a, **kw):
        pass


def _events(chunk):
    frames = []
    for raw in chunk.split(b"\n\n"):
        event = next((line[7:] for line in raw.decode().splitlines() if line.startswith("event: ")), None)
        data = next((line[6:] for line in raw.decode().splitlines() if line.startswith("data: ")), None)
        if event and data:
            frames.append((event, json.loads(data)))
    return frames


async def _read_until(response, event):
    chunks = bytearray()
    async for line in response.content:
        chunks.extend(line)
        if any(name == event for name, _ in _events(bytes(chunks))):
            return bytes(chunks)
    raise AssertionError(f"stream ended before {event}")


def _messages(envelope):
    return [(item.get("phase"), item["content"][0]["text"])
            for item in envelope.get("output", []) if item["type"] == "message"]


@pytest.mark.asyncio
async def test_commentary_reaches_client_before_held_final_and_private_protocol_never_enters_queue(adapter):
    started, release = asyncio.Event(), asyncio.Event()
    secret = "PRIVATE_TOOL_ID_arg_result_reason_status_" + "x" * 2500
    queued = []

    async def run(**kw):
        agent = _StreamAgent(kw)
        kw["agent_ref"][0] = agent
        kw["tool_start_callback"](secret, secret, {"argument": secret})
        kw["reasoning_callback"](secret)
        kw["status_callback"]("lifecycle", secret)
        # Ordinary providers send an interim preamble through _fire_stream_delta first.
        agent._fire_stream_delta("I'll check now.")
        agent._emit_interim_assistant_message({"role": "assistant", "content": "I'll check now.",
                                               "tool_calls": [{"id": secret}]})
        kw["tool_complete_callback"](secret, secret, {}, secret)
        started.set()
        await release.wait()
        return {"final_response": "The answer is 42.", "completed": True,
                "messages": [{"role": "assistant", "content": "I'll check now.",
                              "tool_calls": [{"id": secret, "function": {"name": secret, "arguments": secret}}]},
                             {"role": "tool", "tool_call_id": secret, "content": secret},
                             {"role": "assistant", "content": "The answer is 42."}]}, None

    async with TestClient(TestServer(_create_app(adapter))) as client:
        with patch.object(adapter, "_run_agent", side_effect=run), patch(
            "gateway.platforms.api_server.ThreadSafeAsyncQueue.put_threadsafe",
            autospec=True,
            side_effect=lambda self, item: (queued.append(item), self.put_nowait(item)),
        ):
            response = await client.post("/v1/responses", json={
                "input": "question", "stream": True, "public_commentary": True})
            assert response.status == 200
            await asyncio.wait_for(started.wait(), 3)
            early = await asyncio.wait_for(_read_until(response, "response.output_item.done"), 3)
            early_frames = _events(early)
            assert any(e == "response.output_item.done" and d["item"].get("phase") == "commentary"
                       and d["item"]["content"][0]["text"] == "I'll check now." for e, d in early_frames)
            assert "response.completed" not in [e for e, _ in early_frames]
            assert secret not in early.decode()
            assert secret not in repr(queued)
            release.set()
            wire = early + await response.read()
            frames = _events(wire)
            terminal = [d["response"] for e, d in frames if e == "response.completed"]
            assert len(terminal) == 1
            assert _messages(terminal[0]) == [("commentary", "I'll check now."), (None, "The answer is 42.")]
            assert "".join(d["delta"] for e, d in frames if e == "response.output_text.delta") == "The answer is 42."
            assert secret not in wire.decode()
            ids = [(d["output_index"], d["item"]["id"]) for e, d in frames if e == "response.output_item.done"]
            assert [idx for idx, _ in ids] == [0, 1]
            assert ids[0][1] == terminal[0]["output"][0]["id"]
            assert ids[1][1] == terminal[0]["output"][1]["id"]
            stored = await client.get("/v1/responses/" + terminal[0]["id"])
            assert _messages(await stored.json()) == _messages(terminal[0])
            assert secret not in await stored.text()
            history = adapter._current_response_store().get(terminal[0]["id"])["conversation_history"]
            assert secret in repr(history)  # only the public projection is narrowed


@pytest.mark.asyncio
@pytest.mark.parametrize("result,expected", [
    ({"final_response": "SECRET_DIAGNOSTIC", "failed": True, "error": "SECRET_DIAGNOSTIC"}, "failed"),
    ({"final_response": "SECRET_DIAGNOSTIC", "completed": False, "partial": True}, "incomplete"),
    ({"final_response": "SECRET_DIAGNOSTIC", "interrupted": True}, "incomplete"),
    ({"final_response": "", "error": "SECRET_DIAGNOSTIC"}, "failed"),
    (RuntimeError("SECRET_DIAGNOSTIC"), "failed"),
])
@pytest.mark.parametrize("commentary", [False, True])
async def test_unsuccessful_result_never_promotes_diagnostic_to_speech(adapter, result, expected, commentary):
    async def run(**kw):
        if commentary:
            kw["interim_assistant_callback"]("Safe progress.", already_streamed=False)
        if isinstance(result, Exception):
            raise result
        return result, None

    async with TestClient(TestServer(_create_app(adapter))) as client:
        with patch.object(adapter, "_run_agent", side_effect=run):
            response = await client.post("/v1/responses", json={
                "input": "question", "stream": True, "public_commentary": True})
            wire = await response.read()
        assert response.status == 200
        assert "SECRET_DIAGNOSTIC" not in wire.decode()
        frames = _events(wire)
        terminal = [(e, d["response"]) for e, d in frames if e in
                    ("response.completed", "response.failed", "response.incomplete")]
        assert len(terminal) == 1
        event, envelope = terminal[0]
        assert (event, envelope["status"]) == ("response." + expected, expected)
        assert _messages(envelope) == ([("commentary", "Safe progress.")] if commentary else [])
        assert envelope["hermes"]["partial"] is commentary
        assert not any(e == "response.output_text.delta" for e, _ in frames)
        stored = await client.get("/v1/responses/" + envelope["id"])
        assert "SECRET_DIAGNOSTIC" not in await stored.text()


@pytest.mark.asyncio
async def test_failure_after_delivered_delta_keeps_only_delivered_public_text(adapter):
    async def run(**kw):
        kw["stream_delta_callback"]("Safe spoken prefix.")
        return {"final_response": "SECRET_DIAGNOSTIC", "completed": False,
                "failed": True, "error": "SECRET_DIAGNOSTIC"}, None

    async with TestClient(TestServer(_create_app(adapter))) as client:
        with patch.object(adapter, "_run_agent", side_effect=run):
            response = await client.post("/v1/responses", json={
                "input": "question", "stream": True, "public_commentary": True})
            wire = await response.read()
        frames = _events(wire)
        assert "SECRET_DIAGNOSTIC" not in wire.decode()
        terminal = next(d["response"] for e, d in frames if e == "response.failed")
        assert _messages(terminal) == []
        assert terminal["hermes"]["partial"] is False
        assert not any(e == "response.output_text.delta" for e, _ in frames)
        assert (await client.get("/v1/responses/" + terminal["id"])).status == 200


@pytest.mark.asyncio
async def test_uncategorized_final_is_held_until_agent_completes(adapter):
    entered, finish = asyncio.Event(), asyncio.Event()

    async def run(**kw):
        kw["stream_delta_callback"]("A delayed answer.")
        entered.set()
        await finish.wait()
        return {"final_response": "A delayed answer.", "completed": True}, None

    async with TestClient(TestServer(_create_app(adapter))) as client:
        with patch.object(adapter, "_run_agent", side_effect=run):
            response = await client.post("/v1/responses", json={
                "input": "q", "stream": True, "public_commentary": True})
            await asyncio.wait_for(entered.wait(), 3)
            early = await _read_until(response, "response.created")
            assert [e for e, _ in _events(early)] == ["response.created"]
            # No extra SSE frame while the task is blocked: inspect the transport's
            # already-buffered bytes without a clock-based negative wait.
            assert not any(e == "response.output_text.delta" for raw in response.content._buffer
                           for e, _ in _events(raw))
            finish.set()
            frames = _events(early + await response.read())
            assert [e for e, _ in frames if e == "response.output_text.delta"] == ["response.output_text.delta"]
            assert [e for e, _ in frames if e == "response.completed"] == ["response.completed"]


@pytest.mark.asyncio
async def test_public_opt_in_requires_stream_and_does_not_start_agent(adapter):
    async with TestClient(TestServer(_create_app(adapter))) as client:
        with patch.object(adapter, "_run_agent") as run:
            response = await client.post("/v1/responses", json={
                "input": "question", "public_commentary": True, "stream": False})
        assert response.status == 400
        run.assert_not_called()
        capability = await (await client.get("/v1/capabilities")).json()
        assert capability["features"]["responses_public_speech_streaming"] is True


@pytest.mark.asyncio
async def test_public_store_false_and_generic_stream_are_independent(adapter):
    async def run(**kw):
        kw["stream_delta_callback"]("ordinary delta")
        kw["tool_start_callback"]("call-private", "terminal", {"x": 1})
        return {"final_response": "ordinary delta", "completed": True}, None

    async with TestClient(TestServer(_create_app(adapter))) as client:
        with patch.object(adapter, "_run_agent", side_effect=run):
            public = await client.post("/v1/responses", json={
                "input": "question", "public_commentary": True, "stream": True, "store": False})
            public_frames = _events(await public.read())
            terminal = next(d["response"] for e, d in public_frames if e == "response.completed")
            assert _messages(terminal) == [(None, "ordinary delta")]
            assert not any(item["type"] == "function_call" for item in terminal["output"])
            assert (await client.get("/v1/responses/" + terminal["id"])).status == 404
            generic = await client.post("/v1/responses", json={"input": "question", "stream": True})
            generic_frames = _events(await generic.read())
            assert any(e == "response.output_item.added" and d["item"]["type"] == "function_call"
                       for e, d in generic_frames)
            assert any(e == "response.output_text.delta" for e, _ in generic_frames)
