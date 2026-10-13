"""Request-owned public speech projection for opt-in Responses SSE consumers.

Protocol callbacks stay private before enqueue. Raw text deltas are held until a completed
interim message classifies them as commentary, or the agent completes and they become final
speech. This delay is intentional: ordinary providers send a preamble through the delta
callback before the agent knows it was an interim tool-round message.
"""
import asyncio
import logging
import time
import uuid
from contextlib import suppress
from typing import Any

from gateway.platforms.api_server_openai_routes import _ResponsesStream, _iter_stream_items, _message_item

logger = logging.getLogger("gateway.platforms.api_server")


def spawn_public_speech(adapter, run_kwargs):
    queue = PublicSpeechQueue()
    agent_task, agent_ref = adapter._spawn_stream_agent(
        queue, interim_assistant_callback=public_callbacks(queue),
        tool_start_callback=lambda *_: None, tool_complete_callback=lambda *_: None,
        tool_progress_callback=lambda *_args, **_kwargs: None, **run_kwargs)
    return queue, agent_task, agent_ref


async def start_public_speech(adapter, request, *, body, run_kwargs, conversation_history,
                              user_message, instructions, conversation, store, session_id,
                              gateway_session_key):
    queue, task, agent_ref = spawn_public_speech(adapter, run_kwargs)
    return await write_public_speech(
        adapter, request, response_id=f"resp_{uuid.uuid4().hex[:28]}",
        model=body.get("model", adapter._model_name), created_at=int(time.time()),
        stream_q=queue, agent_task=task, agent_ref=agent_ref,
        conversation_history=conversation_history, user_message=user_message,
        instructions=instructions, conversation=conversation, store=store,
        session_id=session_id, gateway_session_key=gateway_session_key)


class PublicSpeechQueue:
    """Only speech, classification markers and EOS can cross this queue boundary."""

    def __init__(self):
        from gateway.platforms.api_server import ThreadSafeAsyncQueue
        self._queue = ThreadSafeAsyncQueue()

    def put_threadsafe(self, item):
        if isinstance(item, str):
            self._queue.put_threadsafe(item)
        elif isinstance(item, tuple) and len(item) == 2 and item[0] == "__commentary__":
            text = item[1]
            if isinstance(text, dict) and isinstance(text.get("text"), str):
                self._queue.put_threadsafe(item)

    def put_nowait(self, item):
        if item is None:
            self._queue.put_nowait(None)

    async def get(self):
        return await self._queue.get()

    def get_nowait(self):
        return self._queue.get_nowait()


def public_callbacks(queue):
    """Bind agent callback semantics to one request, without changing generic producer callbacks."""
    # A callback runs on the worker thread, after ordinary-provider raw deltas.
    def on_commentary(text, *, already_streamed=False):
        if isinstance(text, str) and text.strip():
            queue.put_threadsafe(("__commentary__", {"text": text, "already_streamed": already_streamed}))
    return on_commentary


class PublicSpeechStream(_ResponsesStream):
    """The opt-in writer owns its own result classification and stored public envelope."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pending_text: list[str] = []
        self.public_text: list[str] = []
        self._commentary: list[dict[str, Any]] = []
        self._terminal_message: dict[str, Any] | None = None

    async def dispatch(self, item: Any) -> None:
        if isinstance(item, str):
            self.pending_text.append(item)
        elif isinstance(item, tuple) and len(item) == 2 and item[0] == "__commentary__":
            payload = item[1]
            if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
                return
            text = payload["text"]
            if payload.get("already_streamed"):
                pending = "".join(self.pending_text)
                # Exact match only: never drop unclassified trailing final tokens.
                if pending.strip() == text.strip():
                    self.pending_text.clear()
                else:
                    return
            await self.emit_commentary(text)

    async def emit_commentary(self, text: str) -> None:
        item = {"id": f"msg_{uuid.uuid4().hex[:24]}", "type": "message",
                "role": "assistant", "status": "completed", "phase": "commentary",
                "content": [{"type": "output_text", "text": text}]}
        idx = self.output_index
        self.output_index += 1
        self._commentary.append(item)
        for event in ("response.output_item.added", "response.output_item.done"):
            await self.write_event(event, {"type": event, "output_index": idx, "item": item})

    def _output(self):
        return [*self._commentary, *([self._terminal_message] if self._terminal_message else [])]

    def _history(self):
        history = self._history_with_user()
        if self.public_text:
            history.append({"role": "assistant", "content": "".join(self.public_text)})
        return history

    def persist_incomplete_if_needed(self):
        if self.store and not self.terminal_snapshot_persisted:
            self.persist_snapshot(self.terminal_envelope("incomplete", self._output()), history=self._history())

    async def _close_public_final(self):
        text = "".join(self.public_text)
        if not text:
            return
        await self._open_message_item()
        await self.write_event("response.output_text.delta", {
            "type": "response.output_text.delta", "item_id": self.message_item_id,
            "output_index": self.message_output_index, "content_index": 0,
            "delta": text, "logprobs": []})
        await self.write_event("response.output_text.done", {
            "type": "response.output_text.done", "item_id": self.message_item_id,
            "output_index": self.message_output_index, "content_index": 0,
            "text": text, "logprobs": []})
        self._terminal_message = {"id": self.message_item_id, "status": "completed", **_message_item(text)}
        await self.write_event("response.output_item.done", {
            "type": "response.output_item.done", "output_index": self.message_output_index,
            "item": self._terminal_message})

    async def finish(self, agent_task):
        try:
            result, usage = await agent_task
            self.usage = usage or self.usage
        except Exception:
            logger.exception("Agent failed during public speech Responses stream")
            result = {"failed": True, "completed": False}
        self.result = result
        failed = not isinstance(result, dict) or bool(result.get("failed") or result.get("error"))
        incomplete = isinstance(result, dict) and bool(
            result.get("interrupted") or result.get("partial") or result.get("completed") is False)
        status = "failed" if failed else "incomplete" if incomplete else "completed"
        # A failed result's final_response can be a decorated provider diagnostic. Never
        # project it as speech even when it differs from result.error.
        if status == "completed":
            self.public_text = [*self.pending_text]
            fallback = result.get("final_response")
            # A result-only final is public only when the run has no diagnostic flags.
            if not self.public_text and isinstance(fallback, str) and fallback:
                self.public_text = [fallback]
        else:
            # Unclassified deltas may be provider diagnostics or failed attempts;
            # nothing not already delivered is promoted after an unsuccessful turn.
            self.public_text = []
        await self._close_public_final()
        env = self.terminal_envelope(status, self._output())
        if status != "completed":
            env["error"] = {"type": "server_error", "message": "Agent did not complete."}
            env["hermes"] = {"completed": False, "failed": failed,
                             "partial": bool(self._commentary or self.public_text),
                             "interrupted": bool(isinstance(result, dict) and result.get("interrupted"))}
            history = self._history()
        else:
            self.final_response_text = "".join(self.public_text)
            history = self.adapter._build_response_conversation_history(
                self.conversation_history, self.user_message, result, self.final_response_text,
                tool_output_max_chars=self.adapter._history_tool_output_max_chars)
        sid = result.get("session_id") if status == "completed" else None
        self.persist_snapshot(env, history=history,
                              session_id=sid if isinstance(sid, str) and sid else None)
        self.terminal_snapshot_persisted = True
        await self.write_event("response." + status, {"type": "response." + status, "response": env})


async def write_public_speech(adapter, request, *, response_id, model, created_at, stream_q,
                              agent_task, agent_ref, conversation_history, user_message,
                              instructions, conversation, store, session_id, gateway_session_key=None):
    """Opt-in writer, separate from the generic Responses writer and batch projection."""
    from gateway.platforms.api_server import _abandon_agent_task
    response = await adapter._prepare_sse_response(request, session_id, gateway_session_key)
    st = PublicSpeechStream(
        adapter, response, response_id=response_id, model=model, created_at=created_at,
        conversation_history=conversation_history, user_message=user_message,
        instructions=instructions, conversation=conversation, store=store, session_id=session_id)
    try:
        await st.emit_created()
        async for item in _iter_stream_items(stream_q, agent_task, response):
            if item is None:
                break
            await st.dispatch(item)
        await st.finish(agent_task)
    except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
        st.persist_incomplete_if_needed()
        await _abandon_agent_task(agent_ref, agent_task, "SSE client disconnected")
    except asyncio.CancelledError:
        st.persist_incomplete_if_needed()
        await _abandon_agent_task(agent_ref, agent_task, "SSE task cancelled",
                                  reap_source="api_server_sse_cancelled", await_cancel=False)
        raise
    except Exception:
        logger.exception("Public speech Responses writer failed")
        st.persist_incomplete_if_needed()
        with suppress(Exception):
            env = st.terminal_envelope("failed", st._output(), error="Agent did not complete.")
            await st.write_event("response.failed", {"type": "response.failed", "response": env})
    return response
