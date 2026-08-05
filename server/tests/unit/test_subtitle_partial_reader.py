"""Tests for `bench.suites.subtitle._PartialReader` (T020 round-2 Critic
blocker: the send-then-receive structure imposed an unconditional floor of
`_TRAILING_SILENCE_S` on the measured latency).

Hermetic: a fake WebSocket stub (`_FakeWS`) stands in for the real
`websockets.sync.client` connection, so this proves `_PartialReader`'s own
concurrency behavior without a real server or socket. The real fix's
network integration (subtitle.py driving a live session) is not executed in
this round -- these tests are the code-level proof the coordinator asked
for that the floor is actually gone, not merely a narrative claim.
"""

from __future__ import annotations

import json
import queue
import threading
import time

import bench.suites.subtitle as subtitle


class _FakeWS:
    """Minimal stand-in for a `websockets.sync.client` connection: `recv()`
    blocks up to `timeout` seconds for a message pushed via `push()`, and
    raises `TimeoutError` (matching the real client) if none arrives."""

    def __init__(self) -> None:
        self._queue: queue.Queue[str] = queue.Queue()

    def recv(self, timeout: float) -> str:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError("no message within timeout")

    def push(self, message: str) -> None:
        self._queue.put(message)


def _translation_message(text: str = "hello") -> str:
    return json.dumps({"type": "translation", "text": text, "segment_id": 1})


def test_partial_reader_captures_a_message_that_arrives_while_the_main_thread_is_still_busy():
    """The whole point of the fix: a message arriving *during* what would be
    the send-pacing window is captured near its actual arrival time, not
    only after that window ends. Simulates the exact shape of the bug --
    the main thread stays "busy" (sleeping, standing in for the paced send
    loop) for far longer than the reply takes to arrive."""
    ws = _FakeWS()
    reader = subtitle._PartialReader(ws, deadline=time.monotonic() + 5.0)
    reader.start()
    start = time.perf_counter()

    threading.Timer(0.2, ws.push, args=(_translation_message(),)).start()

    # Simulate a send phase that, under the pre-fix send-then-receive
    # structure, would have blocked `ws.recv()` from ever being called
    # until this sleep finished -- deliberately much longer than the 0.2s
    # reply delay above.
    time.sleep(0.6)

    found = reader.wait_for_partial(timeout=2.0)
    reader.join(timeout=2.0)

    assert found
    elapsed_s = reader.first_partial_time - start
    assert elapsed_s < 0.4, (
        f"partial captured at {elapsed_s:.3f}s after a 0.2s reply delay during a 0.6s "
        "'busy' phase -- expected ~0.2s (concurrent receive), not floored by the busy phase"
    )
    assert reader.message_type == "translation"


def test_partial_reader_ignores_non_matching_messages_before_the_real_one():
    ws = _FakeWS()
    reader = subtitle._PartialReader(ws, deadline=time.monotonic() + 5.0)
    reader.start()

    ws.push(json.dumps({"type": "status", "status": "connected"}))
    ws.push(json.dumps({"type": "transcription_final", "text": "hi", "is_final": True, "segment_id": 1}))
    ws.push(_translation_message("the real one"))

    found = reader.wait_for_partial(timeout=2.0)
    reader.join(timeout=2.0)

    assert found
    assert reader.message_type == "translation"


def test_partial_reader_prefers_translation_partial_type_when_that_is_what_arrives():
    ws = _FakeWS()
    reader = subtitle._PartialReader(ws, deadline=time.monotonic() + 5.0)
    reader.start()

    ws.push(json.dumps({"type": "translation_partial", "text": "he", "segment_id": 1}))

    found = reader.wait_for_partial(timeout=2.0)
    reader.join(timeout=2.0)

    assert found
    assert reader.message_type == "translation_partial"


def test_partial_reader_gives_up_at_its_own_deadline_when_nothing_arrives():
    """Bounded even if `wait_for_partial` were never called at all: the read
    loop's own `deadline` is what stops it, not the caller. `join(timeout=2.0)`
    returning is not by itself proof of that -- `Thread.join` also returns on
    a timeout without the thread having actually finished (T020 round-3
    Critic finding: this test asserted nothing about *why* `join` returned).
    Asserting `is_alive() is False` afterward is what actually distinguishes
    "the thread honored its 0.3s deadline" from "join's own 2.0s timeout
    elapsed while the thread was still stuck."""
    ws = _FakeWS()
    reader = subtitle._PartialReader(ws, deadline=time.monotonic() + 0.3)
    reader.start()

    reader.join(timeout=2.0)  # must return well within 2s -- the read loop's own 0.3s deadline

    assert not reader._thread.is_alive(), "the read loop must have actually exited by its own deadline"
    assert reader.first_partial_time is None
    assert reader.message_type is None


def test_partial_reader_stops_quietly_when_recv_raises():
    """A closed/errored connection must not crash the background thread --
    and the exception must actually be caught inside `_read_loop`, not
    merely happen to leave the thread in a finished state.

    `is_alive() is False` alone does **not** prove that (T020 round-3
    Critic finding, confirmed empirically): Python still tears a thread
    down when its target function raises uncaught -- `is_alive()` reads
    `False` either way, whether `_read_loop` returned cleanly via `except
    Exception: return` or crashed via an unhandled `ConnectionError`.
    Deleting that `except` clause and re-running this test with only an
    `is_alive()` assertion still passed, which is exactly why that
    assertion alone is not a real regression guard for this behavior.
    Installing a `threading.excepthook` for the duration of this test and
    asserting it was never invoked is what actually distinguishes "caught
    internally" from "escaped uncaught and was silently handled by the
    interpreter's default thread-exception reporting" -- only the fixed
    code leaves that hook untouched.
    """

    class _BrokenWS:
        def recv(self, timeout: float) -> str:
            raise ConnectionError("closed")

    uncaught_thread_exceptions: list[threading.ExceptHookArgs] = []
    previous_excepthook = threading.excepthook
    threading.excepthook = uncaught_thread_exceptions.append
    try:
        reader = subtitle._PartialReader(_BrokenWS(), deadline=time.monotonic() + 5.0)
        reader.start()
        reader.join(timeout=2.0)
    finally:
        threading.excepthook = previous_excepthook

    assert uncaught_thread_exceptions == [], (
        "recv()'s ConnectionError must be caught inside _read_loop, never escape the thread uncaught"
    )
    assert not reader._thread.is_alive(), "the read loop must have exited, not hung, after recv raised"
    assert reader.first_partial_time is None
