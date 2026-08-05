"""`subtitle` suite (contracts/benchmark-cli.md, tasks.md T018).

Owns the server lifecycle end-to-end (contract precondition 5 / "the suite
owns the server lifecycle -- it starts `src.main` as a subprocess, waits for
readiness, and tears it down on every exit path including failure"): starts
`src.main` as a real subprocess, waits for `/health` readiness, drives one
real WebSocket session per iteration, times speech-end -> first partial, and
tears the subprocess down whether the run succeeded, failed a precondition
partway through, or raised -- never leaving an orphaned server behind
listening on the port a later run's own precondition check depends on being
free.

Pre-US5 note: `MessageType.TRANSLATION_PARTIAL` does not exist until Phase 5
(tasks.md T043-T051) lands. Until then the only message carrying translated
text is the final `translation` message, so this suite treats whichever of
`translation_partial` / `translation` arrives first as "first partial" --
the same wire message simply becomes the true streamed partial once US5
ships, with no change required here.

**Measurement origin** (Critic finding, T020 round 1, major 6; the
explicit statement T023 requires of "speech end"): audio is sent **paced
in real time** -- one chunk every `CHUNK_DURATION_MS`, matching how a real
client streams -- not blasted as fast as the loopback socket allows.
Without that pacing, two things go wrong in the same direction: the VAD
silence hold's real wall-clock cost (600-1000ms, depending on the effective
config) never actually elapses, because the trailing-silence *samples*
arrive and get scanned in a instant burst rather than over real time, so
the measured interval **excludes** that wait entirely; at the same time,
whatever CPU the server spends running Silero VAD inference across the
whole burst of queued chunks lands inside the measured interval instead of
being amortized across the seconds a real session would take to arrive,
so the measured interval **includes** VAD-scan compute it should not. Both
errors point the same way -- an implausibly *fast* number -- and pacing
removes both. `speech_end_time` is recorded at the instant the last chunk
of **real speech** (not the synthetic trailing-silence padding appended
after it) finishes sending; this is the instant speaking physically stops,
which is the same anchor `e2e.py`'s stage composition uses (`vad_hold` is
counted as a separate additive stage there specifically because the hold's
wait comes *after* this instant, not before it).

**Anchor quantization** (Critic finding, T020 round 3, minor -- documented,
not fixed: not worth adding code to a several-hundred-millisecond budget
for a two-digit-millisecond effect): sending is chunked at
`CHUNK_DURATION_MS` granularity, and the real-audio/trailing-silence
boundary rarely falls exactly on a chunk edge, so `speech_end_time` is
actually stamped at the end of whichever chunk *crosses* that boundary --
up to one `CHUNK_DURATION_MS` **after** the true instant speech ends, never
before. For `utterance_ko_5s.wav` specifically (80602 samples at 16 kHz,
1600-sample/100ms chunks) that overshoot is ~62 ms. Because
`speech_end_time` lands *late* relative to the true instant, `latency_ms =
first_partial_time - speech_end_time` is biased **short** by that same
~62 ms -- i.e. this suite's numbers, including in the smoke run's evidence,
read up to ~62 ms **faster** than the true speech-end-to-first-partial gap,
never slower. This error is one-directional and bounded by one chunk
duration; it does not change which budget-comparison direction a future
optimization needs to move to pass.

**Concurrent receive, not send-then-receive** (Critic finding, T020 round 2,
blocker): the send loop above paces the *whole* payload -- speech plus
`_TRAILING_SILENCE_S` of padding -- in real time before this round's fix,
`ws.recv()` did not start until every chunk had already been sent. That
makes `_TRAILING_SILENCE_S` (1.5s) an unconditional **floor** on the
measured latency, regardless of how fast the real pipeline is:
`measured = max(true_latency, padding_duration)`. A successful future
optimization landing at, say, 450ms would still be reported as ~1500ms and
FAIL a budget it actually met. Reducing the padding cannot fix this either
-- it must stay above the VAD hold (>= 550ms) or speech-end is never
detected at all. The fix is structural: `_PartialReader` runs `ws.recv()`
on a dedicated background thread for the whole session, timestamping the
first partial the instant it arrives, while the main thread keeps pacing
sends concurrently. This is the pattern the `websockets` sync client
supports directly -- concurrent `recv()` calls from two threads raise
`ConcurrencyError`, but one thread sending while a single other thread
recvs is fine (`websockets.sync.connection`'s own docs restrict only
concurrent *recv*, never a concurrent send alongside it).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

from .. import harness

_SERVER_DIR = Path(__file__).resolve().parents[2]
_FIXTURE = _SERVER_DIR / "tests" / "fixtures" / "utterance_ko_5s.wav"
_READY_TIMEOUT_S = 90.0
_FIRST_PARTIAL_TIMEOUT_S = 30.0
# Trailing silence appended after the fixture so the VAD's silence hold
# actually elapses and a speech-end segment is emitted, whatever the
# effective VAD_SILENCE_DURATION_MS is configured to (1000ms today; T055
# lowers it to 550ms later in the feature).
_TRAILING_SILENCE_S = 1.5


class SubtitleServerError(RuntimeError):
    """The subtitle suite's own server subprocess failed to start, become
    ready, or answer -- distinct from a plain timing assertion failure so
    the same teardown path runs either way (this exception IS a normal exit
    path, not a bypass of one)."""


class _PartialReader:
    """Owns every `ws.recv()` call for one session, on a dedicated
    background thread, so the main thread is free to keep pacing sends
    concurrently (module docstring "Concurrent receive, not
    send-then-receive", Critic T020 round-2 blocker).

    Bounded by construction: the read loop below re-checks `deadline`
    (wall-clock) at least once a second (each `ws.recv(timeout=1.0)` call
    returns on its own after at most 1s even with nothing to read), so the
    thread cannot outlive `deadline` by more than that margin regardless of
    whether the main thread ever calls `wait()`/`join()` at all.
    """

    def __init__(self, ws, deadline: float) -> None:
        self._ws = ws
        self._deadline = deadline
        self.first_partial_time: float | None = None
        self.message_type: str | None = None
        self._found = threading.Event()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _read_loop(self) -> None:
        while time.monotonic() < self._deadline and not self._found.is_set():
            try:
                raw = self._ws.recv(timeout=1.0)
            except TimeoutError:
                continue
            except Exception:
                return  # connection closed/errored underneath us -- stop quietly
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if msg.get("type") in ("translation_partial", "translation"):
                self.first_partial_time = time.perf_counter()
                self.message_type = msg.get("type")
                self._found.set()
                return

    def wait_for_partial(self, timeout: float) -> bool:
        """Block up to `timeout` seconds for the first partial. Bounded
        independently of this call too: the read loop's own `deadline`
        means it gives up on its own even if this timeout were generous."""
        return self._found.wait(timeout=timeout)

    def join(self, timeout: float) -> None:
        self._thread.join(timeout=timeout)


def _wait_for_health(port: int, deadline: float) -> None:
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1.0) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(0.5)
    raise SubtitleServerError(f"server did not become ready on port {port} within the deadline")


def _terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        # Best-effort (Critic T020 round-1 minor 5): a plain SIGKILL never
        # gives `src.main`'s own shutdown handler (which calls
        # `TTSEngine.shutdown()`) a chance to run, so its TTS worker
        # subprocess would otherwise be orphaned. Killing the whole process
        # group this server was started in (see `start_new_session=True`
        # below) reaches that worker too -- though not any further
        # descendant a worker might itself spawn outside this group.
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def run(iterations: int, config: dict) -> tuple[list, list[dict]]:
    import websockets.sync.client as ws_client

    from src.config import AUDIO_PREFIX_SYSTEM, CHUNK_BYTES, PORT, SAMPLE_RATE

    process = subprocess.Popen(
        [sys.executable, "-m", "src.main"], cwd=str(_SERVER_DIR), start_new_session=True
    )
    try:
        _wait_for_health(PORT, time.monotonic() + _READY_TIMEOUT_S)

        audio, sr = sf.read(str(_FIXTURE), dtype="float32")
        if sr != SAMPLE_RATE:
            raise SubtitleServerError(f"fixture samplerate {sr} != {SAMPLE_RATE}")
        trailing_silence = np.zeros(int(_TRAILING_SILENCE_S * SAMPLE_RATE), dtype=np.float32)
        payload = np.concatenate([audio, trailing_silence]).astype(np.float32)
        chunk_samples = CHUNK_BYTES // 4
        chunk_duration_s = chunk_samples / SAMPLE_RATE

        latencies_ms: list[float] = []
        samples: list[dict] = []
        uri = f"ws://127.0.0.1:{PORT}/ws"

        payload_duration_s = len(payload) / SAMPLE_RATE

        for i in range(iterations):
            with ws_client.connect(uri, open_timeout=10) as ws:
                ws.recv(timeout=5)  # initial "connected" status message -- still on the
                # main thread here, sequentially before the reader thread exists, so
                # there is no concurrent-recv hazard with what follows.
                ws.send(
                    json.dumps(
                        {
                            "type": "configure",
                            "mode": "listen",
                            "language": "korean",
                            "target_language": "english",
                        }
                    )
                )

                # From here on, exactly one thread ever calls `ws.recv()` (the
                # reader thread) while the main thread only sends -- the
                # supported concurrency pattern for this client (see module
                # docstring). Bounded by construction: `_PartialReader`'s own
                # internal deadline covers the full expected duration of this
                # iteration (send pacing + how long a real reply may still
                # take after that), independent of anything below.
                reader = _PartialReader(
                    ws, deadline=time.monotonic() + payload_duration_s + _FIRST_PARTIAL_TIMEOUT_S
                )
                reader.start()

                # Paced to real time (module docstring "Measurement origin",
                # Critic T020 round-1 major 6): a `next_send` deadline
                # anchored to the previous target, not a fixed `sleep()` per
                # chunk, so per-chunk overhead does not accumulate drift
                # across the ~6.5s payload. Sending concurrently with the
                # reader thread above -- not after it -- is exactly what
                # removes the send-then-receive measurement floor (module
                # docstring, Critic T020 round-2 blocker).
                speech_end_time: float | None = None
                offset = 0
                next_send = time.monotonic()
                while offset < len(payload):
                    chunk = payload[offset : offset + chunk_samples]
                    offset += chunk_samples
                    ws.send(bytes([AUDIO_PREFIX_SYSTEM]) + chunk.tobytes())
                    if offset >= len(audio) and speech_end_time is None:
                        speech_end_time = time.perf_counter()
                    next_send += chunk_duration_s
                    sleep_for = next_send - time.monotonic()
                    if sleep_for > 0:
                        time.sleep(sleep_for)

                # Sending is done; the reader may already have found the
                # partial concurrently with the pacing loop above (the
                # common case once the pipeline is fast) or may still be
                # waiting for it. Either way this wait is bounded by
                # `_FIRST_PARTIAL_TIMEOUT_S`, and the reader's own deadline
                # bounds it again independently.
                found = reader.wait_for_partial(timeout=_FIRST_PARTIAL_TIMEOUT_S)
                reader.join(timeout=2.0)

                if not found or reader.first_partial_time is None:
                    raise SubtitleServerError(
                        f"no translation_partial/translation arrived within "
                        f"{_FIRST_PARTIAL_TIMEOUT_S}s (iteration {i})"
                    )

                # Not reachable today given real pipeline latency, but the
                # concurrent-receive design (module docstring) means the
                # reader thread's timestamp and this thread's
                # `speech_end_time` are captured independently -- this is
                # the last line of defense against ever reporting a
                # negative/zero latency as a real measurement, should that
                # independence ever break down (T020 round-3 Critic minor).
                if reader.first_partial_time <= speech_end_time:
                    raise SubtitleServerError(
                        f"first partial arrived at or before speech_end_time "
                        f"(iteration {i}) -- not a valid latency measurement"
                    )

                latency_ms = (reader.first_partial_time - speech_end_time) * 1000
                latencies_ms.append(latency_ms)
                samples.append(
                    {
                        "iteration": i,
                        "first_partial_ms": latency_ms,
                        "message_type": reader.message_type,
                    }
                )
    finally:
        _terminate(process)

    mean_latency = sum(latencies_ms) / len(latencies_ms)
    stages = [harness.stage_timing("subtitle_first_token", mean_latency)]
    return stages, samples
