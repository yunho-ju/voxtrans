import logging
import multiprocessing as mp
import threading
import time
from multiprocessing import Queue
from queue import Empty

import numpy as np

from .config import TTS_BACKEND, TTS_MODEL, TTS_SAMPLE_RATE
from .tts_backends import default_model, make_backend

logger = logging.getLogger(__name__)

# Sentinel values for streaming protocol
_CHUNK = "chunk"
_DONE = "done"
_ERROR = "error"


def _tts_worker(request_q: Queue, response_q: Queue, backend_name: str,
                model_name: str, sample_rate: int) -> None:
    """Dedicated TTS process — loads model and processes requests."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] TTS-Worker: %(message)s",
    )
    log = logging.getLogger("tts_worker")

    try:
        import mlx.core as mx

        backend = make_backend(backend_name)
        log.info("Loading model %s (backend %s)", model_name, backend_name)
        t0 = time.time()
        backend.load(model_name, sample_rate, log)
        log.info("Model loaded in %.1fs", time.time() - t0)
    except Exception as e:
        log.exception("Failed to load TTS model")
        response_q.put(("error", str(e)))
        return

    response_q.put("ready")

    while True:
        try:
            req = request_q.get()
            if req is None:  # shutdown signal
                break

            text, voice, lang_code, req_id, ref_audio_np, ref_text = req
            is_voice_clone = ref_audio_np is not None
            log.info("Generating TTS for req %d: voice=%s, lang=%s, clone=%s, text='%s'",
                     req_id, voice, lang_code, is_voice_clone, text[:50])
            try:
                t0 = time.time()
                chunk_count = 0

                # Generate and send chunks immediately
                for result in backend.generate(
                    text, voice, lang_code, ref_audio_np, ref_text, log
                ):
                    audio = result.audio
                    if isinstance(audio, mx.array):
                        mx.eval(audio)
                        audio_np = np.array(audio, dtype=np.float32).flatten()
                        del audio
                        mx.clear_cache()
                    else:
                        audio_np = np.asarray(audio, dtype=np.float32).flatten()
                        del audio
                    if audio_np.size > 0:
                        response_q.put((req_id, _CHUNK, audio_np, sample_rate))
                        chunk_count += 1
                        log.info("  chunk %d: %d samples (%.1fs)",
                                 chunk_count, audio_np.size, audio_np.size / sample_rate)
                # Clean up GPU memory between requests to prevent slowdown
                mx.clear_cache()
                response_q.put((req_id, _DONE, None, None))
                log.info("TTS done for req %d: %d chunks in %.1fs",
                         req_id, chunk_count, time.time() - t0)
            except Exception as e:
                log.exception("Generate failed for req %d", req_id)
                response_q.put((req_id, _ERROR, str(e), None))
        except Exception:
            log.exception("Unexpected error in worker loop")


class TTSEngine:
    _MAX_RESTARTS = 3

    def __init__(self) -> None:
        self._process: mp.Process | None = None
        self._request_q: Queue | None = None
        self._response_q: Queue | None = None
        self._req_counter = 0
        self._active_req_id = 0
        self._restart_count = 0
        self._lock = threading.RLock()
        # Which backend the worker is running. Starts from configuration and
        # can be changed at runtime (`switch_backend`), so it is instance
        # state rather than a module constant read at each spawn -- a restart
        # after a crash has to come back as the backend in use, not the one
        # the server booted with.
        self._backend_name = TTS_BACKEND
        self._model_name = TTS_MODEL

    @property
    def backend(self) -> str:
        with self._lock:
            return self._backend_name

    @property
    def active_backend(self) -> str | None:
        with self._lock:
            return self._backend_name if self._process is not None and self._process.is_alive() else None

    def backend_snapshot(self) -> tuple[str, str | None]:
        """Return configured and running backend from the same lock state."""
        with self._lock:
            active = self._backend_name if self._process is not None and self._process.is_alive() else None
            return self._backend_name, active

    def switch_backend(self, backend_name: str) -> None:
        """Reload the worker on another backend. A no-op when it is already
        the one running, since the reload costs a model load (~1-6s) and
        silences TTS for its duration."""
        with self._lock:
            if self._active_req_id:
                raise RuntimeError("Cannot change TTS backend during synthesis")
            if backend_name == self._backend_name:
                if self._process is None or not self._process.is_alive():
                    self._load_locked()
                return
            make_backend(backend_name)
            model_name = TTS_MODEL if backend_name == TTS_BACKEND else default_model(backend_name)
            previous = (self._backend_name, self._model_name, self._restart_count)
            logger.info("Switching TTS backend: %s -> %s (%s)",
                        self._backend_name, backend_name, model_name)
            self._shutdown_locked()
            self._backend_name, self._model_name = backend_name, model_name
            self._restart_count = 0
            try:
                self._load_locked()
            except Exception as switch_error:
                self._backend_name, self._model_name, self._restart_count = previous
                try:
                    self._load_locked()
                except Exception as restore_error:
                    raise RuntimeError(
                        f"TTS switch to {backend_name} failed: {switch_error}; "
                        f"previous backend restore failed: {restore_error}"
                    ) from switch_error
                raise RuntimeError(f"TTS switch to {backend_name} failed: {switch_error}; "
                                   f"restored {self._backend_name}") from switch_error

    def load(self) -> None:
        with self._lock:
            if self._active_req_id:
                raise RuntimeError("Cannot reload TTS worker during synthesis")
            self._load_locked()

    def _load_locked(self) -> None:
        self._shutdown_locked()
        ctx = mp.get_context("spawn")
        self._request_q = ctx.Queue()
        self._response_q = ctx.Queue()

        logger.info("Starting TTS worker process...")
        self._process = ctx.Process(
            target=_tts_worker,
            args=(self._request_q, self._response_q, self._backend_name,
                  self._model_name, TTS_SAMPLE_RATE),
            daemon=True,
        )
        try:
            self._process.start()
            msg = self._response_q.get(timeout=300)
            if isinstance(msg, tuple) and msg[0] == "error":
                raise RuntimeError(f"TTS worker failed to load: {msg[1]}")
            if msg != "ready":
                raise RuntimeError(f"TTS worker returned unexpected status: {msg}")
        except Exception:
            self._shutdown_locked()
            raise
        logger.info("TTS worker process ready (pid=%d)", self._process.pid)

    def _check_alive(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        code = self._process.exitcode if self._process else "N/A"
        if self._restart_count >= self._MAX_RESTARTS:
            raise RuntimeError(
                f"TTS worker not running (exitcode={code}), max restarts ({self._MAX_RESTARTS}) exceeded"
            )
        self._restart_count += 1
        logger.warning(
            "TTS worker died (exitcode=%s), restarting (%d/%d)...",
            code, self._restart_count, self._MAX_RESTARTS,
        )
        self._load_locked()

    def start_synthesis(
        self, text: str, voice: str = "Ryan", lang_code: str = "en",
        ref_audio: np.ndarray | None = None, ref_text: str | None = None,
    ) -> int:
        """Send a TTS request to the worker. Returns req_id."""
        with self._lock:
            if self._active_req_id:
                raise RuntimeError("TTS synthesis already active")
            if not text.strip():
                return -1
            self._check_alive()
            self._drain_stale()
            self._req_counter += 1
            req_id = self._req_counter
            self._active_req_id = req_id
            try:
                self._request_q.put((text, voice, lang_code, req_id, ref_audio, ref_text))
            except Exception:
                self._active_req_id = 0
                raise
            return req_id

    def _drain_stale(self) -> None:
        """Drain any leftover responses from previous requests."""
        while not self._response_q.empty():
            try:
                self._response_q.get_nowait()
            except Exception:
                break

    def recv_chunk(self, timeout: float = 5.0) -> tuple[str, np.ndarray | None, int]:
        """Receive next chunk from worker.

        Returns (status, audio_np, sample_rate) where status is 'chunk', 'done', or 'error'.
        Ignores responses from orphaned (old) requests.
        """
        with self._lock:
            if not self._active_req_id:
                raise RuntimeError("No active TTS synthesis")
            if self._process is None or not self._process.is_alive():
                self._active_req_id = 0
                raise RuntimeError("TTS worker died during synthesis")
            try:
                req_id, status, data, sr = self._response_q.get(timeout=timeout)
            except Empty:
                return ("timeout", None, 0)
            if req_id != self._active_req_id:
                logger.debug("Skipping stale TTS response (req %d, active %d)", req_id, self._active_req_id)
                return ("timeout", None, 0)
            if status in (_DONE, _ERROR):
                self._active_req_id = 0
            if status == _CHUNK:
                return ("chunk", data, sr)
            if status in (_DONE, _ERROR):
                return (status, None, 0)
            return ("unknown", None, 0)

    def finish_synthesis(self, req_id: int, cancel: bool = False) -> None:
        """Release a completed request; cancellation discards its busy worker."""
        with self._lock:
            if req_id != self._active_req_id:
                return
            self._active_req_id = 0
            if cancel:
                self._shutdown_locked()

    def synthesize(
        self, text: str, voice: str = "Ryan", lang_code: str = "en",
        ref_audio: np.ndarray | None = None, ref_text: str | None = None,
    ) -> list[tuple[np.ndarray, int]]:
        """Blocking synthesize — collects all chunks."""
        req_id = self.start_synthesis(text, voice, lang_code, ref_audio, ref_text)
        if req_id < 0:
            return []

        chunks = []
        try:
            while True:
                status, audio_np, sr = self.recv_chunk(timeout=5.0)
                if status == "chunk":
                    chunks.append((audio_np, sr))
                elif status == "done":
                    break
                elif status == "error":
                    raise RuntimeError("TTS synthesis failed")
            return chunks
        finally:
            self.finish_synthesis(req_id, cancel=True)

    def shutdown(self) -> None:
        with self._lock:
            self._shutdown_locked()

    def _shutdown_locked(self) -> None:
        process = self._process
        if process is not None:
            if process.is_alive():
                try:
                    self._request_q.put_nowait(None)
                except Exception:
                    pass
                process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        for q in (self._request_q, self._response_q):
            if q is not None:
                q.cancel_join_thread()
                q.close()
        self._process = self._request_q = self._response_q = None
        self._active_req_id = 0


# Singleton
tts_engine = TTSEngine()
