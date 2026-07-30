import logging
import multiprocessing as mp
import time
from multiprocessing import Queue
from pathlib import Path

import numpy as np

from .config import TTS_MODEL, TTS_SAMPLE_RATE

logger = logging.getLogger(__name__)

# Sentinel values for streaming protocol
_CHUNK = "chunk"
_DONE = "done"
_ERROR = "error"

# Pre-generated voice references for preset speakers (16kHz numpy arrays)
_VOICES_DIR = Path(__file__).parent / "voices"


def _load_preset_voices(log) -> dict[str, tuple[np.ndarray, str]]:
    """Load pre-generated voice reference files for preset speakers."""
    voices = {}
    if not _VOICES_DIR.exists():
        log.warning("Voices directory not found: %s", _VOICES_DIR)
        return voices
    for npy_file in _VOICES_DIR.glob("*.npy"):
        name = npy_file.stem.lower()
        txt_file = npy_file.with_suffix(".txt")
        if txt_file.exists():
            audio = np.load(npy_file)
            ref_text = txt_file.read_text().strip()
            voices[name] = (audio, ref_text)
            log.info("Loaded preset voice '%s': %d samples, ref='%s'",
                     name, audio.size, ref_text[:50])
    return voices


def _tts_worker(request_q: Queue, response_q: Queue, model_name: str, sample_rate: int) -> None:
    """Dedicated TTS process — loads model and processes requests."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] TTS-Worker: %(message)s",
    )
    log = logging.getLogger("tts_worker")

    try:
        from mlx_audio.tts.utils import load
        import mlx.core as mx

        log.info("Loading model %s", model_name)
        t0 = time.time()
        model = load(model_name)
        log.info("Model loaded in %.1fs", time.time() - t0)

        # Load preset voice references
        preset_voices = _load_preset_voices(log)
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

                # Cap max_tokens by text length to prevent runaway generation
                # ~12.5 codec tokens/sec, ~3 words/sec → ~4 tokens per character
                max_tok = min(2048, max(75, len(text) * 4))

                # Build generate kwargs
                gen_kwargs = dict(
                    lang_code=lang_code,
                    stream=True,
                    streaming_interval=0.5,
                    max_tokens=max_tok,
                    temperature=0.3,
                    top_p=0.8,
                )

                if is_voice_clone:
                    # Voice clone mode: use user's mic audio as reference
                    from scipy.signal import resample as scipy_resample
                    num_samples_24k = int(len(ref_audio_np) * sample_rate / 16000)
                    ref_resampled = scipy_resample(ref_audio_np, num_samples_24k).astype(np.float32)
                    ref_audio_mx = mx.array(ref_resampled)
                    gen_kwargs["ref_audio"] = ref_audio_mx
                    if ref_text:
                        gen_kwargs["ref_text"] = ref_text
                    log.info("  voice clone: %d -> %d samples (16kHz -> 24kHz)",
                             len(ref_audio_np), num_samples_24k)
                else:
                    # Preset mode: use pre-generated voice reference via ICL
                    voice_key = voice.lower()
                    if voice_key in preset_voices:
                        preset_audio, preset_text = preset_voices[voice_key]
                        from scipy.signal import resample as scipy_resample
                        num_samples_24k = int(len(preset_audio) * sample_rate / 16000)
                        ref_resampled = scipy_resample(preset_audio, num_samples_24k).astype(np.float32)
                        gen_kwargs["ref_audio"] = mx.array(ref_resampled)
                        gen_kwargs["ref_text"] = preset_text
                        log.info("  preset voice '%s' via ICL ref (%d samples)",
                                 voice_key, preset_audio.size)
                    else:
                        # Fallback: no voice conditioning
                        gen_kwargs["voice"] = voice
                        log.warning("  preset voice '%s' not found, using unconditioned", voice_key)

                # Generate and send chunks immediately
                for result in model.generate(text, **gen_kwargs):
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
                del result
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

    def load(self) -> None:
        ctx = mp.get_context("spawn")
        self._request_q = ctx.Queue()
        self._response_q = ctx.Queue()

        logger.info("Starting TTS worker process...")
        self._process = ctx.Process(
            target=_tts_worker,
            args=(self._request_q, self._response_q, TTS_MODEL, TTS_SAMPLE_RATE),
            daemon=True,
        )
        self._process.start()

        # Wait for model to load
        msg = self._response_q.get(timeout=300)
        if isinstance(msg, tuple) and msg[0] == "error":
            raise RuntimeError(f"TTS worker failed to load: {msg[1]}")
        if msg != "ready":
            raise RuntimeError(f"TTS worker returned unexpected status: {msg}")
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
        self.load()

    def start_synthesis(
        self, text: str, voice: str = "Ryan", lang_code: str = "en",
        ref_audio: np.ndarray | None = None, ref_text: str | None = None,
    ) -> int:
        """Send a TTS request to the worker. Returns req_id."""
        self._check_alive()
        if not text.strip():
            return -1

        # Drain any orphaned responses from previous requests
        self._drain_stale()

        self._req_counter += 1
        req_id = self._req_counter
        self._active_req_id = req_id
        self._request_q.put((text, voice, lang_code, req_id, ref_audio, ref_text))
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
        self._check_alive()
        try:
            req_id, status, data, sr = self._response_q.get(timeout=timeout)
        except Exception:
            return ("timeout", None, 0)

        # Skip responses from old/orphaned requests
        if req_id != getattr(self, '_active_req_id', req_id):
            logger.debug("Skipping stale TTS response (req %d, active %d)", req_id, self._active_req_id)
            return ("timeout", None, 0)

        if status == _CHUNK:
            return ("chunk", data, sr)
        elif status == _DONE:
            return ("done", None, 0)
        elif status == _ERROR:
            return ("error", None, 0)
        return ("unknown", None, 0)

    def synthesize(
        self, text: str, voice: str = "Ryan", lang_code: str = "en",
        ref_audio: np.ndarray | None = None, ref_text: str | None = None,
    ) -> list[tuple[np.ndarray, int]]:
        """Blocking synthesize — collects all chunks."""
        req_id = self.start_synthesis(text, voice, lang_code, ref_audio, ref_text)
        if req_id < 0:
            return []

        chunks = []
        while True:
            self._check_alive()
            status, audio_np, sr = self.recv_chunk(timeout=5.0)
            if status == "chunk":
                chunks.append((audio_np, sr))
            elif status == "done":
                break
            elif status == "error":
                raise RuntimeError("TTS synthesis failed")
            # timeout: keep waiting
        return chunks

    def shutdown(self) -> None:
        if self._request_q and self._process and self._process.is_alive():
            self._request_q.put(None)
            self._process.join(timeout=5)


# Singleton
tts_engine = TTSEngine()
