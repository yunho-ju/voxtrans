import os

SAMPLE_RATE = 16000

# Network chunk: 100ms (what clients send)
CHUNK_DURATION_MS = 100
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_DURATION_MS // 1000  # 1600 samples
CHUNK_BYTES = CHUNK_SAMPLES * 4  # 6400 bytes (float32)

# Silero VAD window: 32ms = 512 samples @ 16kHz (required by model)
VAD_WINDOW_SAMPLES = 512
VAD_WINDOW_DURATION_MS = VAD_WINDOW_SAMPLES * 1000 // SAMPLE_RATE  # 32ms

# VAD thresholds (in VAD windows, not network chunks)
VAD_THRESHOLD = 0.5
VAD_SILENCE_DURATION_MS = 1000
VAD_SILENCE_WINDOWS = VAD_SILENCE_DURATION_MS // VAD_WINDOW_DURATION_MS  # 31 windows at 1000ms / 32ms
VAD_MIN_SPEECH_MS = 250
VAD_MIN_SPEECH_WINDOWS = VAD_MIN_SPEECH_MS // VAD_WINDOW_DURATION_MS  # ~7 windows
VAD_MAX_SPEECH_S = 30
VAD_MAX_SPEECH_WINDOWS = VAD_MAX_SPEECH_S * 1000 // VAD_WINDOW_DURATION_MS  # ~937 windows

# Server
HOST = "0.0.0.0"
PORT = 8765
MAX_SESSIONS = 5

# ASR
ASR_MODEL = "mlx-community/Qwen3-ASR-1.7B-8bit"

# Translation
# `VOXTRANS_TRANSLATION_MODEL` overrides the default (research.md D8,
# FR-003) so the benchmark harness and a future quality decision can select
# a model without a code change. Both an empty value (e.g. an
# accidentally-set `VOXTRANS_TRANSLATION_MODEL=` in a shell profile) and a
# whitespace-only one (Critic round-2 minor 1: non-empty, so a bare `or`
# check alone would treat it as a real override) are treated as unset --
# `.strip()` collapses both to `""`, which then falls through to the
# default via `or`. A real override also has incidental surrounding
# whitespace stripped.
_TRANSLATION_MODEL_DEFAULT = "mlx-community/Qwen3-8B-4bit"
TRANSLATION_MODEL = (os.environ.get("VOXTRANS_TRANSLATION_MODEL") or "").strip() or _TRANSLATION_MODEL_DEFAULT

# TTS
TTS_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
TTS_SAMPLE_RATE = 24000
TTS_DEFAULT_VOICE = "Ryan"

# Intermediate split: for long continuous speech, split at shorter pauses
# After 5s of speech, split at ~200ms silence (natural breath pauses)
VAD_INTERMEDIATE_SPEECH_S = 8
VAD_INTERMEDIATE_SPEECH_WINDOWS = VAD_INTERMEDIATE_SPEECH_S * 1000 // VAD_WINDOW_DURATION_MS  # ~156 windows
VAD_INTERMEDIATE_SILENCE_MS = 400
VAD_INTERMEDIATE_SILENCE_WINDOWS = VAD_INTERMEDIATE_SILENCE_MS // VAD_WINDOW_DURATION_MS  # ~6 windows

# Mic VAD (speak mode) — longer silence tolerance to avoid splitting speech
MIC_VAD_SILENCE_DURATION_MS = 1200
MIC_VAD_SILENCE_WINDOWS = MIC_VAD_SILENCE_DURATION_MS // VAD_WINDOW_DURATION_MS  # ~46 windows

# Binary WebSocket prefixes
AUDIO_PREFIX_SYSTEM = 0x01
AUDIO_PREFIX_MIC = 0x02
AUDIO_PREFIX_TTS = 0x03
