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
# T055 (research.md D10, FR-008): 1000 -> 550ms, meeting FR-008's own
# <=550ms segmentation-boundary ceiling (spec.md:209, constitution v1.1.0)
# exactly, and sitting 150ms above the 400ms breath-pause fixture
# (pause_ko_400ms.wav) that must stay in one segment (17 windows vs 12.5)
# -- validated by corpus replay for utterances up to VAD_INTERMEDIATE_SPEECH_S
# (8s; tests/unit/test_vad_replay.py), not by argument. Longer listen-mode
# utterances are additionally gated by VAD_INTERMEDIATE_SILENCE_WINDOWS
# below, not this constant -- see that constant's own comment for a known,
# pre-existing gap against the same 400ms floor (evidence/red-us3.txt).
VAD_SILENCE_DURATION_MS = 550
VAD_SILENCE_WINDOWS = VAD_SILENCE_DURATION_MS // VAD_WINDOW_DURATION_MS  # 17 windows at 550ms / 32ms
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
# T035: user-accepted at T034 (evidence/model-comparison.md, decision
# 2026-08-06T09:31:56+0900) — 4B measured 806 ms vs the incumbent 8B's
# 1361 ms; the user selected 4B over 2B and reject-both after reviewing
# the twenty side-by-side eval outputs.
_TRANSLATION_MODEL_DEFAULT = "mlx-community/Qwen3.5-4B-MLX-4bit"
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
# Pre-existing gap, found but deliberately not fixed by T052-T056 (Critic
# round 1, T057): 400ms // 32ms floors to 12 windows = 384ms of actually
# measured silence, which is below FR-008's own 400ms mid-sentence-pause
# preservation floor for listen-mode utterances >= VAD_INTERMEDIATE_SPEECH_S
# (8s) long. See evidence/red-us3.txt's "Pre-existing gap" section --
# out of scope here; the same lever US5/real-usage feedback would tune.

# Mic VAD (speak mode) — T055 (research.md D10, FR-008): 1200 -> 550ms. The
# speak-mode path used the *larger* of the two holds and is the one on US1's
# own latency budget, so lowering only VAD_SILENCE_DURATION_MS would have
# left it 200ms worse than budgeted; both move together.
MIC_VAD_SILENCE_DURATION_MS = 550
MIC_VAD_SILENCE_WINDOWS = MIC_VAD_SILENCE_DURATION_MS // VAD_WINDOW_DURATION_MS  # 17 windows at 550ms / 32ms

# Binary WebSocket prefixes
AUDIO_PREFIX_SYSTEM = 0x01
AUDIO_PREFIX_MIC = 0x02
AUDIO_PREFIX_TTS = 0x03
