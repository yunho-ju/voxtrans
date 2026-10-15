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
PORT = 18765
MAX_SESSIONS = 5

# ASR
# `VOXTRANS_ASR_MODEL` overrides the default (research.md D-ASR, FR-003,
# tasks.md T037b) with the exact same override idiom `TRANSLATION_MODEL`
# below uses: an empty or whitespace-only value is treated as unset (not a
# literal override), and a real override has incidental surrounding
# whitespace stripped. `src/asr.py` continues to import `ASR_MODEL` (not
# this module), so resolution time and propagation match the translation
# path exactly.
_ASR_MODEL_DEFAULT = "mlx-community/Qwen3-ASR-1.7B-8bit"
ASR_MODEL = (os.environ.get("VOXTRANS_ASR_MODEL") or "").strip() or _ASR_MODEL_DEFAULT

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

# Translation prompt template, overridable per session from the app's settings.
# `{lang}` is the target language name; `{text}` is the transcript. A template
# without `{text}` would silently translate nothing, so `TranslationEngine`
# rejects it and falls back to this default rather than emitting empty output.
TRANSLATION_PROMPT_DEFAULT = (
    "Translate to {lang} in a natural, conversational tone. "
    "Do not include emojis or special symbols. "
    "Reply with ONLY the translation.\n\n{text}"
)
TRANSLATION_PROMPT_MAX_LENGTH = 2000

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

# Push-to-talk: the hold *is* the utterance. While the key is held nothing is
# segmented, transcribed, or spoken — the whole hold is buffered and committed
# on release. This is the contract every other PTT translator implements, and
# the reason it matters here is `_handle_mic_audio`'s `tts_playing` gate: with
# mid-hold splitting, TTS for the first half started while the user was still
# talking, and every mic frame after that was *discarded* (not queued), so the
# tail of a paused sentence vanished. Auto (VAD) mode is unchanged.
PTT_MIN_HOLD_S = 0.3   # shorter holds are key-bounce, not speech — dropped
PTT_MAX_HOLD_S = 60    # safety cap for a stuck/forgotten key; commits early

# Subtitle pace: how eagerly listen mode cuts a speaker's audio into segments.
# Every segment costs a full ASR + translation pass, and a subtitle cannot
# appear until its segment ends -- so cutting sooner shows text sooner, at the
# price of translating fragments instead of whole thoughts.
#
# Exposed as three named choices rather than three numbers: they only make
# sense together, and the thing being traded (latency against translation
# quality) is what someone actually has an opinion about.
#
# The three values are sent as numbers; the presets below are starting points
# the app offers as one-click fills, not a closed set of choices.
#
# Bounds exist because these reach the audio path from a client. Anything
# outside them is clamped rather than rejected: a nonsensical value should
# produce sane subtitles, not a session that dies mid-meeting.
SUBTITLE_PACE_SPEECH_S_RANGE = (1, 60)
SUBTITLE_PACE_SILENCE_MS_RANGE = (VAD_WINDOW_DURATION_MS, 3000)
SUBTITLE_PACE_MAX_S_RANGE = (3, 120)

#   (intermediate_speech_s, intermediate_silence_ms, max_speech_s)
SUBTITLE_PACE_PRESETS = {
    # Cuts at the first real breath. Subtitles keep up with a fast speaker;
    # sentences are often split across two of them.
    "fast": (3, 300, 12),
    # The measured default: whole sentences, noticeably later.
    "balanced": (8, 400, 30),
    # Waits for long pauses. Best translations, worst latency -- useful when
    # reading a transcript afterwards matters more than following live.
    "accurate": (15, 600, 45),
}
SUBTITLE_PACE_DEFAULT = "fast"

# Binary WebSocket prefixes
AUDIO_PREFIX_SYSTEM = 0x01
AUDIO_PREFIX_MIC = 0x02
AUDIO_PREFIX_TTS = 0x03
