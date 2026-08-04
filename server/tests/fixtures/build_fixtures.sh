#!/usr/bin/env bash
# Builds the fixed audio corpus validated by server/tests/unit/test_fixtures.py
# (tasks.md T005/T006, research.md D3).
#
# Pipeline per clip: `say -v <voice> -r <rate> -o *.aiff <text>` (macOS TTS,
# deterministic, offline, no model weights needed) -> `afconvert -f WAVE
# -d F32@16000 -c 1` (Apple's own sample-rate converter, one step to the
# exact target format: 16 kHz mono 32-bit float WAV). `pause_ko_400ms.wav`
# additionally splices two independently synthesized halves of one sentence
# around an explicit digital-silence gap using the project's own Python
# venv (server/.venv) + soundfile/numpy, because sample-accurate silence
# insertion has no clean equivalent in `say`/`afconvert` alone.
#
# Idempotent: `say` + `afconvert` are deterministic for a fixed
# voice/rate/text triple (verified byte-for-byte across repeated runs), so
# re-running this script regenerates byte-identical files. No network
# access of any kind is used.
#
# Requires macOS voices Yuna (ko_KR) and Samantha (en_US) to be installed.
# Fails loudly and by name if either is missing, rather than silently
# falling back to a different voice or producing a subtly wrong file -- this
# script is the only place fixture generation may depend on which voices
# happen to be installed; server/tests/unit/test_fixtures.py never invokes
# `say` and never checks voice availability.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PYTHON="${SERVER_DIR}/.venv/bin/python"

REQUIRED_VOICES=("Yuna" "Samantha")

fail() {
    echo "build_fixtures.sh: ERROR: $*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || fail "required command '$1' not found on PATH"
}

require_command say
require_command afconvert

for voice in "${REQUIRED_VOICES[@]}"; do
    if ! say -v '?' | awk '{print $1}' | grep -qx "${voice}"; then
        fail "macOS voice '${voice}' is not installed. Install it via" \
             "System Settings > Accessibility > Spoken Content > System Voice" \
             "> Manage Voices (or 'System Voice' on older macOS), then re-run" \
             "this script. Installed voices: $(say -v '?' | awk '{print $1}' | tr '\n' ' ')"
    fi
done

[[ -x "${VENV_PYTHON}" ]] || fail "server virtualenv not found at ${VENV_PYTHON}." \
    "Run T002 (python3.13 -m venv server/.venv && pip install -r requirements.txt" \
    "-r requirements-dev.txt) first."

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

synth() {
    # synth <voice> <rate> <out_aiff> <text>
    local voice="$1" rate="$2" out_aiff="$3" text="$4"
    say -v "${voice}" -r "${rate}" -o "${out_aiff}" "${text}"
}

to_16k_mono_float32_wav() {
    # to_16k_mono_float32_wav <in_aiff> <out_wav>
    afconvert -f WAVE -d F32@16000 -c 1 "$1" "$2"
}

verify_wav() {
    # verify_wav <wav_path> <min_duration_s> <max_duration_s>
    "${VENV_PYTHON}" - "$1" "$2" "$3" <<'PYEOF'
import sys
import soundfile as sf

path, min_s, max_s = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
info = sf.info(path)
problems = []
if info.samplerate != 16000:
    problems.append(f"samplerate {info.samplerate} != 16000")
if info.channels != 1:
    problems.append(f"channels {info.channels} != 1")
if info.subtype != "FLOAT":
    problems.append(f"subtype {info.subtype} != FLOAT")
if not (min_s <= info.duration <= max_s):
    problems.append(f"duration {info.duration:.3f}s not in [{min_s}, {max_s}]")
if problems:
    print(f"build_fixtures.sh: ERROR: {path}: " + "; ".join(problems), file=sys.stderr)
    sys.exit(1)
print(f"{path}: {info.duration:.3f}s, {info.samplerate}Hz, {info.channels}ch, {info.subtype}")
PYEOF
}

# --- utterance_ko_5s.wav: rate tuned so natural Korean speech lands at ~5.0s ---
synth Yuna 190 "${WORK_DIR}/utterance_ko_5s.aiff" \
    "오늘 날씨가 꽤 변덕스러워서 우리는 오후 등산 계획을 일요일로 미루기로 했습니다."
to_16k_mono_float32_wav "${WORK_DIR}/utterance_ko_5s.aiff" "${SCRIPT_DIR}/utterance_ko_5s.wav"
verify_wav "${SCRIPT_DIR}/utterance_ko_5s.wav" 4.9 5.1

# --- utterance_en_5s.wav: rate tuned so natural English speech lands at ~5.0s ---
synth Samantha 248 "${WORK_DIR}/utterance_en_5s.aiff" \
    "The weather has been quite unpredictable this week, so we decided to reschedule our afternoon hiking trip until Sunday."
to_16k_mono_float32_wav "${WORK_DIR}/utterance_en_5s.aiff" "${SCRIPT_DIR}/utterance_en_5s.wav"
verify_wav "${SCRIPT_DIR}/utterance_en_5s.wav" 4.9 5.1

# --- short_150ms.wav: below VAD_MIN_SPEECH_MS (250ms, config.py) ---
synth Yuna 320 "${WORK_DIR}/short_150ms.aiff" "네"
to_16k_mono_float32_wav "${WORK_DIR}/short_150ms.aiff" "${SCRIPT_DIR}/short_150ms.wav"
verify_wav "${SCRIPT_DIR}/short_150ms.wav" 0.05 0.25

# --- warmup_500ms.wav: short clip for startup warm-up inference (research.md D6) ---
synth Samantha 237 "${WORK_DIR}/warmup_500ms.aiff" "Hello there"
to_16k_mono_float32_wav "${WORK_DIR}/warmup_500ms.aiff" "${SCRIPT_DIR}/warmup_500ms.wav"
verify_wav "${SCRIPT_DIR}/warmup_500ms.wav" 0.4 0.6

# --- pause_ko_400ms.wav: one Korean sentence, two halves either side of a ---
# --- genuine ~400ms mid-sentence silence. Natural rate (175) on each half ---
# --- so both are unambiguously loud, real speech for VAD on either side.  ---
synth Yuna 175 "${WORK_DIR}/pause_half1.aiff" "오늘 아침에 일찍 일어났는데"
synth Yuna 175 "${WORK_DIR}/pause_half2.aiff" "생각보다 피곤하지 않았어요"
to_16k_mono_float32_wav "${WORK_DIR}/pause_half1.aiff" "${WORK_DIR}/pause_half1.wav"
to_16k_mono_float32_wav "${WORK_DIR}/pause_half2.aiff" "${WORK_DIR}/pause_half2.wav"

"${VENV_PYTHON}" - "${WORK_DIR}/pause_half1.wav" "${WORK_DIR}/pause_half2.wav" \
    "${SCRIPT_DIR}/pause_ko_400ms.wav" "${SERVER_DIR}" <<'PYEOF'
import sys
import numpy as np
import soundfile as sf
import scipy.io.wavfile as wavfile

half1_path, half2_path, out_path, server_dir = sys.argv[1:5]
sys.path.insert(0, server_dir)
from src.config import VAD_WINDOW_SAMPLES  # noqa: E402

half1, sr1 = sf.read(half1_path, dtype="float32")
half2, sr2 = sf.read(half2_path, dtype="float32")
assert sr1 == sr2 == 16000, (sr1, sr2)

# Critic round 1 finding: synthesizing each half as a standalone `say`
# utterance leaves a natural utterance-final decay on half1's tail and a
# utterance-initial ramp-in on half2's head. Left in place, those stack on
# top of the inserted 400ms of digital silence and the *VAD-visible* quiet
# run balloons past 500ms -- consuming nearly all of research.md D10's
# 4.5-window margin below the 550ms/17-window production hold. Trim each
# half's own near-silent edge, at VAD-window granularity, before splicing,
# so the inserted 400ms *is* the whole gap rather than a floor under it.
#
# Floor: -24dB relative to that half's own median window RMS (same
# reasoning/value as test_fixtures.py's SILENCE_FLOOR_DB -- loud enough to
# exclude literal digital silence and a natural decay/ramp, not so strict it
# eats real speech). Trims land on whole-window boundaries so the inserted
# silence block starts and ends exactly on a VAD window boundary too.
SILENCE_FLOOR_DB = -24.0


def _window_rms(data, window_samples=VAD_WINDOW_SAMPLES):
    n_windows = len(data) // window_samples
    return np.array(
        [
            np.sqrt(np.mean(data[i * window_samples : (i + 1) * window_samples] ** 2))
            for i in range(n_windows)
        ]
    )


def _floor_for(data):
    return float(np.median(_window_rms(data))) * (10 ** (SILENCE_FLOOR_DB / 20))


def trim_trailing_silence(data, window_samples=VAD_WINDOW_SAMPLES):
    floor = _floor_for(data)
    rms = _window_rms(data, window_samples)
    last_speech = next((i for i in range(len(rms) - 1, -1, -1) if rms[i] >= floor), None)
    assert last_speech is not None, "half is entirely below its own silence floor"
    return data[: (last_speech + 1) * window_samples]


def trim_leading_silence(data, window_samples=VAD_WINDOW_SAMPLES):
    floor = _floor_for(data)
    rms = _window_rms(data, window_samples)
    first_speech = next((i for i in range(len(rms)) if rms[i] >= floor), None)
    assert first_speech is not None, "half is entirely below its own silence floor"
    return data[first_speech * window_samples :]


half1_trimmed = trim_trailing_silence(half1)
half2_trimmed = trim_leading_silence(half2)

pause_samples = int(round(0.400 * 16000))
silence = np.zeros(pause_samples, dtype="float32")

spliced = np.concatenate([half1_trimmed, silence, half2_trimmed])
# scipy's WAV writer (not soundfile's) deliberately: libsndfile auto-adds a
# "PEAK" chunk containing a wall-clock write timestamp for float subtypes,
# which made this file differ byte-for-byte between otherwise-identical
# runs. scipy.io.wavfile writes a minimal fmt+fact+data WAV with no such
# metadata, so the output is a pure function of `spliced` and reproducible.
wavfile.write(out_path, 16000, spliced)
PYEOF
verify_wav "${SCRIPT_DIR}/pause_ko_400ms.wav" 3.5 5.5

echo "build_fixtures.sh: OK, all fixtures written to ${SCRIPT_DIR}"
