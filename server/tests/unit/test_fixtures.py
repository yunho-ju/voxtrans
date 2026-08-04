"""Validates the fixed audio corpus built by server/tests/fixtures/build_fixtures.sh (T006).

Covers: every fixture file exists, is 16 kHz mono 32-bit float WAV, and has the
duration its name promises within +/-100 ms (research.md D3, tasks.md T006).
`pause_ko_400ms.wav` additionally must contain a genuinely silent run at
VAD-window granularity that is not at the very start or end of the file —
that is the load-bearing property the VAD-hold replay test (T052) depends on.

Hermetic: everything here reads only the checked-in fixture files under
`fixtures_dir` (server/tests/fixtures/). No test invokes `say`, no test cares
which macOS voices are installed on the machine running the suite — voice
availability is entirely the build script's problem (see build_fixtures.sh),
not this test's. That is deliberate: a missing voice must fail loudly and
actionably from the script that needs it, not confusingly from a test that
merely reads pre-built files.

Silence detection here is a small, independent windowed-RMS scan written
directly in this file — it does not import or call anything from `src/vad`
(no Silero, no VoiceActivityDetector), so this test never gates on the
mechanism T052 verifies against this same fixture. It *does* import a few
plain numeric constants from `src/config` (window size, sample rate, the
production silence-hold window count) — those are shared data, not
mechanism, the same pattern `test_conftest_contract.py` already uses for
`SAMPLE_RATE`.
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.config import SAMPLE_RATE, VAD_SILENCE_WINDOWS, VAD_WINDOW_SAMPLES

# Expected total duration per fixture, in seconds. For pause_ko_400ms.wav
# this is the *total file* duration (two speech halves + the mid-sentence
# pause), not the 400 ms pause alone -- the pause itself is checked
# separately by test_pause_ko_400ms_contains_one_mid_sentence_silent_run.
EXPECTED_DURATIONS_S = {
    "utterance_ko_5s.wav": 5.0,
    "utterance_en_5s.wav": 5.0,
    "pause_ko_400ms.wav": 4.311,
    "short_150ms.wav": 0.15,
    "warmup_500ms.wav": 0.5,
}

DURATION_TOLERANCE_S = 0.1

# short_150ms.wav must be shorter than VAD_MIN_SPEECH_MS (250 ms, config.py)
# -- this is the property the fixture exists to provide, independent of the
# +/-100ms tolerance check above.
VAD_MIN_SPEECH_S = 0.25

WINDOW_MS = VAD_WINDOW_SAMPLES * 1000 // SAMPLE_RATE  # 32ms, matches vad.py's window

# The pause fixture must be judged at the same granularity the production
# VAD actually runs at: consecutive, non-overlapping VAD_WINDOW_SAMPLES
# (512-sample / 32ms) windows counted from stream start (vad.py:98-100),
# not arbitrary 10ms amplitude frames -- a run that looks silent under a
# fine-grained amplitude scan can still span too many *VAD windows* to stay
# under the production hold (config.py VAD_SILENCE_WINDOWS), which is
# exactly the property this fixture exists to provide for T052.
#
# The floor is expressed relative to the file's own measured "typical
# window loudness" (median window RMS -- robust to the handful of quiet
# windows themselves, since most windows in a ~4.4s two-clause recording
# are speech), not as a bare absolute magnitude: -24dB below that median is
# loud enough to exclude literal digital silence and a natural utterance's
# trailing decay/leading ramp, but not so strict that it starts admitting
# actual speech as "silence". This mirrors research.md D10's own reasoning
# for the 550ms/17-window production hold: pick a number that demonstrably
# keeps one genuine mid-sentence pause below the threshold, and verify by
# measurement, not by argument.
SILENCE_FLOOR_DB = -24.0

# Acceptance band for the longest silent *window* run, in windows (not ms):
# >= 10 windows (320ms) means there is a genuine pause, not an accidentally
# near-erased one; <= 13 windows (416ms) keeps real margin (>=4 windows /
# 128ms) below VAD_SILENCE_WINDOWS (17 windows / 550ms, config.py) -- the
# threshold that actually flushes a segment in vad.py:151. Critic round 1
# found the original fixture measured ~16-18 windows here (consuming almost
# all of the 4.5-window margin research.md D10 assumed), because each half
# was synthesized as a standalone utterance and carried its own
# utterance-final decay / utterance-initial ramp on top of the inserted
# 400ms of digital silence.
PAUSE_SILENCE_RUN_MIN_WINDOWS = 10
PAUSE_SILENCE_RUN_MAX_WINDOWS = 13


def _load(path: Path) -> tuple[np.ndarray, int]:
    data, samplerate = sf.read(path, dtype="float32", always_2d=False)
    return data, samplerate


def _window_rms(data: np.ndarray, window_samples: int) -> np.ndarray:
    n_windows = len(data) // window_samples
    return np.array(
        [
            np.sqrt(np.mean(data[i * window_samples : (i + 1) * window_samples] ** 2))
            for i in range(n_windows)
        ]
    )


def _longest_silent_window_run(data: np.ndarray, samplerate: int) -> tuple[int, int, int]:
    """Return (run_length_windows, run_start_window, total_windows) for the
    longest contiguous run of VAD_WINDOW_SAMPLES windows whose RMS falls
    below SILENCE_FLOOR_DB relative to this file's own median window RMS.
    """
    assert samplerate == SAMPLE_RATE, f"expected {SAMPLE_RATE}Hz, got {samplerate}"
    rms = _window_rms(data, VAD_WINDOW_SAMPLES)
    floor = float(np.median(rms)) * (10 ** (SILENCE_FLOOR_DB / 20))

    best_len, best_start = 0, 0
    run_len, run_start = 0, 0
    for i, window_rms_value in enumerate(rms):
        if window_rms_value < floor:
            if run_len == 0:
                run_start = i
            run_len += 1
            if run_len > best_len:
                best_len, best_start = run_len, run_start
        else:
            run_len = 0

    return best_len, best_start, len(rms)


@pytest.mark.parametrize("filename", sorted(EXPECTED_DURATIONS_S))
def test_fixture_exists(fixtures_dir: Path, filename: str) -> None:
    assert (fixtures_dir / filename).is_file(), f"missing fixture: {filename}"


@pytest.mark.parametrize("filename", sorted(EXPECTED_DURATIONS_S))
def test_fixture_is_16k_mono_float32(fixtures_dir: Path, filename: str) -> None:
    info = sf.info(fixtures_dir / filename)
    assert info.samplerate == 16000, f"{filename}: expected 16000 Hz, got {info.samplerate}"
    assert info.channels == 1, f"{filename}: expected mono, got {info.channels} channels"
    assert info.subtype == "FLOAT", f"{filename}: expected 32-bit float, got {info.subtype}"


@pytest.mark.parametrize("filename", sorted(EXPECTED_DURATIONS_S))
def test_fixture_duration_within_tolerance(fixtures_dir: Path, filename: str) -> None:
    info = sf.info(fixtures_dir / filename)
    expected = EXPECTED_DURATIONS_S[filename]
    assert abs(info.duration - expected) <= DURATION_TOLERANCE_S, (
        f"{filename}: expected {expected}s +/-{DURATION_TOLERANCE_S}s, got {info.duration}s"
    )


def test_short_150ms_is_below_vad_min_speech_threshold(fixtures_dir: Path) -> None:
    info = sf.info(fixtures_dir / "short_150ms.wav")
    assert info.duration < VAD_MIN_SPEECH_S, (
        f"short_150ms.wav must be shorter than VAD_MIN_SPEECH_MS "
        f"({VAD_MIN_SPEECH_S}s), got {info.duration}s"
    )


def test_pause_ko_400ms_contains_one_mid_sentence_silent_run(fixtures_dir: Path) -> None:
    data, samplerate = _load(fixtures_dir / "pause_ko_400ms.wav")
    run_windows, run_start_window, total_windows = _longest_silent_window_run(data, samplerate)
    run_ms = run_windows * WINDOW_MS

    assert PAUSE_SILENCE_RUN_MIN_WINDOWS <= run_windows <= PAUSE_SILENCE_RUN_MAX_WINDOWS, (
        f"pause_ko_400ms.wav: expected a {PAUSE_SILENCE_RUN_MIN_WINDOWS}-"
        f"{PAUSE_SILENCE_RUN_MAX_WINDOWS}-window ({PAUSE_SILENCE_RUN_MIN_WINDOWS * WINDOW_MS}-"
        f"{PAUSE_SILENCE_RUN_MAX_WINDOWS * WINDOW_MS}ms) silent run at VAD-window granularity, "
        f"longest run found was {run_windows} windows ({run_ms}ms)"
    )
    assert run_windows < VAD_SILENCE_WINDOWS, (
        f"pause_ko_400ms.wav: silent run of {run_windows} windows must stay below "
        f"VAD_SILENCE_WINDOWS ({VAD_SILENCE_WINDOWS}, config.py) or T052's replay would flush "
        f"mid-sentence"
    )
    assert run_start_window > 0, "silent run must not start at the very beginning of the file"
    assert run_start_window + run_windows < total_windows, (
        "silent run must not extend to the very end of the file"
    )
