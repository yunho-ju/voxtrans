"""T053: replaying `short_150ms.wav` (below `VAD_MIN_SPEECH_MS`) through the
production `VoiceActivityDetector` must yield zero segments, so the caller
never hands this sub-threshold blip on to the ASR model (spec Edge Cases ->
"Empty or sub-threshold utterance"; analyze finding H2).

**What this actually guards (corrected, Critic round 1, T057 minor 2)**:
the short-utterance discard is gated on `VAD_MIN_SPEECH_WINDOWS` (derived
from `VAD_MIN_SPEECH_MS`, `vad.py`'s `flush()` and `_process_window()`), a
constant T055 does not touch and this test does not exercise the value of.
`short_150ms.wav` is short enough that the whole clip fits within a handful
of VAD windows and its speech never reaches `VAD_MIN_SPEECH_WINDOWS`
(~7 windows), so the discard fires regardless of the silence-hold value
in force -- confirmed directly across `silence_windows` 17 (T055's new
listen/mic hold), 31 (the old listen hold), and 37 (an arbitrary value
above both): all three yield zero segments. This test is therefore
invariant to the T055 change, not a regression guard *on* it; it is
included in this task because T055 touches the surrounding module and this
result must keep matching regardless. T054 records today's (pre-change)
passing result as that reference.
"""

from pathlib import Path
from typing import Callable

import numpy as np

from src.vad import VoiceActivityDetector


def test_short_150ms_yields_zero_segments(
    vad: VoiceActivityDetector,
    fixtures_dir: Path,
    replay_wav_through_vad: Callable[[VoiceActivityDetector, Path], list[np.ndarray]],
) -> None:
    segments = replay_wav_through_vad(vad, fixtures_dir / "short_150ms.wav")

    assert segments == [], (
        f"expected short_150ms.wav (below VAD_MIN_SPEECH_MS) to yield zero "
        f"segments, got {len(segments)}"
    )
