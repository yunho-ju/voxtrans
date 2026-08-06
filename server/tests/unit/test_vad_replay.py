"""T052: replay `pause_ko_400ms.wav` through the production `VoiceActivityDetector`
and check the two VAD silence-hold constants against FR-008's ceiling.

FR-008 (spec.md, amended for constitution v1.1.0) requires the VAD silence
hold to contribute no more than 550ms of latency, while preserving
single-segment behavior across mid-sentence pauses up to 400ms.
`config.py` currently holds `VAD_SILENCE_DURATION_MS` at 1000 and
`MIC_VAD_SILENCE_DURATION_MS` at 1200 -- both exceed the ceiling, which is
what makes `test_vad_silence_duration_ms_is_within_fr008_ceiling` and
`test_mic_vad_silence_duration_ms_is_within_fr008_ceiling` fail today (T054
records this). `MIC_VAD_SILENCE_DURATION_MS` is checked here too, not only
`VAD_SILENCE_DURATION_MS`: the mic/speak-mode path uses the *larger* of the
two holds and is the one on US1's own latency budget (research.md D10), so
a fix that only lowered the listen-mode constant would leave that budget
200ms worse than assumed.

The two `*_yields_exactly_one_segment` tests are the other half of FR-008:
a genuine ~400ms mid-sentence breath pause (see
`test_fixtures.py::test_pause_ko_400ms_contains_one_mid_sentence_silent_run`
for the fixture's own guarantee that its silent run stays below whichever
hold is configured) must not split the sentence into two segments. Both
tests are expected to already pass at today's 1000ms hold -- the pause is
well under it -- and must keep passing once T055 lowers the hold to 550ms;
they exist to prove the lower hold does not accidentally start splitting
sentences (tasks.md T056: if it did, the hold moves up within the ceiling).

**Scope of these two tests, stated precisely (Critic round 1, T057)**: both
cover a <= 8s utterance (`pause_ko_400ms.wav` is 4.31s) replayed under both
production `VoiceActivityDetector` constructions actually used by
`session.py` -- the mic/speak-mode one (`session.py:55`, only
`silence_windows` set, no intermediate split) and the listen one
(`session.py:44-47`, `silence_windows` plus
`intermediate_speech_windows=VAD_INTERMEDIATE_SPEECH_WINDOWS` /
`intermediate_silence_windows=VAD_INTERMEDIATE_SILENCE_WINDOWS`). Neither
test says anything about a listen-mode utterance *longer* than
`VAD_INTERMEDIATE_SPEECH_S` (8s): that is a distinct, pre-existing behavior
(intermediate mid-sentence splitting at `VAD_INTERMEDIATE_SILENCE_MS` =
400ms = 12 windows = 384ms of actual silence, which is *below* FR-008's own
400ms pause-preservation floor) that T055 did not introduce and does not
touch -- see `evidence/red-us3.txt`'s "Pre-existing gap" section.
"""

from pathlib import Path
from typing import Callable

import numpy as np

from src.config import (
    MIC_VAD_SILENCE_DURATION_MS,
    VAD_INTERMEDIATE_SILENCE_WINDOWS,
    VAD_INTERMEDIATE_SPEECH_WINDOWS,
    VAD_SILENCE_DURATION_MS,
)
from src.vad import VoiceActivityDetector

# FR-008 (spec.md:209, constitution v1.1.0): the VAD silence hold must
# contribute no more than 550ms. Deliberately equal to the production
# target, not the earlier, looser 600ms draft ceiling -- a hold of, say,
# 599ms would pass a 600ms ceiling but must fail this one.
FR008_CEILING_MS = 550


def test_pause_ko_400ms_yields_exactly_one_segment_mic_construction(
    vad: VoiceActivityDetector,
    fixtures_dir: Path,
    replay_wav_through_vad: Callable[[VoiceActivityDetector, Path], list[np.ndarray]],
) -> None:
    """`vad` is built the same way `session.py:55` builds `Session.mic_vad`:
    `silence_windows` only, no intermediate split."""
    segments = replay_wav_through_vad(vad, fixtures_dir / "pause_ko_400ms.wav")

    assert len(segments) == 1, (
        f"expected pause_ko_400ms.wav's single mid-sentence pause to stay in one "
        f"segment under the mic-equivalent construction, got {len(segments)} segments"
    )


def test_pause_ko_400ms_yields_exactly_one_segment_listen_construction(
    fixtures_dir: Path,
    vad_factory: Callable[..., VoiceActivityDetector],
    replay_wav_through_vad: Callable[[VoiceActivityDetector, Path], list[np.ndarray]],
) -> None:
    """Critic round 1 (T057) finding: the plain `vad` fixture's default
    construction (no intermediate split) matches `session.py:55`'s mic
    path, but *not* `session.py:44-47`'s listen path, which also sets
    `intermediate_speech_windows`/`intermediate_silence_windows`. A 17s
    input reproduced 1 segment under the (mic-shaped) default construction
    vs. 2 under the real listen construction. `pause_ko_400ms.wav` is only
    4.31s, well under `VAD_INTERMEDIATE_SPEECH_S` (8s), so the intermediate
    split's speech-length gate never arms and this replay is expected to
    still yield exactly one segment -- this test exists to make that an
    observed fact for the listen construction specifically, not an
    assumption carried over from the mic-shaped one above.
    """
    listen_vad = vad_factory(
        intermediate_speech_windows=VAD_INTERMEDIATE_SPEECH_WINDOWS,
        intermediate_silence_windows=VAD_INTERMEDIATE_SILENCE_WINDOWS,
    )
    segments = replay_wav_through_vad(listen_vad, fixtures_dir / "pause_ko_400ms.wav")

    assert len(segments) == 1, (
        f"expected pause_ko_400ms.wav's single mid-sentence pause to stay in one "
        f"segment under the real listen construction (session.py:44-47), got "
        f"{len(segments)} segments"
    )


def test_vad_silence_duration_ms_is_within_fr008_ceiling() -> None:
    assert VAD_SILENCE_DURATION_MS <= FR008_CEILING_MS, (
        f"VAD_SILENCE_DURATION_MS ({VAD_SILENCE_DURATION_MS}ms) must be <= "
        f"{FR008_CEILING_MS}ms (FR-008)"
    )


def test_mic_vad_silence_duration_ms_is_within_fr008_ceiling() -> None:
    assert MIC_VAD_SILENCE_DURATION_MS <= FR008_CEILING_MS, (
        f"MIC_VAD_SILENCE_DURATION_MS ({MIC_VAD_SILENCE_DURATION_MS}ms) must be <= "
        f"{FR008_CEILING_MS}ms (FR-008) -- the speak-mode path uses this larger hold "
        f"and is the one on US1's latency budget (research.md D10)"
    )
