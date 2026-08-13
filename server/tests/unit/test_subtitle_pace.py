"""Subtitle pace is a per-session setting, entered as numbers.

How eagerly listen mode cuts speech into segments decides how soon a subtitle
can appear: nothing is transcribed until its segment ends. The values arrive
straight from a settings field, so the server clamps rather than trusts them --
a nonsensical entry must produce sane subtitles, not a session that dies
mid-meeting.

Pure conversion and dispatch: no models, no audio.
"""

import pytest

from src.config import (
    SUBTITLE_PACE_DEFAULT,
    SUBTITLE_PACE_MAX_S_RANGE,
    SUBTITLE_PACE_PRESETS,
    SUBTITLE_PACE_SILENCE_MS_RANGE,
    SUBTITLE_PACE_SPEECH_S_RANGE,
    VAD_WINDOW_DURATION_MS,
)
from src.protocol import ConfigureMessage
from src.session import clamp_pace, pace_windows

DEFAULTS = SUBTITLE_PACE_PRESETS[SUBTITLE_PACE_DEFAULT]


def test_a_valid_pace_passes_through_untouched():
    assert clamp_pace(3, 300, 12) == (3, 300, 12)


def test_it_converts_to_window_counts():
    speech_w, silence_w, max_w = pace_windows(8, 400, 30)
    assert speech_w == 8 * 1000 // VAD_WINDOW_DURATION_MS
    assert silence_w == 400 // VAD_WINDOW_DURATION_MS
    assert max_w == 30 * 1000 // VAD_WINDOW_DURATION_MS


def test_the_presets_are_ordered_from_eager_to_patient():
    """"faster" has to mean faster in all three knobs at once; a preset that
    cut sooner but capped later would be incoherent."""
    fast = SUBTITLE_PACE_PRESETS["fast"]
    balanced = SUBTITLE_PACE_PRESETS["balanced"]
    accurate = SUBTITLE_PACE_PRESETS["accurate"]
    for i in range(3):
        assert fast[i] < balanced[i] < accurate[i]


@pytest.mark.parametrize(
    "entered,expected",
    [
        ((0, 300, 12), (SUBTITLE_PACE_SPEECH_S_RANGE[0], 300, 12)),
        ((999, 300, 999), (SUBTITLE_PACE_SPEECH_S_RANGE[1], 300, SUBTITLE_PACE_MAX_S_RANGE[1])),
        ((3, 1, 12), (3, SUBTITLE_PACE_SILENCE_MS_RANGE[0], 12)),
        ((3, 99999, 12), (3, SUBTITLE_PACE_SILENCE_MS_RANGE[1], 12)),
    ],
)
def test_out_of_range_values_are_clamped(entered, expected):
    assert clamp_pace(*entered) == expected


def test_a_cap_below_the_split_threshold_is_raised_to_meet_it():
    """The forced split at `max_s` fires first, so a cap under the split
    threshold would make the threshold dead config that silently does
    nothing."""
    speech, _, maximum = clamp_pace(20, 400, 10)
    assert maximum == speech == 20


@pytest.mark.parametrize("junk", ["abc", None, float("nan")])
def test_unparseable_numbers_fall_back_to_the_default(junk):
    speech, silence, maximum = clamp_pace(junk, junk, junk)
    assert (speech, silence, maximum) == DEFAULTS


def test_configure_message_defaults_to_leaving_the_pace_alone():
    assert ConfigureMessage.from_dict({"type": "configure"}).subtitle_pace is None


def test_configure_message_carries_three_numbers():
    msg = ConfigureMessage.from_dict(
        {"type": "configure", "subtitle_pace": [5, 350, 20]}
    )
    assert msg.subtitle_pace == (5.0, 350.0, 20.0)


@pytest.mark.parametrize(
    "raw", ["fast", [1, 2], [1, 2, 3, 4], {"speech": 3}, ["a", "b", "c"], None]
)
def test_a_malformed_pace_is_read_as_not_supplied(raw):
    """Not supplied means "keep the session's current pace" -- adopting a
    malformed one would change segmentation for the rest of the meeting."""
    msg = ConfigureMessage.from_dict({"type": "configure", "subtitle_pace": raw})
    assert msg.subtitle_pace is None


class _SpyVad:
    def __init__(self) -> None:
        self.paces: list[tuple[int, int, int]] = []

    def set_pace(self, *args: int) -> None:
        self.paces.append(args)


def _session_with_spy_vad():
    """A ClientSession without constructing a real VAD (which loads Silero)."""
    from src.session import ClientSession

    session = ClientSession.__new__(ClientSession)
    session.session_id = "stub"
    session.subtitle_pace = DEFAULTS
    session.vad = _SpyVad()
    session.mode = "listen"
    session.tts_voice = ""
    session.speak_target_language = ""
    session.voice_clone = True
    session.mic_mode = "auto"
    session.translation_prompt = ""
    return session


def test_changing_the_pace_retunes_the_running_detector():
    session = _session_with_spy_vad()

    session.configure("auto", subtitle_pace=(15, 600, 45))

    assert session.subtitle_pace == (15, 600, 45)
    assert session.vad.paces == [pace_windows(15, 600, 45)], (
        "the live detector must be retuned -- rebuilding it would reload the "
        "model and drop whatever is mid-utterance"
    )


def test_reconfiguring_with_the_same_pace_leaves_the_detector_alone():
    session = _session_with_spy_vad()
    session.configure("auto", subtitle_pace=DEFAULTS)
    assert session.vad.paces == []


def test_an_omitted_pace_does_not_reset_a_chosen_one():
    """Other settings change constantly (voice, language). None of them may
    quietly revert the pace."""
    session = _session_with_spy_vad()
    session.configure("auto", subtitle_pace=(15, 600, 45))
    session.vad.paces.clear()

    session.configure("auto", tts_voice="Aiden")

    assert session.subtitle_pace == (15, 600, 45)
    assert session.vad.paces == []


def test_a_clamped_pace_is_stored_clamped():
    """What the session reports must be what the VAD is actually using, or
    the next configure compares against a value that was never in effect."""
    session = _session_with_spy_vad()

    session.configure("auto", subtitle_pace=(0, 1, 999))

    assert session.subtitle_pace == clamp_pace(0, 1, 999)
