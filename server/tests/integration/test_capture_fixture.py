"""Validates the PortAudio capture fixture against real hardware (T004).

T003's fixture is worthless unless it can tell silence from sound, *and*
tell one signal's content from another's: if it cannot, every audio test
built on top of it passes unconditionally and proves nothing. Feature 001
needed five Gauntlet rounds for the silence-vs-sound version of this
mistake; this feature needed two more Critic rounds to find the
content-vs-content version, because the round-1 capture path (`ffmpeg -f
avfoundation`) spliced ~20% of every recording, which collapsed the
matched filter's entire dynamic range without ever showing up as a
round-1-visible defect (see conftest.py's module docstring and
`specs/002-audio-routing/evidence/T003-T004.md` "Critic round 2 fixes").

This module:

- `test_idle_loopback_is_near_silent` — nothing feeding `BlackHole 2ch`
  reads ~0 energy.
- `test_tone_played_into_loopback_is_recovered` — a known tone played into
  `BlackHole 2ch` is recoverable: matched-filter content match, an audible
  level, and the right FFT frequency.
- `test_speech_played_into_loopback_is_recovered` — a real, natural-
  amplitude TTS-scale speech fixture is recoverable: matched-filter content
  match and an audible level. The signal class SC-001 actually cares about,
  not a full-scale sine (round 1 finding 1: a sine is ~12-13dB louder than
  real speech).
- `test_speech_content_does_not_match_a_different_utterance_same_voice` —
  the **hard negative control** round 2 required checked into the suite: a
  different English sentence, same voice (`say -v Samantha`), similar
  length, must score *below* the match threshold against the true
  reference. The round-1 negative control (different speaker *and*
  different language) was the easy case and gave a false sense of
  separation — in round 2, the wrong utterance outscored the true one in
  2 of 8 trials against the spliced capture.
- Hermetic (no hardware) tests of the scorer functions themselves live at
  the bottom of this module: `_matched_filter_score` and
  `_peak_frequency_hz` on synthetic inputs with known answers. Nothing
  previously asked whether the scorer scores (round 2 finding 3); a mutant
  that hardcodes the score to a constant must be caught by *something*,
  and it should not require real hardware to catch it.

Content matching is normalized cross-correlation (a matched filter)
against the *known* reference — gain-independent by construction (both
signals are normalized by their own norm before the dot product). A scorer
tied to one absolute signal level would break the moment that level
changes, and `BlackHole 2ch`'s output volume genuinely did change between
Critic rounds on this machine — deliberately, by a person repairing the
machine outside this suite, not "in the wild" as an earlier version of
this note speculated (Critic round 3 finding 4 corrected that inference;
see evidence/T003-T004.md). What round 1 and round 2 *did* verify directly
is that the scorer's gain-independence holds across a wide amplitude range
regardless of cause: round 1 measured a stable score across amplitudes
1.0/0.5/0.25 (~0.16-0.18); round 2's Critic measured it stable across a
1000x amplitude range (0.137-0.209).

Content matching alone is *not* a level check (round 2 finding 2): a
matched filter is deliberately scale-invariant, so it cannot by itself
distinguish "audible speech through a working path" from "the same speech
1000x too quiet through a broken one" — both have identical shape.
`_assert_recorded_level_is_plausible` closes this by comparing the
recording's RMS to a level predicted from the *canonical* (pristine,
independently re-read) reference RMS and the device's measured
`kAudioDevicePropertyVolumeScalar`-derived gain (`conftest.blackhole_expected_gain`)
— anchored to hardware state, not an absolute constant, and structurally
decoupled from whatever array actually gets played, so a mutation that
attenuates *what is emitted* cannot also move *what is expected* (see that
function's docstring for the mutation-resistance argument and Critic round
2's exact reproduction of the defect it fixes).

Playing the reference requires writing audio to a specific *output* device
without depending on (or changing) the system default output. `ffmpeg`'s
`avfoundation` support is capture-only on this machine — confirmed via
`ffmpeg -h muxer=avfoundation` → "Unknown format 'avfoundation'" — so
playback uses `sounddevice` (PortAudio), the same library conftest.py now
uses for capture.

Concurrency: audio devices are shared, stateful hardware, so no other test
recording or playing may run while these do. `_exclusive_device_access`
wraps every hardware-backed test in this module in the interprocess lock
conftest.py exposes (`audio_device_lock`), held for the test's full
duration — covering both the reference-playing thread and the capture
together, not just the capture.
"""

import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import sounddevice as sd
from scipy.signal import correlate, resample

from integration import conftest

BLACKHOLE = "BlackHole 2ch"

SAMPLE_RATE = 48_000
CAPTURE_DURATION_S = 2.0
# Observed avfoundation/PortAudio startup latency into BlackHole is up to
# ~1s before a steady stream is flowing; starting the capture before that
# would record leading silence and could understate the reference's
# presence.
LEAD_IN_S = 1.0

TONE_FREQUENCY_HZ = 1000.0
TONE_AMPLITUDE = 1.0
TONE_DURATION_S = 4.0

_FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
SPEECH_FIXTURE = _FIXTURES_DIR / "utterance_en_5s.wav"
# Same voice (`say -v Samantha -r 248`) and a similar length (~5.3s vs
# ~4.9s) as SPEECH_FIXTURE, but a different sentence -- the hard negative
# control. See specs/002-audio-routing/evidence/T003-T004.md "Critic round
# 2 fixes" for the exact `say` command used to generate it.
HARD_NEGATIVE_SPEECH_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "hard_negative_en.wav"

# Idle BlackHole capture measured exactly 0.0 RMS across every trial (no
# hardware noise floor — it is a virtual loopback), so a small absolute
# floor is a reliable, gain-independent silence check: 0 * any gain is
# still 0.
SILENT_RMS_MAX = 1e-4

# Matched-filter (normalized cross-correlation against the known reference)
# threshold for "this recording contains the reference content". Measured
# directly against real hardware with the PortAudio capture path (n=5 true
# matches for speech, n=5 for tone; n=5 hard-negative same-voice, n=3
# reversed-reference, n=3 different-language, n=3 white-noise -- see
# evidence/T003-T004.md "Critic round 2 fixes"):
#   true matches (speech):  0.9433 - 0.9453
#   true matches (tone):    0.9564 - 0.9785
#   hard negative (same voice, different sentence): 0.1969 - 0.2023
#   reversed reference:     0.2186 - 0.2189
#   different language:     0.0458 - 0.0465
#   white noise:             0.0236 - 0.0238
# 0.5 sits with >1.8x margin below every true match and >2.2x margin above
# every negative control observed, including the hard one.
MATCH_SCORE_MIN = 0.5

# FFT peak recovered for a 1000 Hz tone through the PortAudio capture path
# measured exactly 1000.00 Hz (0.00 Hz error) across 5 trials -- the
# splice-induced smearing that required a 20-30 Hz tolerance under the old
# ffmpeg path is gone (nothing is spliced). A tolerance of 5 Hz (roughly
# 10x a 2.0s capture window's ~0.5 Hz bin resolution) still leaves margin
# for a shorter capture window or a noisier trial without being
# meaningless.
FREQUENCY_TOLERANCE_HZ = 5.0

# Recorded-to-predicted RMS ratio floor for the level check
# (`_assert_recorded_level_is_plausible`). Measured directly: real
# (working) trials landed at 0.87-0.98 of the level predicted from the
# canonical reference RMS and the measured device gain (speech: 0.872-0.874;
# tone: 0.956-0.978). A simulated 60dB (0.001x) source-level defect — using
# the *same* protected prediction, decoupled from what was actually played —
# measured a ratio of 0.00087, more than 500x below 0.5. See
# evidence/T003-T004.md "Critic round 2 fixes".
LEVEL_RATIO_MIN = 0.5


@pytest.fixture(autouse=True)
def _exclusive_device_access(audio_device_lock):
    """Serialize every test in this module against other device users.

    See module docstring "Concurrency". Function-scoped so the lock is held
    only for the duration of each individual test, not the whole session.
    """
    with audio_device_lock():
        yield


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


def _peak_frequency_hz(samples: np.ndarray, sample_rate: int) -> float:
    windowed = samples * np.hanning(len(samples))
    spectrum = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(len(samples), d=1.0 / sample_rate)
    return float(freqs[int(np.argmax(np.abs(spectrum)))])


def _matched_filter_score(reference: np.ndarray, recorded: np.ndarray) -> float:
    """Best-lag normalized cross-correlation between `recorded` and
    `reference`, both float64 mono at the same sample rate. ~1.0 (or ~-1.0:
    absolute value is what matters, see below) is a scaled match; ~0 is
    unrelated content or silence. Gain-independent: scaling `recorded` by
    any positive constant leaves the score unchanged, because both signals
    are normalized by their own norms before the dot product.

    Takes the absolute value at the call site: a clean capture can still
    land its best-magnitude alignment on a phase-inverted lag depending on
    where the correlation search starts (observed directly in round 1 under
    the (since-removed) splicing capture path). The *strength* of the
    match, not its sign, is what distinguishes "this content is present"
    from "it is not".
    """
    reference = reference - reference.mean()
    recorded = recorded - recorded.mean()
    if len(recorded) > len(reference):
        reference, recorded = recorded, reference
    if np.allclose(recorded, 0.0) or np.allclose(reference, 0.0):
        return 0.0
    xcorr = correlate(reference, recorded, mode="valid")
    best_lag = int(np.argmax(np.abs(xcorr)))
    reference_segment = reference[best_lag : best_lag + len(recorded)]
    denominator = np.linalg.norm(reference_segment) * np.linalg.norm(recorded)
    if denominator == 0:
        return 0.0
    return float(np.dot(reference_segment, recorded) / denominator)


def _assert_recorded_level_is_plausible(recorded_mono: np.ndarray, canonical_emitted_rms: float) -> None:
    """SC-001 requires *audible* speech, not merely shape-matching speech at
    any level (Critic round 2 finding 2): a matched-filter score alone
    cannot distinguish a working path from one delivering the same content
    1000x too quiet. This anchors an absolute floor to *measured* hardware
    state (the device's own volume scalar, converted through
    `conftest.blackhole_expected_gain`) rather than a constant tied to one
    slider position (round 1 finding 2) or nothing at all (round 2 finding 2).

    Skips (does not fail) if the volume scalar cannot be read — that
    reflects what this device/scope implements, not a defect under test.

    `canonical_emitted_rms` must be computed from the reference *before* it
    is handed to whatever plays it, from a source a mutation of the
    playback path cannot also move (e.g. re-reading the fixture file
    independently, or a fixed synthesis-parameter constant) — never from
    the array actually passed to `sd.play()`. Otherwise this check
    degrades to the same scale-invariant blind spot as the content check:
    if "expected" and "actual" both track the same (possibly attenuated)
    array, their ratio stays constant at any scale and the defect this
    exists to catch becomes invisible again. Confirmed directly: emitting
    the reference at 0.001x while still predicting from the untouched,
    independently-read canonical reference measured a ratio of 0.00087 —
    more than 500x below `LEVEL_RATIO_MIN` — whereas computing the
    prediction from the same attenuated array would reproduce round 2's
    exact blind spot.
    """
    volume_scalar = conftest.read_core_audio_volume_scalar(BLACKHOLE, "output")
    if volume_scalar is None:
        pytest.skip(f"cannot read {BLACKHOLE!r}'s output volume scalar on this machine")
    expected_gain = conftest.blackhole_expected_gain(volume_scalar)
    predicted_rms = canonical_emitted_rms * expected_gain
    recorded_rms = _rms(recorded_mono)
    minimum_acceptable_rms = predicted_rms * LEVEL_RATIO_MIN
    assert recorded_rms >= minimum_acceptable_rms, (
        f"recorded level implausibly low: rms={recorded_rms:.6g}, expected >= "
        f"{minimum_acceptable_rms:.6g} ({LEVEL_RATIO_MIN}x of {predicted_rms:.6g} "
        f"predicted from canonical emitted rms {canonical_emitted_rms:.6g} x "
        f"measured gain {expected_gain:.6g} [volume scalar {volume_scalar}])"
    )


def _play_reference_and_capture(record_input_device, reference: np.ndarray, reference_sample_rate: int):
    """Play `reference` (mono) into BlackHole's output side via PortAudio
    while recording its input side via the T003 fixture. Returns the
    recorded `CaptureResult`. Skips if BlackHole has no PortAudio-visible
    output side (same ambient-state rule as the capture fixture itself).

    Propagates a failure in the playback thread onto the test (Critic
    round 3 finding 1): `sd.play()` runs on a background thread so the
    capture can proceed concurrently, but a bare background thread that
    raises just prints a traceback and vanishes — the calling test would
    see a normal-looking (silent) capture and no failure at all. Both an
    exception raised inside `_play` and a playback thread that is still
    running after its join timeout (stuck, rather than merely slow) are
    raised here instead.
    """
    output_index = conftest._resolve_portaudio_output_index(BLACKHOLE)
    if output_index is None:
        pytest.skip(f"{BLACKHOLE!r} has no PortAudio-visible output side")

    stereo = np.column_stack([reference, reference]).astype(np.float32)
    play_exception: list[BaseException] = []

    def _play() -> None:
        try:
            sd.play(stereo, samplerate=reference_sample_rate, device=output_index, blocking=True)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the test thread below
            play_exception.append(exc)

    player = threading.Thread(target=_play)
    player.start()
    join_timeout = (len(reference) / reference_sample_rate) + 2.0
    try:
        time.sleep(LEAD_IN_S)
        result = record_input_device(BLACKHOLE, CAPTURE_DURATION_S)
    finally:
        player.join(timeout=join_timeout)

    if player.is_alive():
        raise RuntimeError(
            f"reference playback thread did not finish within {join_timeout:.1f}s (sd.play appears stuck)"
        )
    if play_exception:
        raise RuntimeError(f"reference playback failed: {play_exception[0]!r}") from play_exception[0]

    return result


def test_idle_loopback_is_near_silent(record_input_device):
    """Nothing is feeding BlackHole, so its input side should read ~0 energy."""
    result = record_input_device(BLACKHOLE, CAPTURE_DURATION_S)
    mono = result.samples.mean(axis=1)

    rms = _rms(mono)

    assert rms < SILENT_RMS_MAX, f"expected near-silence from idle {BLACKHOLE!r}, got rms={rms}"


def _synthesize_tone() -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * TONE_DURATION_S)) / SAMPLE_RATE
    return (TONE_AMPLITUDE * np.sin(2 * np.pi * TONE_FREQUENCY_HZ * t)).astype(np.float32)


def test_tone_played_into_loopback_is_recovered(record_input_device):
    """A real tone fed to BlackHole's output side must show up: matching content, at the right frequency.

    Content-only (Critic round 3 finding 2): the audible-level claim for
    the same setup is `test_tone_level_is_plausible`, a separate test, so
    that a machine/driver where the volume-scalar property becomes
    unreadable can only ever skip *that* narrow claim — it must never be
    able to swallow this one, which is the proof that the fixture can hear
    the tone's content at all.
    """
    tone_to_play = _synthesize_tone()

    result = _play_reference_and_capture(record_input_device, tone_to_play, SAMPLE_RATE)

    mono = result.samples.mean(axis=1).astype(np.float64)

    score = abs(_matched_filter_score(tone_to_play.astype(np.float64), mono))
    assert score > MATCH_SCORE_MIN, (
        f"expected the recording to match the emitted tone (score > {MATCH_SCORE_MIN}), got {score:.4f}"
    )

    peak_hz = _peak_frequency_hz(mono, result.sample_rate)
    assert abs(peak_hz - TONE_FREQUENCY_HZ) <= FREQUENCY_TOLERANCE_HZ, (
        f"expected FFT peak within {FREQUENCY_TOLERANCE_HZ} Hz of {TONE_FREQUENCY_HZ} Hz, got {peak_hz} Hz"
    )


def test_tone_level_is_plausible(record_input_device):
    """Level-only companion to `test_tone_played_into_loopback_is_recovered`
    (Critic round 3 finding 2). Skips only itself if BlackHole's output
    volume scalar cannot be read on this machine -- never the content test
    above, which always runs and always asserts regardless of whether this
    one can.
    """
    # TONE_AMPLITUDE is the fixed module-level constant this canonical RMS
    # is derived from, independent of whatever array actually gets played
    # -- see `_assert_recorded_level_is_plausible` for why that decoupling
    # matters.
    canonical_tone_rms = TONE_AMPLITUDE / np.sqrt(2)  # RMS of a full-length sine at this amplitude
    tone_to_play = _synthesize_tone()

    result = _play_reference_and_capture(record_input_device, tone_to_play, SAMPLE_RATE)

    mono = result.samples.mean(axis=1).astype(np.float64)
    _assert_recorded_level_is_plausible(mono, canonical_tone_rms)


def test_speech_played_into_loopback_is_recovered(record_input_device):
    """A real, natural-amplitude speech utterance fed to BlackHole must be recoverable: matching content.

    Content-only (Critic round 3 finding 2): see
    `test_tone_played_into_loopback_is_recovered`'s docstring for why the
    level claim is a separate test (`test_speech_level_is_plausible`).
    """
    if not SPEECH_FIXTURE.exists():
        pytest.skip(f"missing fixture: {SPEECH_FIXTURE}")
    speech_to_play, speech_sample_rate = sf.read(str(SPEECH_FIXTURE), dtype="float32")

    result = _play_reference_and_capture(record_input_device, speech_to_play, speech_sample_rate)

    mono = result.samples.mean(axis=1).astype(np.float64)

    reference_at_capture_rate = resample(
        speech_to_play.astype(np.float64), int(len(speech_to_play) * result.sample_rate / speech_sample_rate)
    )
    score = abs(_matched_filter_score(reference_at_capture_rate, mono))
    assert score > MATCH_SCORE_MIN, (
        f"expected the recording to match the emitted speech (score > {MATCH_SCORE_MIN}), got {score:.4f}"
    )


def test_speech_level_is_plausible(record_input_device):
    """Level-only companion to `test_speech_played_into_loopback_is_recovered`
    (Critic round 3 finding 2). Skips only itself if BlackHole's output
    volume scalar cannot be read on this machine.
    """
    if not SPEECH_FIXTURE.exists():
        pytest.skip(f"missing fixture: {SPEECH_FIXTURE}")
    # Read twice, deliberately: `speech_to_play` is what is actually
    # emitted; `canonical_speech` is an independent re-read used only to
    # anchor the level check's expectation, so that expectation cannot be
    # dragged down by any mutation of the playback path -- see
    # `_assert_recorded_level_is_plausible`.
    speech_to_play, speech_sample_rate = sf.read(str(SPEECH_FIXTURE), dtype="float32")
    canonical_speech, _ = sf.read(str(SPEECH_FIXTURE), dtype="float32")
    canonical_speech_rms = _rms(canonical_speech)

    result = _play_reference_and_capture(record_input_device, speech_to_play, speech_sample_rate)

    mono = result.samples.mean(axis=1).astype(np.float64)
    _assert_recorded_level_is_plausible(mono, canonical_speech_rms)


def _assert_capture_contains_emitted_but_not_reference(
    mono: np.ndarray,
    capture_sample_rate: int,
    emitted: np.ndarray,
    emitted_sample_rate: int,
    true_reference: np.ndarray,
    true_reference_sample_rate: int,
) -> None:
    """Shared by every negative-control test below (Critic round 3 finding
    1). Two assertions, in order:

    1. The capture must score *above* `MATCH_SCORE_MIN` against `emitted`
       — the signal the test actually played — proving something real
       reached the microphone. Without this, "the capture doesn't match
       the true reference" is trivially satisfied by silence, a dead or
       exception-raising playback thread, or a corrupted/empty fixture
       (all reproduced directly against the pre-fix version of this suite:
       a no-op `sd.play`, `sd.play` raising, and `hard_negative_en.wav`
       replaced with zeros all left the negative-control tests green).
    2. Only once (1) holds does the capture *not* matching `true_reference`
       mean anything, so it is asserted second.
    """
    emitted_at_capture_rate = resample(
        emitted.astype(np.float64), int(len(emitted) * capture_sample_rate / emitted_sample_rate)
    )
    self_score = abs(_matched_filter_score(emitted_at_capture_rate, mono))
    assert self_score > MATCH_SCORE_MIN, (
        f"expected the capture to contain the signal actually emitted (self-score > {MATCH_SCORE_MIN}), "
        f"got {self_score:.4f} -- the negative-control assertion below cannot mean anything unless this holds"
    )

    reference_at_capture_rate = resample(
        true_reference.astype(np.float64),
        int(len(true_reference) * capture_sample_rate / true_reference_sample_rate),
    )
    reference_score = abs(_matched_filter_score(reference_at_capture_rate, mono))
    assert reference_score < MATCH_SCORE_MIN, (
        f"expected the emitted signal to score below {MATCH_SCORE_MIN} against the true reference, "
        f"got {reference_score:.4f}"
    )


def test_speech_content_does_not_match_a_different_utterance_same_voice(record_input_device):
    """Hard negative control (Critic round 2 blocker): playing a *different*
    English sentence, same voice and similar length as `SPEECH_FIXTURE`,
    must score *below* the match threshold against `SPEECH_FIXTURE`'s
    content. Checked into the suite as its own test, not just a one-off
    mutation-testing exercise: this is the discrimination property SC-001
    actually depends on, and the round-1 negative control (different
    speaker *and* different language) never exercised it.
    """
    if not SPEECH_FIXTURE.exists() or not HARD_NEGATIVE_SPEECH_FIXTURE.exists():
        pytest.skip(f"missing fixture(s): {SPEECH_FIXTURE}, {HARD_NEGATIVE_SPEECH_FIXTURE}")
    true_reference, ref_sample_rate = sf.read(str(SPEECH_FIXTURE), dtype="float32")
    hard_negative, hn_sample_rate = sf.read(str(HARD_NEGATIVE_SPEECH_FIXTURE), dtype="float32")

    result = _play_reference_and_capture(record_input_device, hard_negative, hn_sample_rate)

    mono = result.samples.mean(axis=1).astype(np.float64)
    _assert_capture_contains_emitted_but_not_reference(
        mono, result.sample_rate, hard_negative, hn_sample_rate, true_reference, ref_sample_rate
    )


def test_speech_content_does_not_match_the_reference_time_reversed(record_input_device):
    """Second Critic round 2 emitted-signal negative control: the same
    reference audio, played backwards, must also score below the match
    threshold against the forward reference. Same spectral content (an FFT
    magnitude spectrum is insensitive to time-reversal), different temporal
    structure -- a check that only used spectral content, not a temporal
    matched filter, would not catch this.
    """
    if not SPEECH_FIXTURE.exists():
        pytest.skip(f"missing fixture: {SPEECH_FIXTURE}")
    true_reference, ref_sample_rate = sf.read(str(SPEECH_FIXTURE), dtype="float32")
    reversed_reference = true_reference[::-1].copy()

    result = _play_reference_and_capture(record_input_device, reversed_reference, ref_sample_rate)

    mono = result.samples.mean(axis=1).astype(np.float64)
    _assert_capture_contains_emitted_but_not_reference(
        mono, result.sample_rate, reversed_reference, ref_sample_rate, true_reference, ref_sample_rate
    )


def test_white_noise_does_not_match_speech_content(record_input_device):
    """Third Critic round 2 emitted-signal negative control: white noise
    played through the device must score below the match threshold against
    real speech content — non-silence alone (which white noise also is)
    must not be mistaken for the specific reference content.
    """
    if not SPEECH_FIXTURE.exists():
        pytest.skip(f"missing fixture: {SPEECH_FIXTURE}")
    true_reference, ref_sample_rate = sf.read(str(SPEECH_FIXTURE), dtype="float32")
    rng = np.random.default_rng(123)
    noise = np.clip(rng.normal(0.0, 0.3, len(true_reference)), -1.0, 1.0).astype(np.float32)

    result = _play_reference_and_capture(record_input_device, noise, ref_sample_rate)

    mono = result.samples.mean(axis=1).astype(np.float64)
    _assert_capture_contains_emitted_but_not_reference(
        mono, result.sample_rate, noise, ref_sample_rate, true_reference, ref_sample_rate
    )


# ---------------------------------------------------------------------------
# Hermetic scorer tests (Critic round 2 finding 3): no hardware needed.
# Nothing previously asked whether the scorer functions themselves score
# correctly on inputs with a known right answer.
# ---------------------------------------------------------------------------


def test_matched_filter_score_is_near_one_for_an_identical_signal():
    rng = np.random.default_rng(0)
    signal = rng.normal(size=4000)
    assert _matched_filter_score(signal, signal.copy()) == pytest.approx(1.0, abs=1e-9)


def test_matched_filter_score_is_unchanged_by_positive_scaling():
    rng = np.random.default_rng(1)
    reference = rng.normal(size=4000)
    recorded = reference[500:1500] * 0.0001
    score_at_full_scale = _matched_filter_score(reference, reference[500:1500].copy())
    score_at_tiny_scale = _matched_filter_score(reference, recorded)
    assert score_at_tiny_scale == pytest.approx(score_at_full_scale, abs=1e-9)


def test_matched_filter_score_is_low_for_unrelated_same_class_noise():
    rng = np.random.default_rng(2)
    reference = rng.normal(size=4000)
    unrelated = np.random.default_rng(3).normal(size=1200)
    score = abs(_matched_filter_score(reference, unrelated))
    assert score < MATCH_SCORE_MIN


def test_matched_filter_score_is_zero_for_pure_silence():
    reference = np.random.default_rng(4).normal(size=4000)
    silence = np.zeros(1200)
    assert _matched_filter_score(reference, silence) == 0.0


def test_peak_frequency_hz_recovers_a_synthetic_tone():
    sample_rate = 48_000
    t = np.arange(int(sample_rate * 1.0)) / sample_rate
    synthetic_tone = np.sin(2 * np.pi * 1234.0 * t)
    peak_hz = _peak_frequency_hz(synthetic_tone, sample_rate)
    assert abs(peak_hz - 1234.0) <= 1.0


def test_peak_frequency_hz_distinguishes_two_different_tones():
    sample_rate = 48_000
    t = np.arange(int(sample_rate * 1.0)) / sample_rate
    low_tone = np.sin(2 * np.pi * 500.0 * t)
    high_tone = np.sin(2 * np.pi * 3000.0 * t)
    assert _peak_frequency_hz(low_tone, sample_rate) != pytest.approx(
        _peak_frequency_hz(high_tone, sample_rate), abs=10.0
    )


# ---------------------------------------------------------------------------
# Hermetic level-check tests (Critic round 3 finding 3): no hardware needed.
# `_assert_recorded_level_is_plausible` itself had zero tests -- replacing
# it with a no-op left the suite at 23 passed. These exercise its logic
# directly against a monkeypatched, known volume scalar, independent of
# whatever CoreAudio actually reports on the machine running the suite.
# ---------------------------------------------------------------------------


def test_assert_recorded_level_is_plausible_rejects_a_too_quiet_recording(monkeypatch):
    monkeypatch.setattr(conftest, "read_core_audio_volume_scalar", lambda device_name, scope: 1.0)
    canonical_rms = 0.2  # predicted_rms = 0.2 * gain(1.0) = 0.2
    too_quiet = np.full(1000, 0.2 * LEVEL_RATIO_MIN * 0.1, dtype=np.float64)  # far below the floor

    with pytest.raises(AssertionError):
        _assert_recorded_level_is_plausible(too_quiet, canonical_rms)


def test_assert_recorded_level_is_plausible_accepts_a_plausible_recording(monkeypatch):
    monkeypatch.setattr(conftest, "read_core_audio_volume_scalar", lambda device_name, scope: 1.0)
    canonical_rms = 0.2
    plausible = np.full(1000, canonical_rms, dtype=np.float64)  # matches the predicted level exactly

    _assert_recorded_level_is_plausible(plausible, canonical_rms)  # must not raise


def test_assert_recorded_level_is_plausible_scales_its_floor_with_the_measured_gain(monkeypatch):
    """Same (quiet) recording, two different (monkeypatched) volume
    scalars: plausible when the measured gain is low (predicts a low
    floor), implausible once the measured gain rises to unity (predicts a
    floor the same recording no longer clears) -- proving the floor tracks
    the reader's output rather than being a fixed constant in disguise."""
    canonical_rms = 0.2
    recorded = np.full(1000, 0.0002, dtype=np.float64)  # quiet: only plausible at low gain

    monkeypatch.setattr(conftest, "read_core_audio_volume_scalar", lambda device_name, scope: 0.0)
    _assert_recorded_level_is_plausible(recorded, canonical_rms)  # fine: predicted floor is tiny too

    monkeypatch.setattr(conftest, "read_core_audio_volume_scalar", lambda device_name, scope: 1.0)
    with pytest.raises(AssertionError):
        _assert_recorded_level_is_plausible(recorded, canonical_rms)  # too quiet for a unity-gain floor


def test_assert_recorded_level_is_plausible_skips_only_itself_when_the_scalar_is_unreadable(monkeypatch):
    monkeypatch.setattr(conftest, "read_core_audio_volume_scalar", lambda device_name, scope: None)

    with pytest.raises(pytest.skip.Exception):
        _assert_recorded_level_is_plausible(np.zeros(10), 0.2)
