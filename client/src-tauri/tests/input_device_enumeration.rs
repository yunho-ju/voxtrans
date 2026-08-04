//! Rust-layer companion to US3 (T006-T012): proves the mic input device
//! enumeration crosses the Swift FFI and returns well-formed `(name, uid)`
//! pairs. This is not the SC-005 signal-content proof -- that requires
//! actually driving `mic_start_capture` against real audio, which the
//! `mic_capture` module cannot do from here (it is a private module of
//! `voxtrans_client_lib`, not exported for external test crates), so that
//! proof lives at `server/tests/integration/test_input_selection.py`,
//! against the same Swift source compiled as a standalone dylib. See that
//! file's module docstring for the full design rationale.
//!
//! What *is* verifiable from here, cheaply, is exactly what D9 assigns to
//! this layer: "device-list parsing, UID plumbing" -- the enumeration
//! contract itself (`contracts/audio-devices.md`), mirroring
//! `scaffolding.rs`'s proof for the output-device side.

#[test]
fn input_device_enumeration_crosses_the_swift_ffi_and_returns_well_formed_nonempty_pairs() {
    let devices = voxtrans_client_lib::audio_capture::list_input_devices();

    // Every Mac has a built-in microphone, so unlike a count comparison
    // tied to *which* devices are plugged in (which would depend on
    // ambient machine state -- constraint 1), "at least one" is a safe,
    // non-flaky floor. Critic round 1 finding 5: a loop over an empty
    // `Vec` never runs, so a prior version of this test that only asserted
    // per-element properties passed vacuously on an empty enumeration --
    // indistinguishable from "the FFI call silently returns nothing".
    assert!(
        !devices.is_empty(),
        "expected at least one microphone-capable input device (every Mac has a built-in \
         microphone), got an empty enumeration"
    );

    for (name, uid) in &devices {
        assert!(!name.is_empty(), "device name must not be empty: {devices:?}");
        assert!(!uid.is_empty(), "device uid must not be empty: {devices:?}");
    }
}

#[test]
fn input_device_enumeration_is_distinguishable_from_the_output_device_enumeration() {
    // Critic round 1 finding 5: a well-formed, non-empty check alone does
    // not prove `list_input_devices` filters by *input* streams at all --
    // it would equally satisfy a filter bug that enumerates every device
    // (or that returns exactly `list_output_devices`' own set). This
    // machine has at least one output-only device (a built-in speaker) and
    // at least one input-only device (a built-in microphone), so the two
    // sets are provably not identical when the underlying Swift stream
    // filters (`kAudioDevicePropertyStreams` scoped to input vs. output)
    // are both intact -- guaranteed by built-in hardware, not by what
    // happens to be plugged in.
    let inputs = voxtrans_client_lib::audio_capture::list_input_devices();
    let outputs = voxtrans_client_lib::audio_output::list_output_devices();

    let input_uids: std::collections::HashSet<&str> =
        inputs.iter().map(|(_, uid)| uid.as_str()).collect();
    let output_uids: std::collections::HashSet<&str> =
        outputs.iter().map(|(_, uid)| uid.as_str()).collect();

    assert_ne!(
        input_uids, output_uids,
        "expected the input and output device sets to differ -- an output-only device \
         (e.g. a speaker) must not enumerate as a microphone and vice versa; got identical \
         sets, inputs={inputs:?} outputs={outputs:?}"
    );
}
