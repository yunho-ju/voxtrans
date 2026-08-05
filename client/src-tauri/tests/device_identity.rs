//! T017b (US1, FR-010: "Audio devices MUST be identified by stable UID
//! rather than by name or index"; specs/002-audio-routing/briefs/
//! US1-round2-T017-T017b.md). "The machine carries two virtual drivers, so
//! this is a live risk, not a hypothetical" (tasks.md T017b) -- a real UID
//! and a real device's *name* used as if it were a UID must resolve
//! differently, and the exported API surface for starting playback never
//! accepts an index at all.
//!
//! Only `audio_output::list_output_devices` and `audio_output::
//! start_tts_playback` are exercised here: `mic_capture` is a private
//! module of `voxtrans_client_lib` (`lib.rs` :4) and not reachable from an
//! external test crate -- see `input_device_enumeration.rs`'s module
//! docstring for the same constraint on the input side. Device resolution
//! itself lives entirely in Swift's `findAudioDeviceByUID`
//! (`AudioCapture.swift` :545-588), a private `func` with no `@_cdecl`
//! export -- it is reached only indirectly here, through
//! `start_tts_playback` -> `audio_output_start`.
//!
//! FR-010's "never by index" clause is not exercised by a runtime test
//! here: `list_output_devices` returns `Vec<(String, String)>` (name, uid)
//! pairs and `start_tts_playback` takes the UID `String` directly -- there
//! is no index anywhere in this layer's exported signatures for a
//! device-list reordering to destabilize. That is a structural fact
//! checked by the compiler on every build, not a behaviour a runtime
//! assertion could usefully add power to (an assertion that only fails if
//! the function signature itself changed would be indistinguishable from
//! "always true").

use std::sync::mpsc as std_mpsc;
use std::sync::Mutex;
use std::thread;
use std::time::Duration;
use tokio::sync::mpsc;

use voxtrans_client_lib::audio_output;

const START_TIMEOUT: Duration = Duration::from_secs(5);

// `cargo test` runs every `#[test]` fn in this binary on its own thread
// concurrently by default. Under the *correct* implementation, only
// `a_real_devices_own_uid_resolves_and_starts` ever successfully opens a
// real `AVAudioEngine` (the other tests' UIDs are expected not to resolve
// at all), so there is normally nothing to collide with. A deliberate
// mutation of `findAudioDeviceByUID` to also match by display name (run
// while writing this file, then reverted -- see US1.md "Round 2
// mutations") made a second test resolve the *same* physical device
// concurrently, and two `AVAudioEngine`s racing for one real device
// crashed the whole test process (`SIGABRT`, an uncatchable
// Objective-C/C++ exception) instead of failing cleanly. Serializing every
// test in this file against this lock is cheap insurance against that
// class of crash masking a real assertion failure, mirroring the Python
// suite's `audio_device_lock` for the same reason (shared, stateful
// hardware).
static REAL_DEVICE_ACCESS: Mutex<()> = Mutex::new(());

/// Runs `start_tts_playback(uid, name, ...)` on a thread and returns once it
/// has either failed to start (an immediate `Err`, no thread left running)
/// or been confirmed started and then told to stop. Bounded by
/// `START_TIMEOUT` either way -- a case that hangs here would be the
/// T017a defect, not this file's concern, so a timeout here is reported
/// distinctly from a resolution failure.
fn try_start_and_stop(uid: &str, name: &str) -> Result<(), String> {
    let (_tts_tx, tts_rx) = mpsc::channel::<Vec<u8>>(1);
    let (stop_tx, stop_rx) = mpsc::channel::<()>(1);
    let (_tts_end_tx, tts_end_rx) = mpsc::channel::<i64>(1);
    let (playback_complete_tx, _playback_complete_rx) = mpsc::channel::<String>(1);
    let (done_tx, done_rx) = std_mpsc::channel::<Result<(), String>>();

    let uid_owned = uid.to_string();
    let name_owned = name.to_string();
    thread::spawn(move || {
        let result = audio_output::start_tts_playback(
            uid_owned,
            name_owned,
            tts_rx,
            stop_rx,
            tts_end_rx,
            playback_complete_tx,
        );
        let _ = done_tx.send(result);
    });

    // If `uid` fails to resolve, `start_tts_playback` returns `Err`
    // immediately (before ever entering its receive loop) -- give it a
    // moment to do so before assuming it started successfully and needs a
    // stop signal.
    match done_rx.recv_timeout(Duration::from_millis(500)) {
        Ok(result) => return result,
        Err(std_mpsc::RecvTimeoutError::Timeout) => {} // presumably started; fall through to stop it
        Err(std_mpsc::RecvTimeoutError::Disconnected) => {
            panic!("start_tts_playback's thread ended without sending a result")
        }
    }

    // Critic round 2 minor finding 3: if the thread happened to finish and
    // drop `stop_rx` in the narrow window between the 500ms timeout above
    // and this send (it does not need a full extra `START_TIMEOUT` to do
    // that once it has decided to exit), `blocking_send` returning `Err`
    // means only "the thread is already gone", not an infrastructure
    // failure -- it already sent its real result over `done_tx`, which the
    // `recv_timeout` below will pick up regardless of whether this send
    // succeeded. Panicking here on that ordinary race was flake, not signal.
    let _ = stop_tx.blocking_send(());
    done_rx
        .recv_timeout(START_TIMEOUT)
        .unwrap_or_else(|_| panic!("start_tts_playback did not stop within {START_TIMEOUT:?}"))
}

/// A real device's own UID resolves and starts successfully -- the positive
/// half of "identified by UID" (FR-010): without this, the negative cases
/// below would be meaningless (a resolver that rejects everything would
/// trivially "pass" them too).
#[test]
fn a_real_devices_own_uid_resolves_and_starts() {
    let _guard = REAL_DEVICE_ACCESS.lock().unwrap_or_else(|e| e.into_inner());
    let outputs = audio_output::list_output_devices();
    if outputs.is_empty() {
        eprintln!("no output devices enumerated on this machine -- skipping");
        return;
    }
    let (name, uid) = &outputs[0];

    let result = try_start_and_stop(uid, "");
    assert!(
        result.is_ok(),
        "expected {uid:?} (device {name:?}, taken directly from list_output_devices()) to \
         resolve and start successfully, got {result:?}"
    );
}

/// A device's **name** used in place of its UID must not resolve -- proving
/// lookup is not silently falling back to (or accidentally matching on)
/// display name. Skips only the specific device(s) where name happens to
/// equal UID (not expected on this machine, but not this test's claim to
/// rule out globally); if every enumerated device has name == uid, there is
/// nothing left to distinguish and the test is skipped rather than
/// vacuously passed.
#[test]
fn a_devices_display_name_does_not_resolve_it_as_a_uid() {
    let _guard = REAL_DEVICE_ACCESS.lock().unwrap_or_else(|e| e.into_inner());
    let outputs = audio_output::list_output_devices();
    if outputs.is_empty() {
        eprintln!("no output devices enumerated on this machine -- skipping");
        return;
    }

    let mut exercised_at_least_one = false;
    for (name, uid) in &outputs {
        if name == uid {
            continue; // no name/UID distinction to test for this device
        }
        exercised_at_least_one = true;
        let result = try_start_and_stop(name, "");
        assert!(
            result.is_err(),
            "expected device name {name:?} (real UID is {uid:?}) to fail to resolve as a UID \
             -- got Ok, meaning device selection accepted a display name"
        );
    }

    if !exercised_at_least_one {
        eprintln!(
            "every enumerated output device's name equals its own UID on this machine -- \
             nothing to distinguish, skipping"
        );
    }
}

/// A UID string that resembles a real one closely (its exact text with one
/// character appended) but matches no device must still fail -- proving
/// resolution is an exact match, not a prefix/substring match that could
/// make an unrelated, partially-similar UID resolve to the wrong device.
///
/// This is also T017's FR-009 mutation-sensitivity case (Critic round 2
/// major finding 1): before this case existed, nothing anywhere in the
/// suite called `start_tts_playback` with a *non-empty* `device_name` and
/// then inspected the resulting `Err` string, so reverting
/// `output_start_error_message` to the old constant `"Failed to start
/// audio output"`, or breaking `commands.rs`'s uid/name threading, changed
/// nothing observable. This drives the real FFI failure path (a UID
/// `audio_output_start` genuinely rejects) and asserts the returned
/// message actually names both the UID and the last-known device name it
/// was given.
#[test]
fn a_uid_that_is_not_an_exact_match_for_any_device_fails_and_names_the_uid_and_last_known_name() {
    let _guard = REAL_DEVICE_ACCESS.lock().unwrap_or_else(|e| e.into_inner());
    let outputs = audio_output::list_output_devices();
    if outputs.is_empty() {
        eprintln!("no output devices enumerated on this machine -- skipping");
        return;
    }
    let (_name, real_uid) = &outputs[0];
    let near_miss_uid = format!("{real_uid}-does-not-exist");
    let fabricated_name = "Fabricated Device Name For T017";

    let result = try_start_and_stop(&near_miss_uid, fabricated_name);
    let err = match result {
        Err(e) => e,
        Ok(()) => panic!(
            "expected a UID that is not an exact match for any device ({near_miss_uid:?}, \
             derived from the real UID {real_uid:?}) to fail to resolve, got Ok"
        ),
    };
    assert!(
        err.contains(&near_miss_uid),
        "expected the error to name the UID {near_miss_uid:?} (FR-009), got {err:?}"
    );
    assert!(
        err.contains(fabricated_name),
        "expected the error to name the last-known device name {fabricated_name:?} (FR-009), \
         got {err:?}"
    );
}

/// The same failure with no known device name (the branch
/// `output_start_error_message` takes when the caller never learned one)
/// -- still names the UID.
#[test]
fn a_uid_that_is_not_an_exact_match_for_any_device_fails_and_names_the_uid_alone_when_no_name_is_known(
) {
    let _guard = REAL_DEVICE_ACCESS.lock().unwrap_or_else(|e| e.into_inner());
    let outputs = audio_output::list_output_devices();
    if outputs.is_empty() {
        eprintln!("no output devices enumerated on this machine -- skipping");
        return;
    }
    let (_name, real_uid) = &outputs[0];
    let near_miss_uid = format!("{real_uid}-does-not-exist-no-name");

    let result = try_start_and_stop(&near_miss_uid, "");
    let err = match result {
        Err(e) => e,
        Ok(()) => panic!(
            "expected a UID that is not an exact match for any device ({near_miss_uid:?}) to \
             fail to resolve, got Ok"
        ),
    };
    assert!(
        err.contains(&near_miss_uid),
        "expected the error to name the UID {near_miss_uid:?} even with no known device name \
         (FR-009), got {err:?}"
    );
}
