//! T017a's Rust half (specs/002-audio-routing/briefs/US1-round2-T017-T017b.md
//! "조사로 확정된 사실" -> "재생 스레드 (T017a)"):
//! `audio_output::start_tts_playback`'s receive loop calls
//! `tts_rx.blocking_recv()` with **no timeout** (`audio_output.rs` :63 at the
//! time this test was written). The stop signal is only re-checked via a
//! non-blocking `try_recv()` immediately *before* that blocking call
//! (:57-61), so if no TTS audio chunk is ever sent, the stop signal sent
//! after that point is never re-checked and the thread never exits -- the
//! comment at :62 claims "with timeout to re-check stop", which this test
//! exists to prove false.
//!
//! This is the Rust counterpart to `test_output_device_lost.py`'s
//! stop-responsiveness case, which drives the same concern from the Swift
//! side (`audio_output_stop()` returning promptly after the device is
//! destroyed); this file drives it from the Rust side of the channel
//! protocol, independently of any device state at all -- the defect here
//! reproduces even with a healthy, present device, as long as no chunk
//! happens to have been sent yet when `stop` fires.

use std::sync::mpsc as std_mpsc;
use std::thread;
use std::time::Duration;
use tokio::sync::mpsc;

// Generous relative to how quickly a real machine's default audio output
// should start (well under 1s observed) -- see
// `test_output_device_lost.py`'s `STOP_RESPONSIVENESS_TIMEOUT_S` for the
// same 5s choice on the Swift side of this same concern.
const STOP_RESPONSIVENESS_TIMEOUT: Duration = Duration::from_secs(5);

/// `start_tts_playback` must return promptly once `stop` is signalled, even
/// if not a single TTS audio chunk was ever sent on `tts_tx` -- the exact
/// condition the brief's "재생 스레드" analysis identifies as the
/// `blocking_recv`-without-timeout defect.
#[test]
fn stop_signal_is_honored_even_when_no_tts_audio_is_ever_sent() {
    // Empty UID = system default output (every Mac has one -- built-in
    // speakers/headphones -- so this does not need a hardware-presence
    // skip the way a specific named device would; see
    // `mic_capture::start_mic_capture`'s identical empty-UID convention).
    let outputs = voxtrans_client_lib::audio_output::list_output_devices();
    if outputs.is_empty() {
        eprintln!(
            "no output devices enumerated on this machine -- skipping (no real audio hardware \
             to drive `start_tts_playback` against)"
        );
        return;
    }

    let (_tts_tx, tts_rx) = mpsc::channel::<Vec<u8>>(1);
    let (stop_tx, stop_rx) = mpsc::channel::<()>(1);
    let (_tts_end_tx, tts_end_rx) = mpsc::channel::<i64>(1);
    let (playback_complete_tx, _playback_complete_rx) = mpsc::channel::<String>(1);
    // No device switches in this test; the receiver only has to stay alive.
    let (_device_change_tx, device_change_rx) = mpsc::channel::<(String, String)>(1);

    let (done_tx, done_rx) = std_mpsc::channel::<Result<(), String>>();
    thread::spawn(move || {
        let result = voxtrans_client_lib::audio_output::start_tts_playback(
            String::new(),
            String::new(),
            tts_rx,
            stop_rx,
            tts_end_rx,
            playback_complete_tx,
            device_change_rx,
        );
        // The receiver may already be gone if the test below timed out and
        // returned -- send() failing here is not this test's concern.
        let _ = done_tx.send(result);
    });

    // Give the spawned thread time to actually call `audio_output_start`
    // and enter its receive loop before the stop signal is sent -- sending
    // stop before the loop is even reached would trivially pass regardless
    // of the timeout defect (the `try_recv()` before the first
    // `blocking_recv()` would already catch it).
    thread::sleep(Duration::from_millis(300));
    stop_tx
        .blocking_send(())
        .expect("failed to send stop signal to start_tts_playback's stop_rx");

    match done_rx.recv_timeout(STOP_RESPONSIVENESS_TIMEOUT) {
        Ok(result) => {
            assert!(
                result.is_ok(),
                "expected start_tts_playback to shut down cleanly after the stop signal, got {result:?}"
            );
        }
        Err(_) => panic!(
            "start_tts_playback did not return within {STOP_RESPONSIVENESS_TIMEOUT:?} of the \
             stop signal being sent with no TTS audio ever enqueued -- tts_rx.blocking_recv() \
             has no timeout to re-check stop_rx (audio_output.rs)"
        ),
    }
}
