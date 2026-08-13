//! Changing the output device mid-session retargets the *running* playback
//! thread.
//!
//! What these guard: if the device stays bound to whatever
//! `start_tts_playback` opened, choosing a different output in settings
//! updates the UI and the stored preference while playback continues to the
//! original device. Nothing reports the mismatch, so the symptom is "the
//! setting says MacBook 스피커 and I hear nothing", and the only way to see
//! the truth is the app's stderr.
//!
//! Like `device_identity.rs`, these drive real CoreAudio: a machine with no
//! usable output device cannot run them, so they skip rather than fail.

use std::sync::mpsc as std_mpsc;
use std::sync::Mutex;
use std::thread;
use std::time::Duration;
use tokio::sync::mpsc;

use voxtrans_client_lib::audio_output;

/// Both tests open a real `AVAudioEngine`, and two racing for one physical
/// device abort the whole process (`SIGABRT`, uncatchable) rather than failing
/// an assertion. Same guard, and same reason, as `device_identity.rs`'s
/// `REAL_DEVICE_ACCESS`.
static REAL_DEVICE_ACCESS: Mutex<()> = Mutex::new(());

/// Long enough for the playback loop (20ms poll) to observe a switch many
/// times over, short enough that a hang is still a fast failure.
const SETTLE: Duration = Duration::from_millis(300);

fn first_real_device() -> Option<(String, String)> {
    audio_output::list_output_devices()
        .into_iter()
        .find(|(_, uid)| !uid.is_empty())
        .map(|(name, uid)| (uid, name))
}

#[test]
fn a_switch_reaches_the_running_playback_thread_without_ending_it() {
    let _guard = REAL_DEVICE_ACCESS.lock().unwrap_or_else(|e| e.into_inner());
    let Some((uid, name)) = first_real_device() else {
        eprintln!("skipping: no real output device to drive playback against");
        return;
    };

    let (_tts_tx, tts_rx) = mpsc::channel::<Vec<u8>>(1);
    let (stop_tx, stop_rx) = mpsc::channel::<()>(1);
    let (_tts_end_tx, tts_end_rx) = mpsc::channel::<i64>(1);
    let (playback_complete_tx, _playback_complete_rx) = mpsc::channel::<String>(1);
    let (device_change_tx, device_change_rx) = mpsc::channel::<(String, String)>(4);
    let (done_tx, done_rx) = std_mpsc::channel::<Result<(), String>>();

    thread::spawn(move || {
        let result = audio_output::start_tts_playback(
            String::new(), // start on the system default
            String::new(),
            tts_rx,
            stop_rx,
            tts_end_rx,
            playback_complete_tx,
            device_change_rx,
        );
        let _ = done_tx.send(result);
    });

    // It must still be running: an immediate result here means it never
    // entered its loop, so the switch below would prove nothing.
    assert!(
        done_rx.recv_timeout(SETTLE).is_err(),
        "playback ended before the switch was even sent"
    );

    device_change_tx
        .blocking_send((uid.clone(), name.clone()))
        .expect("playback thread should still be accepting device switches");
    thread::sleep(SETTLE);

    // The switch must not have killed the session -- the old behaviour people
    // fell back on was stop/start, and the whole point is not needing it.
    assert!(
        done_rx.recv_timeout(SETTLE).is_err(),
        "switching the output device ended the playback session"
    );

    stop_tx.blocking_send(()).expect("stop should be accepted");
    let result = done_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("playback thread should exit after stop");
    assert!(result.is_ok(), "clean shutdown expected, got {result:?}");
}

#[test]
fn a_failed_switch_does_not_end_the_session() {
    let _guard = REAL_DEVICE_ACCESS.lock().unwrap_or_else(|e| e.into_inner());
    // A UID that resolves to nothing: the switch fails, and the thread must
    // keep running so the next (valid) selection can recover it. Ending the
    // session on a typo'd device would strand the user mid-meeting.
    let (_tts_tx, tts_rx) = mpsc::channel::<Vec<u8>>(1);
    let (stop_tx, stop_rx) = mpsc::channel::<()>(1);
    let (_tts_end_tx, tts_end_rx) = mpsc::channel::<i64>(1);
    let (playback_complete_tx, _playback_complete_rx) = mpsc::channel::<String>(1);
    let (device_change_tx, device_change_rx) = mpsc::channel::<(String, String)>(4);
    let (done_tx, done_rx) = std_mpsc::channel::<Result<(), String>>();

    thread::spawn(move || {
        let result = audio_output::start_tts_playback(
            String::new(),
            String::new(),
            tts_rx,
            stop_rx,
            tts_end_rx,
            playback_complete_tx,
            device_change_rx,
        );
        let _ = done_tx.send(result);
    });

    if done_rx.recv_timeout(SETTLE).is_ok() {
        eprintln!("skipping: no usable default output device on this machine");
        return;
    }

    device_change_tx
        .blocking_send((
            "NoSuchDevice_UID".to_string(),
            "Nonexistent Device".to_string(),
        ))
        .expect("playback thread should still be accepting device switches");
    thread::sleep(SETTLE);

    assert!(
        done_rx.recv_timeout(SETTLE).is_err(),
        "a failed device switch must not end the playback session"
    );

    stop_tx.blocking_send(()).expect("stop should be accepted");
    assert!(
        done_rx.recv_timeout(Duration::from_secs(5)).is_ok_and(|r| r.is_ok()),
        "playback thread should still shut down cleanly after a failed switch"
    );
}
