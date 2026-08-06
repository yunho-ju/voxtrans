use log::info;
use parking_lot::Mutex;
use std::ffi::CStr;
use std::os::raw::c_char;
use tokio::sync::mpsc;

const CHUNK_SAMPLES: usize = 1600; // 100ms @ 16kHz
pub const AUDIO_PREFIX_SYSTEM: u8 = 0x01;

static AUDIO_TX: Mutex<Option<mpsc::Sender<Vec<u8>>>> = Mutex::new(None);

// T035 (US4, research.md D6): a device-change rebuild of the capture graph
// happens entirely inside Swift and needs no Rust involvement when it
// succeeds -- audio keeps flowing through the same registered
// `audio_data_callback` below. If a rebuild fails, Swift reports it through
// this channel (fed by `audio_error_callback`) rather than the capture
// thread silently continuing to report a healthy session (FR-008).
static AUDIO_ERROR_TX: Mutex<Option<mpsc::Sender<String>>> = Mutex::new(None);

// FFI to Swift ScreenCaptureKit module
extern "C" {
    fn sck_start_capture(
        callback: extern "C" fn(*const f32, usize, usize, usize) -> (),
        error_callback: extern "C" fn(*const c_char) -> (),
    ) -> bool;
    fn sck_stop_capture();
    fn audio_list_input_devices() -> *mut c_char;
}

/// Called from Swift with PCM float32 mono 16kHz audio data
extern "C" fn audio_data_callback(
    data: *const f32,
    num_samples: usize,
    _sample_rate: usize,
    _channels: usize,
) {
    if data.is_null() || num_samples == 0 {
        return;
    }

    let samples = unsafe { std::slice::from_raw_parts(data, num_samples) };

    let tx_guard = AUDIO_TX.lock();
    if let Some(ref tx) = *tx_guard {
        // Accumulate into CHUNK_SAMPLES-sized chunks
        thread_local! {
            static BUFFER: std::cell::RefCell<Vec<f32>> = std::cell::RefCell::new(Vec::with_capacity(CHUNK_SAMPLES * 2));
        }

        BUFFER.with(|buf| {
            let mut buf = buf.borrow_mut();
            buf.extend_from_slice(samples);

            while buf.len() >= CHUNK_SAMPLES {
                let chunk: Vec<f32> = buf.drain(..CHUNK_SAMPLES).collect();
                let audio_bytes: &[u8] = bytemuck::cast_slice(&chunk);
                // Prepend 0x01 prefix for system audio
                let mut bytes = Vec::with_capacity(1 + audio_bytes.len());
                bytes.push(AUDIO_PREFIX_SYSTEM);
                bytes.extend_from_slice(audio_bytes);
                if tx.try_send(bytes).is_err() {
                    // Channel full, drop chunk
                }
            }
        });
    }
}

/// Called from Swift when a device-change rebuild of the capture graph
/// fails (T035, research.md D6) -- the failure path FR-008 requires instead
/// of going silently quiet. Forwards the message to `AUDIO_ERROR_TX`, mirroring
/// `audio_data_callback`'s use of `AUDIO_TX` above.
extern "C" fn audio_error_callback(message: *const c_char) {
    if message.is_null() {
        return;
    }
    let text = unsafe { CStr::from_ptr(message) }
        .to_string_lossy()
        .to_string();

    let tx_guard = AUDIO_ERROR_TX.lock();
    if let Some(ref tx) = *tx_guard {
        let _ = tx.try_send(text);
    }
}

/// List available microphone input devices as (name, uid) pairs.
///
/// FR-001/FR-009/FR-010/FR-011 (research.md D1): the app must let a user
/// pick a capture device by stable UID rather than depend on the system
/// default input, which is what let a loopback device silently swallow the
/// user's real microphone in the reported bug. Parsed exactly like
/// `audio_output::list_output_devices` (`audio_output.rs:26-35`) -- same
/// `name|uid` line format, produced by Swift's `audio_list_input_devices`.
pub fn list_input_devices() -> Vec<(String, String)> {
    let ptr = unsafe { audio_list_input_devices() };
    if ptr.is_null() {
        return vec![];
    }
    let raw = unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .to_string();
    unsafe { libc::free(ptr as *mut _) };

    raw.lines()
        .filter_map(|line| {
            let parts: Vec<&str> = line.splitn(2, '|').collect();
            if parts.len() == 2 {
                Some((parts[0].to_string(), parts[1].to_string()))
            } else {
                None
            }
        })
        .collect()
}

/// Prefix Swift uses (`AudioCapture.swift`'s `scheduleRebuildAudioWatchdog`
/// and its aggregate-rate-alignment retry path) to mark a message as a
/// warning rather than a fatal error (T037 Critic round 2 blocker): a
/// rebuild that succeeds but has not yet produced audio despite audio
/// flowing right before it is *suspicious*, not proof the session is dead
/// -- round 1's watchdog fired on this same shape of message and stopped
/// capture outright, which is what turned a normal silent listen-mode
/// session into a killed one.
const WARNING_PREFIX: &str = "WARN:";

/// Blocks until either `stop_rx` receives a stop signal (or is dropped) or
/// `err_rx` receives a message from Swift's device-change machinery
/// (T035, research.md D6) -- whichever comes first. Polls both with
/// `try_recv` rather than blocking on either alone (mirrors
/// `audio_output.rs`'s T017a fix for the same "a single blocking receive
/// never re-checks the other channel" defect): `sck_start_capture` itself
/// returns once, at session start, so a later message -- reported through
/// `audio_error_callback`, not through the FFI's return value -- has no
/// other path back into this function's `Result`.
///
/// A message prefixed `WARN:` (see `WARNING_PREFIX`) is forwarded to
/// `warning_tx` and the loop **continues** -- capture stays up, matching
/// Swift's own non-destructive intent for that class of message. Any other
/// message ends the session with `Err(message)`, unchanged from before.
///
/// A pure function, independent of the real FFI/global statics, so the
/// "rebuild failure surfaces as an error rather than going quiet" and the
/// "a warning does not kill the session" halves of SC-004/FR-008 are both
/// unit-testable without real Core Audio hardware -- `test_device_change.py`'s
/// own hardware-backed tests prove the positive half (audio keeps flowing
/// across a real device change) at the layer that can actually observe it
/// (research.md D9).
fn wait_for_stop_or_rebuild_error(
    mut stop_rx: mpsc::Receiver<()>,
    mut err_rx: mpsc::Receiver<String>,
    warning_tx: mpsc::Sender<String>,
) -> Result<(), String> {
    const POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(20);
    loop {
        match err_rx.try_recv() {
            Ok(message) => {
                if message.starts_with(WARNING_PREFIX) {
                    let _ = warning_tx.try_send(message);
                } else {
                    return Err(message);
                }
            }
            Err(mpsc::error::TryRecvError::Disconnected) => {}
            Err(mpsc::error::TryRecvError::Empty) => {}
        }
        match stop_rx.try_recv() {
            Ok(()) => return Ok(()),
            Err(mpsc::error::TryRecvError::Disconnected) => return Ok(()),
            Err(mpsc::error::TryRecvError::Empty) => {}
        }
        std::thread::sleep(POLL_INTERVAL);
    }
}

/// Start capturing system audio via ScreenCaptureKit.
/// Sends 100ms chunks of float32 mono 16kHz audio as raw bytes via the channel.
/// Each chunk is prefixed with 0x01.
///
/// `warning_tx` (T037 Critic round 2) receives non-fatal warnings (see
/// `WARNING_PREFIX`) as they occur, while capture keeps running -- the
/// caller is expected to forward each one to the user (e.g. `commands.rs`
/// emitting `server-error`) without treating it as session-ending, unlike
/// this function's own `Result`.
pub fn start_capture(
    audio_tx: mpsc::Sender<Vec<u8>>,
    stop_rx: mpsc::Receiver<()>,
    warning_tx: mpsc::Sender<String>,
) -> Result<(), String> {
    // Store the sender globally so the callback can access it (Mutex allows restart)
    *AUDIO_TX.lock() = Some(audio_tx);

    let (err_tx, err_rx) = mpsc::channel::<String>(1);
    *AUDIO_ERROR_TX.lock() = Some(err_tx);

    info!("Starting ScreenCaptureKit audio capture...");
    let ok = unsafe { sck_start_capture(audio_data_callback, audio_error_callback) };
    if !ok {
        *AUDIO_TX.lock() = None;
        *AUDIO_ERROR_TX.lock() = None;
        return Err(
            "Failed to start audio capture. Grant Screen Recording permission in System Settings."
                .to_string(),
        );
    }
    info!("Audio capture started via ScreenCaptureKit");

    let result = wait_for_stop_or_rebuild_error(stop_rx, err_rx, warning_tx);

    unsafe { sck_stop_capture() };
    *AUDIO_TX.lock() = None;
    *AUDIO_ERROR_TX.lock() = None;
    info!("Audio capture stopped");
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T033 (US4) Red: nothing named `wait_for_stop_or_rebuild_error` exists
    /// yet -- `start_capture` only ever blocks on `stop_rx`, so a device-
    /// change rebuild failure reported by Swift (research.md D6, FR-008)
    /// has no path back into this function's `Result` at all. This module
    /// does not even compile until T035 adds it.
    #[test]
    fn rebuild_error_short_circuits_before_a_later_stop_signal() {
        let (stop_tx, stop_rx) = mpsc::channel::<()>(1);
        let (err_tx, err_rx) = mpsc::channel::<String>(1);
        let (warning_tx, _warning_rx) = mpsc::channel::<String>(1);
        err_tx
            .try_send("rebuild failed: could not create aggregate device".to_string())
            .unwrap();
        // A stop signal queued behind the error must not suppress it --
        // the whole point of T035/T036 is that a rebuild failure surfaces
        // rather than the session quietly continuing (or quietly ending)
        // as if nothing happened.
        stop_tx.try_send(()).unwrap();

        let result = wait_for_stop_or_rebuild_error(stop_rx, err_rx, warning_tx);

        assert_eq!(
            result,
            Err("rebuild failed: could not create aggregate device".to_string())
        );
    }

    /// T037 Critic round 2: a `WARN:`-prefixed message must not end the
    /// session (unlike every other message, proven by the test above) --
    /// it is forwarded to `warning_tx` and the loop keeps polling.
    #[test]
    fn warn_message_does_not_return_err_and_is_forwarded_to_warning_channel() {
        let (stop_tx, stop_rx) = mpsc::channel::<()>(1);
        let (err_tx, err_rx) = mpsc::channel::<String>(1);
        let (warning_tx, mut warning_rx) = mpsc::channel::<String>(1);
        err_tx
            .try_send("WARN: audio capture rebuilt but produced no audio within 4s".to_string())
            .unwrap();
        stop_tx.try_send(()).unwrap();

        let result = wait_for_stop_or_rebuild_error(stop_rx, err_rx, warning_tx);

        assert_eq!(result, Ok(()));
        assert_eq!(
            warning_rx.try_recv().unwrap(),
            "WARN: audio capture rebuilt but produced no audio within 4s"
        );
    }

    /// A message that merely *contains* "WARN" without the exact prefix
    /// (case, colon, position) must still end the session -- only the
    /// documented `WARNING_PREFIX` shape is treated as non-fatal.
    #[test]
    fn a_message_only_mentioning_warn_without_the_exact_prefix_still_returns_err() {
        let (stop_tx, stop_rx) = mpsc::channel::<()>(1);
        let (err_tx, err_rx) = mpsc::channel::<String>(1);
        let (warning_tx, mut warning_rx) = mpsc::channel::<String>(1);
        err_tx
            .try_send("rebuild failed, this is not a WARN".to_string())
            .unwrap();
        stop_tx.try_send(()).unwrap();

        let result = wait_for_stop_or_rebuild_error(stop_rx, err_rx, warning_tx);

        assert_eq!(
            result,
            Err("rebuild failed, this is not a WARN".to_string())
        );
        assert!(warning_rx.try_recv().is_err());
    }

    #[test]
    fn stop_signal_alone_returns_ok() {
        let (stop_tx, stop_rx) = mpsc::channel::<()>(1);
        let (_err_tx, err_rx) = mpsc::channel::<String>(1);
        let (warning_tx, _warning_rx) = mpsc::channel::<String>(1);
        stop_tx.try_send(()).unwrap();

        let result = wait_for_stop_or_rebuild_error(stop_rx, err_rx, warning_tx);

        assert_eq!(result, Ok(()));
    }

    /// Mirrors `audio_output.rs`'s T017a fix (`tts_playback_stop_responsiveness.rs`):
    /// a dropped sender (the caller went away) must resolve promptly rather
    /// than hang forever the way `stop_rx.blocking_recv()` alone once did.
    #[test]
    fn dropped_stop_sender_returns_ok_not_hang() {
        let (stop_tx, stop_rx) = mpsc::channel::<()>(1);
        drop(stop_tx);
        let (_err_tx, err_rx) = mpsc::channel::<String>(1);
        let (warning_tx, _warning_rx) = mpsc::channel::<String>(1);

        let result = wait_for_stop_or_rebuild_error(stop_rx, err_rx, warning_tx);

        assert_eq!(result, Ok(()));
    }
}
