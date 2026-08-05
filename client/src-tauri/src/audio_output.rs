use log::info;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use tokio::sync::mpsc;

// FFI to Swift audio output
extern "C" {
    fn audio_output_start(device_uid: *const c_char) -> bool;
    fn audio_output_stop();
    fn audio_output_enqueue(data: *const f32, num_samples: usize);
    fn audio_list_output_devices() -> *mut c_char;
}

/// List available output devices as (name, uid) pairs.
pub fn list_output_devices() -> Vec<(String, String)> {
    let ptr = unsafe { audio_list_output_devices() };
    if ptr.is_null() {
        return vec![];
    }
    let raw = unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .to_string();
    unsafe { libc::free(ptr as *mut _) };

    // Format: "name|uid\nname|uid\n..."
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

/// Builds `start_tts_playback`'s `Err` message for a failed
/// `audio_output_start`, naming `device_uid` and (when known) `device_name`
/// (FR-009) -- split out from `start_tts_playback` itself so its three
/// branches are unit-testable without needing a real (or absent) audio
/// device to force `audio_output_start` to fail (Critic round 2, US1 round
/// 2 major finding 1: the previous inline version was reachable by no test
/// anywhere, so reverting it to the old constant string, or breaking the
/// caller's name-threading, changed nothing observable in the whole suite).
fn output_start_error_message(device_uid: &str, device_name: &str) -> String {
    if device_uid.is_empty() {
        "Failed to start audio output. Check your system's audio output settings.".to_string()
    } else if device_name.is_empty() {
        format!(
            "Failed to start audio output on device '{}'. The device may be missing, \
             removed, or disconnected.",
            device_uid
        )
    } else {
        format!(
            "Failed to start audio output on device '{}' ('{}'). The device may be missing, \
             removed, or disconnected.",
            device_uid, device_name
        )
    }
}

/// Start TTS playback: receives float32 24kHz audio from tts_rx and plays via device.
///
/// `device_name` is the last name known for `device_uid` at the moment the
/// user selected it (captured by the frontend from its device list, not
/// re-queried here) -- once the device has actually disappeared, a fresh
/// `list_output_devices()` lookup can no longer find a name for it, so the
/// caller must thread the name through at selection time (FR-009; see
/// `mic_capture::start_mic_capture` for the same pattern on the input side).
pub fn start_tts_playback(
    device_uid: String,
    device_name: String,
    mut tts_rx: mpsc::Receiver<Vec<u8>>,
    mut stop_rx: mpsc::Receiver<()>,
) -> Result<(), String> {
    let c_uid = CString::new(device_uid.as_str())
        .map_err(|e| format!("Invalid device UID: {}", e))?;

    info!(
        "Starting TTS audio output on device: {}",
        if device_uid.is_empty() {
            "default"
        } else {
            device_uid.as_str()
        }
    );
    let ok = unsafe { audio_output_start(c_uid.as_ptr()) };
    if !ok {
        return Err(output_start_error_message(&device_uid, &device_name));
    }
    info!("TTS audio output started");

    // T017a (Edge Case "Output device disappears mid-playback"): this used
    // to `tts_rx.blocking_recv()` with no timeout, so if no TTS chunk ever
    // arrived after `stop` was signalled, the stop check above (a
    // non-blocking `try_recv()`, run only *before* each blocking receive)
    // was never re-run and this thread never exited -- reproduced by
    // `tests/tts_playback_stop_responsiveness.rs`. Polling `try_recv()` on
    // both channels re-checks stop every `POLL_INTERVAL`, bounding how long
    // a stop signal can go unnoticed regardless of whether TTS audio is
    // still arriving.
    const POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(20);
    loop {
        match stop_rx.try_recv() {
            Ok(()) => break,
            Err(mpsc::error::TryRecvError::Disconnected) => break,
            Err(mpsc::error::TryRecvError::Empty) => {}
        }
        match tts_rx.try_recv() {
            Ok(audio_bytes) => {
                let num_samples = audio_bytes.len() / 4;
                if num_samples > 0 {
                    let data_ptr = audio_bytes.as_ptr() as *const f32;
                    unsafe { audio_output_enqueue(data_ptr, num_samples) };
                }
            }
            Err(mpsc::error::TryRecvError::Disconnected) => break, // channel closed
            Err(mpsc::error::TryRecvError::Empty) => {
                std::thread::sleep(POLL_INTERVAL);
            }
        }
    }

    unsafe { audio_output_stop() };
    info!("TTS audio output stopped");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Empty `device_uid` means "system default" (contracts/audio-devices.md);
    /// this branch is unreachable from `device_identity.rs`'s real-hardware
    /// tests (`audio_output_start("")` succeeds on any Mac with a default
    /// output device), so it is only unit-testable here.
    #[test]
    fn empty_uid_reports_the_generic_default_output_message_and_names_no_device() {
        let msg = output_start_error_message("", "");
        assert!(
            !msg.contains("device '"),
            "expected the default-output message to name no specific device (no \"device \
             '...'\" segment, the pattern the named-device branches use), got {msg:?}"
        );
    }

    #[test]
    fn a_uid_with_no_known_name_is_reported_alone() {
        let msg = output_start_error_message("com.example.uid-123", "");
        assert!(
            msg.contains("com.example.uid-123"),
            "expected the UID in the message, got {msg:?}"
        );
        assert!(
            !msg.contains("('"),
            "expected no name segment when no name is known, got {msg:?}"
        );
    }

    #[test]
    fn a_uid_with_a_known_name_reports_both() {
        let msg = output_start_error_message("com.example.uid-123", "Example Speakers");
        assert!(
            msg.contains("com.example.uid-123"),
            "expected the UID in the message, got {msg:?}"
        );
        assert!(
            msg.contains("Example Speakers"),
            "expected the last-known name in the message, got {msg:?}"
        );
    }
}
