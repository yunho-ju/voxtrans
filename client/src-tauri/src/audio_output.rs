use crate::playback_lifecycle::PlaybackLifecycle;
use log::{info, warn};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use tokio::sync::mpsc;

// FFI to Swift audio output
extern "C" {
    fn audio_output_start(device_uid: *const c_char) -> bool;
    fn audio_output_stop();
    fn audio_output_enqueue(data: *const f32, num_samples: usize);
    fn audio_list_output_devices() -> *mut c_char;
    // T025: outstanding-buffer count (`data-model.md` §3 `scheduled_buffers`)
    // -- incremented in `audio_output_enqueue`, decremented by the
    // `.dataPlayedBack` completion callback (`AudioCapture.swift`). Swift's
    // `Int` is word-sized and matches C `long` on every Apple platform this
    // app targets.
    fn audio_output_outstanding_buffers() -> std::os::raw::c_long;
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
///
/// `tts_end_rx` (T026) carries `segment_id`s parsed from the server's
/// `tts_end` messages by `ws_client.rs`'s receive loop -- Swift never sees
/// a segment_id, only Rust does (`playback_lifecycle`'s doc comment). Once
/// `PlaybackLifecycle` decides a segment has both closed its stream and
/// drained its queue, its `playback_complete` is sent on
/// `playback_complete_tx` (the same outbound text-message channel
/// `commands.rs`'s other control messages use, so no new send path is
/// needed on the WebSocket side).
/// Switches the live output stream to `uid` without ending the session.
///
/// Before this existed the device was bound once at `start_tts_playback` and
/// never re-read, so changing the output in settings mid-session appeared to
/// work and silently kept playing to the old device -- the failure that cost an
/// afternoon: the setting said "MacBook 스피커" while TTS was still going to
/// BlackHole, with nothing in the UI or the logs disagreeing.
///
/// A failed switch leaves the stream stopped rather than pretending: the caller
/// reports it, and the next successful switch (or a stop/start) recovers.
fn switch_output_device(uid: &str, name: &str) -> Result<(), String> {
    unsafe { audio_output_stop() };
    let c_uid = CString::new(uid).map_err(|e| format!("Invalid device UID: {}", e))?;
    if unsafe { audio_output_start(c_uid.as_ptr()) } {
        info!(
            "TTS output switched to: {}",
            if uid.is_empty() { "default" } else { uid }
        );
        Ok(())
    } else {
        Err(output_start_error_message(uid, name))
    }
}

pub fn start_tts_playback(
    device_uid: String,
    device_name: String,
    mut tts_rx: mpsc::Receiver<Vec<u8>>,
    mut stop_rx: mpsc::Receiver<()>,
    mut tts_end_rx: mpsc::Receiver<i64>,
    playback_complete_tx: mpsc::Sender<String>,
    mut device_change_rx: mpsc::Receiver<(String, String)>,
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
    let mut lifecycle = PlaybackLifecycle::new();
    'outer: loop {
        match stop_rx.try_recv() {
            Ok(()) => {
                // T026: a segment cut off by stop before it satisfied both
                // completion conditions must not be reported complete (see
                // `PlaybackLifecycle::on_stop`) -- the deadline is its
                // release path instead.
                lifecycle.on_stop();
                break;
            }
            Err(mpsc::error::TryRecvError::Disconnected) => {
                lifecycle.on_stop();
                break;
            }
            Err(mpsc::error::TryRecvError::Empty) => {}
        }

        // Device switches are applied here, between stop checks and draining,
        // so a switch never interleaves with a chunk being enqueued. Only the
        // newest request matters -- older ones in the queue are superseded.
        let mut pending_device: Option<(String, String)> = None;
        while let Ok(selection) = device_change_rx.try_recv() {
            pending_device = Some(selection);
        }
        if let Some((uid, name)) = pending_device {
            if let Err(e) = switch_output_device(&uid, &name) {
                warn!("TTS output switch failed: {}", e);
            }
        }

        // Read tts_end *before* draining, but hold it rather than applying
        // it yet (Critic round-3 follow-up: closing the race the first
        // drain-first fix left open). `tts_end_rx` and `tts_rx` are
        // different channels fed by the same sequential receive loop
        // (`ws_client.rs`), in server-send order -- so the instant this
        // read observes a segment's `tts_end`, that segment's last chunk is
        // *already* sitting in `tts_rx` (pushed strictly earlier in that
        // loop's single sequential body) and cannot un-arrive. Draining
        // `tts_rx` *after* this read (not before it, and not interleaved
        // with it) therefore always reaches that last chunk before
        // completion is evaluated below, with no gap for it to land in
        // between "drain finished" and "tts_end checked" -- the previous
        // ordering (drain, then check `tts_end_rx`) still had exactly that
        // gap: a last chunk and its `tts_end` arriving together in the
        // instant between this tick's drain loop reading `Empty` and its
        // subsequent `tts_end_rx.try_recv()` would have been missed by the
        // drain and then immediately judged complete.
        let pending_segment_end = tts_end_rx.try_recv().ok();

        // Drain-first (Critic round-3 blocker fix): every audio chunk
        // already sitting in `tts_rx` this tick must be enqueued into
        // Swift *before* completion is evaluated below -- see above for why
        // `pending_segment_end` is read first but applied only after this
        // drain completes.
        let mut drained_anything = false;
        loop {
            match tts_rx.try_recv() {
                Ok(audio_bytes) => {
                    drained_anything = true;
                    lifecycle.on_chunk_arrived();
                    let num_samples = audio_bytes.len() / 4;
                    if num_samples > 0 {
                        let data_ptr = audio_bytes.as_ptr() as *const f32;
                        unsafe { audio_output_enqueue(data_ptr, num_samples) };
                    }
                    lifecycle.on_chunk_enqueued();
                }
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    // Channel closed -- no more TTS audio will ever arrive
                    // for this session; the same "don't report a segment
                    // complete that never actually finished" reasoning as
                    // `on_stop` applies here too.
                    lifecycle.on_stop();
                    break 'outer;
                }
                Err(mpsc::error::TryRecvError::Empty) => break,
            }
        }

        // Now that every chunk that could possibly precede it has been
        // drained, apply the segment end read above. `on_stream_ended`
        // itself no-ops a duplicate `tts_end` for the segment already being
        // tracked.
        if let Some(segment_id) = pending_segment_end {
            lifecycle.on_stream_ended(segment_id);
        }

        lifecycle.observe_outstanding_buffers(unsafe { audio_output_outstanding_buffers() } as i64);
        if let Some(segment_id) = lifecycle.take_completed_segment() {
            let message = format!(
                r#"{{"type":"playback_complete","segment_id":{}}}"#,
                segment_id
            );
            if playback_complete_tx.try_send(message).is_err() {
                warn!(
                    "playback_complete channel full or closed, dropping notification for segment_id={}",
                    segment_id
                );
            }
        }

        if !drained_anything {
            std::thread::sleep(POLL_INTERVAL);
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
