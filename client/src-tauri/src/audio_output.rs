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

/// Start TTS playback: receives float32 24kHz audio from tts_rx and plays via device.
pub fn start_tts_playback(
    device_uid: String,
    mut tts_rx: mpsc::Receiver<Vec<u8>>,
    mut stop_rx: mpsc::Receiver<()>,
) -> Result<(), String> {
    let c_uid = CString::new(device_uid.as_str())
        .map_err(|e| format!("Invalid device UID: {}", e))?;

    info!("Starting TTS audio output on device: {}", device_uid);
    let ok = unsafe { audio_output_start(c_uid.as_ptr()) };
    if !ok {
        return Err("Failed to start audio output".to_string());
    }
    info!("TTS audio output started");

    // Use blocking_recv on non-tokio thread
    loop {
        // Check stop signal (non-blocking)
        match stop_rx.try_recv() {
            Ok(()) => break,
            Err(mpsc::error::TryRecvError::Disconnected) => break,
            Err(mpsc::error::TryRecvError::Empty) => {}
        }
        // Block waiting for TTS audio (with timeout to re-check stop)
        match tts_rx.blocking_recv() {
            Some(audio_bytes) => {
                let num_samples = audio_bytes.len() / 4;
                if num_samples > 0 {
                    let data_ptr = audio_bytes.as_ptr() as *const f32;
                    unsafe { audio_output_enqueue(data_ptr, num_samples) };
                }
            }
            None => break, // channel closed
        }
    }

    unsafe { audio_output_stop() };
    info!("TTS audio output stopped");
    Ok(())
}
