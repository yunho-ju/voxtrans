use log::info;
use parking_lot::Mutex;
use std::ffi::CStr;
use std::os::raw::c_char;
use tokio::sync::mpsc;

const CHUNK_SAMPLES: usize = 1600; // 100ms @ 16kHz
pub const AUDIO_PREFIX_SYSTEM: u8 = 0x01;

static AUDIO_TX: Mutex<Option<mpsc::Sender<Vec<u8>>>> = Mutex::new(None);

// FFI to Swift ScreenCaptureKit module
extern "C" {
    fn sck_start_capture(
        callback: extern "C" fn(*const f32, usize, usize, usize) -> (),
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

/// Start capturing system audio via ScreenCaptureKit.
/// Sends 100ms chunks of float32 mono 16kHz audio as raw bytes via the channel.
/// Each chunk is prefixed with 0x01.
pub fn start_capture(
    audio_tx: mpsc::Sender<Vec<u8>>,
    mut stop_rx: mpsc::Receiver<()>,
) -> Result<(), String> {
    // Store the sender globally so the callback can access it (Mutex allows restart)
    *AUDIO_TX.lock() = Some(audio_tx);

    info!("Starting ScreenCaptureKit audio capture...");
    let ok = unsafe { sck_start_capture(audio_data_callback) };
    if !ok {
        *AUDIO_TX.lock() = None;
        return Err(
            "Failed to start audio capture. Grant Screen Recording permission in System Settings."
                .to_string(),
        );
    }
    info!("Audio capture started via ScreenCaptureKit");

    // Block until stop signal (blocking_recv works on non-tokio threads)
    let _ = stop_rx.blocking_recv();

    unsafe { sck_stop_capture() };
    *AUDIO_TX.lock() = None;
    info!("Audio capture stopped");
    Ok(())
}
