use log::info;
use parking_lot::Mutex;
use std::ffi::CString;
use std::os::raw::c_char;
use tokio::sync::mpsc;

const CHUNK_SAMPLES: usize = 1600; // 100ms @ 16kHz
pub const AUDIO_PREFIX_MIC: u8 = 0x02;

static MIC_AUDIO_TX: Mutex<Option<mpsc::Sender<Vec<u8>>>> = Mutex::new(None);

// FFI to Swift mic capture
extern "C" {
    fn mic_start_capture(
        callback: extern "C" fn(*const f32, usize, usize, usize) -> (),
        device_uid: *const c_char,
    ) -> bool;
    fn mic_stop_capture();
}

/// Called from Swift with PCM float32 mono 16kHz mic audio data
extern "C" fn mic_data_callback(
    data: *const f32,
    num_samples: usize,
    _sample_rate: usize,
    _channels: usize,
) {
    if data.is_null() || num_samples == 0 {
        return;
    }

    let samples = unsafe { std::slice::from_raw_parts(data, num_samples) };

    let tx_guard = MIC_AUDIO_TX.lock();
    if let Some(ref tx) = *tx_guard {
        thread_local! {
            static BUFFER: std::cell::RefCell<Vec<f32>> = std::cell::RefCell::new(Vec::with_capacity(CHUNK_SAMPLES * 2));
        }

        BUFFER.with(|buf| {
            let mut buf = buf.borrow_mut();
            buf.extend_from_slice(samples);

            while buf.len() >= CHUNK_SAMPLES {
                let chunk: Vec<f32> = buf.drain(..CHUNK_SAMPLES).collect();
                let audio_bytes: &[u8] = bytemuck::cast_slice(&chunk);
                // Prepend 0x02 prefix for mic audio
                let mut bytes = Vec::with_capacity(1 + audio_bytes.len());
                bytes.push(AUDIO_PREFIX_MIC);
                bytes.extend_from_slice(audio_bytes);
                if tx.try_send(bytes).is_err() {
                    // Channel full, drop chunk
                }
            }
        });
    }
}

/// Start capturing microphone audio from `device_uid` (empty means the
/// system default input -- an explicit choice now, rather than the only
/// possibility; see research.md D0/D1). A non-empty UID that cannot be
/// resolved fails the start and names the UID in the returned error; it
/// never silently falls back to the default (FR-009).
/// Sends 100ms chunks of float32 mono 16kHz audio prefixed with 0x02.
pub fn start_mic_capture(
    mic_tx: mpsc::Sender<Vec<u8>>,
    mut stop_rx: mpsc::Receiver<()>,
    device_uid: String,
) -> Result<(), String> {
    *MIC_AUDIO_TX.lock() = Some(mic_tx);

    let c_uid =
        CString::new(device_uid.as_str()).map_err(|e| format!("Invalid device UID: {}", e))?;

    info!(
        "Starting microphone capture on device: {}",
        if device_uid.is_empty() {
            "default"
        } else {
            device_uid.as_str()
        }
    );
    let ok = unsafe { mic_start_capture(mic_data_callback, c_uid.as_ptr()) };
    if !ok {
        *MIC_AUDIO_TX.lock() = None;
        return Err(if device_uid.is_empty() {
            "Failed to start mic capture. Grant Microphone permission in System Settings."
                .to_string()
        } else {
            format!(
                "Failed to start mic capture on device '{}'. The device may be missing, \
                 removed, or microphone permission may not be granted.",
                device_uid
            )
        });
    }
    info!("Microphone capture started");

    // Block until stop signal (blocking_recv works on non-tokio threads)
    let _ = stop_rx.blocking_recv();

    unsafe { mic_stop_capture() };
    *MIC_AUDIO_TX.lock() = None;
    info!("Microphone capture stopped");
    Ok(())
}
