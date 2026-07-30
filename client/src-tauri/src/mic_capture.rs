use log::info;
use parking_lot::Mutex;
use tokio::sync::mpsc;

const CHUNK_SAMPLES: usize = 1600; // 100ms @ 16kHz
pub const AUDIO_PREFIX_MIC: u8 = 0x02;

static MIC_AUDIO_TX: Mutex<Option<mpsc::Sender<Vec<u8>>>> = Mutex::new(None);

// FFI to Swift mic capture
extern "C" {
    fn mic_start_capture(
        callback: extern "C" fn(*const f32, usize, usize, usize) -> (),
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

/// Start capturing microphone audio.
/// Sends 100ms chunks of float32 mono 16kHz audio prefixed with 0x02.
pub fn start_mic_capture(
    mic_tx: mpsc::Sender<Vec<u8>>,
    mut stop_rx: mpsc::Receiver<()>,
) -> Result<(), String> {
    *MIC_AUDIO_TX.lock() = Some(mic_tx);

    info!("Starting microphone capture...");
    let ok = unsafe { mic_start_capture(mic_data_callback) };
    if !ok {
        *MIC_AUDIO_TX.lock() = None;
        return Err(
            "Failed to start mic capture. Grant Microphone permission in System Settings."
                .to_string(),
        );
    }
    info!("Microphone capture started");

    // Block until stop signal (blocking_recv works on non-tokio threads)
    let _ = stop_rx.blocking_recv();

    unsafe { mic_stop_capture() };
    *MIC_AUDIO_TX.lock() = None;
    info!("Microphone capture stopped");
    Ok(())
}
