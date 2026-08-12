use parking_lot::Mutex;
use std::sync::Arc;
use tokio::sync::mpsc;

#[derive(Clone, Debug, PartialEq)]
pub enum ConnectionStatus {
    Disconnected,
    Connecting,
    Connected,
}

pub struct AppState {
    pub server_url: Mutex<String>,
    pub status: Mutex<ConnectionStatus>,
    pub is_capturing: Mutex<bool>,
    /// Channel to send stop signals to the audio+ws pipeline
    pub stop_tx: Mutex<Option<mpsc::Sender<()>>>,
    /// Channel to send text (control) messages to WebSocket
    pub ws_text_tx: Mutex<Option<mpsc::Sender<String>>>,
    /// Channel to send stop signal to mic capture thread
    pub stop_mic_tx: Mutex<Option<mpsc::Sender<()>>>,
    /// Channel to send TTS audio data for playback
    pub tts_audio_tx: Mutex<Option<mpsc::Sender<Vec<u8>>>>,
    /// Channel to send stop signal to TTS output thread
    pub stop_tts_tx: Mutex<Option<mpsc::Sender<()>>>,
    /// Channel carrying `(uid, name)` output-device switches to the running
    /// TTS playback thread. Without it the device is fixed for the session and
    /// changing it in settings does nothing until stop/start.
    pub output_device_tx: Mutex<Option<mpsc::Sender<(String, String)>>>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            server_url: Mutex::new("ws://127.0.0.1:8765/ws".to_string()),
            status: Mutex::new(ConnectionStatus::Disconnected),
            is_capturing: Mutex::new(false),
            stop_tx: Mutex::new(None),
            ws_text_tx: Mutex::new(None),
            stop_mic_tx: Mutex::new(None),
            tts_audio_tx: Mutex::new(None),
            stop_tts_tx: Mutex::new(None),
            output_device_tx: Mutex::new(None),
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

pub type SharedState = Arc<AppState>;
