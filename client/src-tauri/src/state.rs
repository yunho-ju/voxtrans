use parking_lot::Mutex;
use tauri_plugin_shell::process::CommandChild;
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
    /// The server process this app spawned, held so it can be killed when the
    /// app exits. Dropping the handle does *not* kill the process, and an
    /// orphaned server keeps port 18765 and answers `/health` with 200 while
    /// resetting every WebSocket -- a state indistinguishable from a broken app
    /// unless you read the client's stderr.
    pub server_child: Mutex<Option<CommandChild>>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            server_url: Mutex::new("ws://127.0.0.1:18765/ws".to_string()),
            status: Mutex::new(ConnectionStatus::Disconnected),
            is_capturing: Mutex::new(false),
            stop_tx: Mutex::new(None),
            ws_text_tx: Mutex::new(None),
            stop_mic_tx: Mutex::new(None),
            tts_audio_tx: Mutex::new(None),
            stop_tts_tx: Mutex::new(None),
            output_device_tx: Mutex::new(None),
            server_child: Mutex::new(None),
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

pub type SharedState = Arc<AppState>;
