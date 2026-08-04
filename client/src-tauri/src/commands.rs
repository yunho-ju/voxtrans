use crate::audio_capture;
use crate::audio_output;
use crate::mic_capture;
use crate::state::{ConnectionStatus, SharedState};
use crate::ws_client;
use log::{error, info};
use tauri::{AppHandle, Emitter, State};
use tokio::sync::mpsc;
use std::path::PathBuf;

/// Called when user clicks "녹화 시작".
/// Generates a timestamped meeting file, writes ~/meeting_current.json,
/// then opens a Terminal window running `claude /meeting-start <path>`.
#[tauri::command]
pub async fn start_meeting(
    title: String,
    codebase_path: Option<String>,
    jira_project: Option<String>,
) -> Result<String, String> {
    let home = std::env::var("HOME").map_err(|_| "HOME not set".to_string())?;
    let meetings_dir = PathBuf::from(&home).join("meetings");
    std::fs::create_dir_all(&meetings_dir)
        .map_err(|e| format!("Failed to create meetings dir: {e}"))?;

    let now = chrono::Local::now();
    let filename = meetings_dir
        .join(format!("meeting_{}.md", now.format("%Y%m%d_%H%M")))
        .to_string_lossy()
        .to_string();

    // Write ~/meeting_current.json so the skill and server know the active file
    let config = serde_json::json!({
        "title": title,
        "filename": filename,
        "started_at": now.to_rfc3339(),
        "codebase_path": codebase_path.unwrap_or_default(),
        "jira_project": jira_project.unwrap_or_default(),
    });
    let config_path = PathBuf::from(&home).join("meeting_current.json");
    std::fs::write(&config_path, config.to_string())
        .map_err(|e| format!("Failed to write meeting_current.json: {e}"))?;

    // Open Terminal running claude with the meeting-assistant skill
    let claude_cmd = format!(
        "claude -p '회의 시작됨. ~/meeting_current.json 읽고 /meeting-start 실행. 이후 자동으로 /loop /meeting-monitor 시작.'",
    );
    let script = format!(
        r#"tell application "Terminal"
    activate
    do script "{claude_cmd}"
end tell"#
    );
    std::process::Command::new("osascript")
        .arg("-e")
        .arg(&script)
        .spawn()
        .map_err(|e| format!("Failed to open Terminal: {e}"))?;

    info!("Meeting started: {filename}");
    Ok(filename)
}

/// Called when user clicks "녹화 종료".
/// Appends the MEETING_END marker so the skill generates the final document.
#[tauri::command]
pub async fn stop_meeting() -> Result<(), String> {
    let home = std::env::var("HOME").map_err(|_| "HOME not set".to_string())?;
    let config_path = PathBuf::from(&home).join("meeting_current.json");
    let raw = std::fs::read_to_string(&config_path)
        .map_err(|_| "No active meeting".to_string())?;
    let config: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|_| "Invalid meeting_current.json".to_string())?;
    let filename = config["filename"].as_str().unwrap_or_default().to_string();

    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(&filename)
        .map_err(|e| format!("Failed to open meeting file: {e}"))?;
    use std::io::Write;
    writeln!(file, "\n<!-- MEETING_END -->")
        .map_err(|e| format!("Failed to write end marker: {e}"))?;

    info!("Meeting ended: {filename}");
    Ok(())
}

#[tauri::command]
pub fn list_devices() -> Vec<Vec<String>> {
    audio_capture::list_input_devices()
        .into_iter()
        .map(|(name, uid)| vec![name, uid])
        .collect()
}

#[tauri::command]
pub fn list_output_devices() -> Vec<Vec<String>> {
    audio_output::list_output_devices()
        .into_iter()
        .map(|(name, uid)| vec![name, uid])
        .collect()
}

#[tauri::command]
pub fn get_status(state: State<SharedState>) -> String {
    let status = state.status.lock();
    match *status {
        ConnectionStatus::Disconnected => "disconnected".to_string(),
        ConnectionStatus::Connecting => "connecting".to_string(),
        ConnectionStatus::Connected => "connected".to_string(),
    }
}

#[tauri::command]
pub fn set_server_url(state: State<SharedState>, url: String) {
    *state.server_url.lock() = url;
}

#[tauri::command]
pub async fn send_configure(
    state: State<'_, SharedState>,
    language: String,
    target_language: String,
    mode: Option<String>,
    tts_voice: Option<String>,
    speak_target_language: Option<String>,
    voice_clone: Option<bool>,
    mic_mode: Option<String>,
) -> Result<(), String> {
    let tx = state.ws_text_tx.lock().clone();
    if let Some(tx) = tx {
        let mut json = serde_json::json!({
            "type": "configure",
            "language": language,
            "target_language": target_language,
        });
        if let Some(m) = mode {
            json["mode"] = serde_json::Value::String(m);
        }
        if let Some(v) = tts_voice {
            json["tts_voice"] = serde_json::Value::String(v);
        }
        if let Some(s) = speak_target_language {
            json["speak_target_language"] = serde_json::Value::String(s);
        }
        if let Some(vc) = voice_clone {
            json["voice_clone"] = serde_json::Value::Bool(vc);
        }
        if let Some(mm) = mic_mode {
            json["mic_mode"] = serde_json::Value::String(mm);
        }
        tx.send(json.to_string())
            .await
            .map_err(|e| format!("Failed to send configure: {}", e))?;
    } else {
        return Err("Not connected".to_string());
    }
    Ok(())
}

#[tauri::command]
pub async fn send_mic_active(
    state: State<'_, SharedState>,
    active: bool,
) -> Result<(), String> {
    let tx = state.ws_text_tx.lock().clone();
    if let Some(tx) = tx {
        let json = serde_json::json!({
            "type": "mic_active",
            "active": active,
        });
        tx.send(json.to_string())
            .await
            .map_err(|e| format!("Failed to send mic_active: {}", e))?;
    } else {
        return Err("Not connected".to_string());
    }
    Ok(())
}

#[tauri::command]
pub async fn send_speak_text(
    state: State<'_, SharedState>,
    text: String,
) -> Result<(), String> {
    let tx = state.ws_text_tx.lock().clone();
    if let Some(tx) = tx {
        let json = serde_json::json!({
            "type": "speak_text",
            "text": text,
        });
        tx.send(json.to_string())
            .await
            .map_err(|e| format!("Failed to send speak_text: {}", e))?;
    } else {
        return Err("Not connected".to_string());
    }
    Ok(())
}

#[tauri::command]
pub async fn start_capture(
    app: AppHandle,
    state: State<'_, SharedState>,
    output_device_uid: Option<String>,
    input_device_uid: Option<String>,
) -> Result<(), String> {
    // Check if already capturing
    if *state.is_capturing.lock() {
        return Err("Already capturing".to_string());
    }

    let server_url = state.server_url.lock().clone();

    // System audio channels
    let (audio_tx, audio_rx) = mpsc::channel::<Vec<u8>>(100);
    let (stop_audio_tx, stop_audio_rx) = mpsc::channel::<()>(1);
    let (text_tx, text_rx) = mpsc::channel::<String>(32);

    // Mic audio channels
    let (mic_tx, mic_rx) = mpsc::channel::<Vec<u8>>(100);
    let (stop_mic_tx, stop_mic_rx) = mpsc::channel::<()>(1);

    // TTS audio channels
    let (tts_tx, tts_rx) = mpsc::channel::<Vec<u8>>(200);
    let (stop_tts_tx, stop_tts_rx) = mpsc::channel::<()>(1);

    // Store senders
    *state.stop_tx.lock() = Some(stop_audio_tx);
    *state.ws_text_tx.lock() = Some(text_tx);
    *state.stop_mic_tx.lock() = Some(stop_mic_tx);
    *state.tts_audio_tx.lock() = Some(tts_tx.clone());
    *state.stop_tts_tx.lock() = Some(stop_tts_tx);
    *state.is_capturing.lock() = true;
    *state.status.lock() = ConnectionStatus::Connecting;
    let _ = app.emit("connection-status", "connecting");

    let app_ws = app.clone();
    let state_inner = (*state).clone();

    // Spawn WebSocket task (now with mic_rx and tts_tx)
    tokio::spawn(async move {
        match ws_client::run_ws_pipeline(
            app_ws.clone(),
            server_url,
            audio_rx,
            mic_rx,
            text_rx,
            tts_tx,
        )
        .await
        {
            Ok(()) => info!("WebSocket pipeline finished"),
            Err(e) => {
                error!("WebSocket pipeline error: {}", e);
                let _ = app_ws.emit("server-error", e);
            }
        }
        *state_inner.is_capturing.lock() = false;
        *state_inner.status.lock() = ConnectionStatus::Disconnected;
        *state_inner.ws_text_tx.lock() = None;
    });

    // Update status
    *state.status.lock() = ConnectionStatus::Connected;
    let _ = app.emit("connection-status", "connected");

    // Spawn system audio capture on a blocking thread
    let state_audio = (*state).clone();
    let app_audio = app.clone();
    std::thread::spawn(move || {
        if let Err(e) = audio_capture::start_capture(audio_tx, stop_audio_rx) {
            error!("Audio capture error: {}", e);
            let _ = app_audio.emit("server-error", e);
        }
        *state_audio.is_capturing.lock() = false;
    });

    // Spawn mic capture on a blocking thread
    let mic_device_uid = input_device_uid.unwrap_or_default();
    let app_mic = app.clone();
    std::thread::spawn(move || {
        if let Err(e) = mic_capture::start_mic_capture(mic_tx, stop_mic_rx, mic_device_uid) {
            error!("Mic capture error: {}", e);
            let _ = app_mic.emit("server-error", e);
        }
    });

    // Spawn TTS playback on a blocking thread
    let device_uid = output_device_uid.unwrap_or_default();
    let app_tts = app.clone();
    std::thread::spawn(move || {
        if let Err(e) = audio_output::start_tts_playback(device_uid, tts_rx, stop_tts_rx) {
            error!("TTS playback error: {}", e);
            let _ = app_tts.emit("server-error", e);
        }
    });

    info!("Capture started (system audio + mic + TTS output)");
    Ok(())
}

#[tauri::command]
pub async fn stop_capture(state: State<'_, SharedState>) -> Result<(), String> {
    // Stop system audio
    let tx = state.stop_tx.lock().take();
    if let Some(tx) = tx {
        let _ = tx.send(()).await;
    }
    // Stop mic
    let mic_tx = state.stop_mic_tx.lock().take();
    if let Some(tx) = mic_tx {
        let _ = tx.send(()).await;
    }
    // Stop TTS output
    let tts_tx = state.stop_tts_tx.lock().take();
    if let Some(tx) = tts_tx {
        let _ = tx.send(()).await;
    }
    *state.is_capturing.lock() = false;
    *state.status.lock() = ConnectionStatus::Disconnected;
    *state.ws_text_tx.lock() = None;
    *state.tts_audio_tx.lock() = None;
    info!("Capture stopped");
    Ok(())
}
