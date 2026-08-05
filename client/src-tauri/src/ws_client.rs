use futures_util::{SinkExt, StreamExt};
use log::{error, info, warn};
use tauri::{AppHandle, Emitter};
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};

const AUDIO_PREFIX_TTS: u8 = 0x03;

/// Run the WebSocket client pipeline:
/// - Receives audio chunks from audio_rx (system audio, 0x01 prefixed) and sends to server
/// - Receives mic chunks from mic_audio_rx (mic audio, 0x02 prefixed) and sends to server
/// - Receives text messages from text_rx and sends to server
/// - Receives server messages: JSON → Tauri events, binary TTS → tts_audio_tx
/// - Forwards `tts_end`'s `segment_id` to `tts_end_tx` (T026): the TTS
///   playback thread (`audio_output::start_tts_playback`) has no other way
///   to learn a segment's stream has closed -- Swift never sees a
///   segment_id, only this receive loop parses one off the wire (see
///   `playback_lifecycle`'s doc comment).
pub async fn run_ws_pipeline(
    app: AppHandle,
    url: String,
    mut audio_rx: mpsc::Receiver<Vec<u8>>,
    mut mic_audio_rx: mpsc::Receiver<Vec<u8>>,
    mut text_rx: mpsc::Receiver<String>,
    tts_audio_tx: mpsc::Sender<Vec<u8>>,
    tts_end_tx: mpsc::Sender<i64>,
) -> Result<(), String> {
    info!("Connecting to {}", url);
    let (ws_stream, _) = connect_async(&url)
        .await
        .map_err(|e| format!("WebSocket connect failed: {}", e))?;

    let (mut ws_tx, mut ws_rx) = ws_stream.split();
    info!("WebSocket connected");

    let app_recv = app.clone();

    // Task: receive messages from server
    let recv_task = tokio::spawn(async move {
        while let Some(msg) = ws_rx.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                        let msg_type = json.get("type").and_then(|t| t.as_str()).unwrap_or("");
                        match msg_type {
                            "transcription_final" | "transcription_partial" => {
                                let _ = app_recv.emit("transcription", text.to_string());
                            }
                            "translation" => {
                                let _ = app_recv.emit("translation", text.to_string());
                            }
                            "speak_transcription" => {
                                let _ = app_recv.emit("speak-transcription", text.to_string());
                            }
                            "speak_translation" => {
                                let _ = app_recv.emit("speak-translation", text.to_string());
                            }
                            "tts_end" => {
                                if let Some(segment_id) =
                                    json.get("segment_id").and_then(|v| v.as_i64())
                                {
                                    if tts_end_tx.try_send(segment_id).is_err() {
                                        warn!(
                                            "tts_end channel full or closed, dropping \
                                             segment_id={}",
                                            segment_id
                                        );
                                    }
                                }
                                let _ = app_recv.emit("tts-end", text.to_string());
                            }
                            "status" => {
                                let _ = app_recv.emit("server-status", text.to_string());
                            }
                            "error" => {
                                warn!("Server error: {}", text);
                                let _ = app_recv.emit("server-error", text.to_string());
                            }
                            _ => {
                                info!("Server message: {}", text);
                            }
                        }
                    }
                }
                Ok(Message::Binary(data)) => {
                    // Binary from server: check prefix
                    if !data.is_empty() && data[0] == AUDIO_PREFIX_TTS {
                        // TTS audio: strip prefix and forward to playback
                        let audio_data = data[1..].to_vec();
                        if tts_audio_tx.try_send(audio_data).is_err() {
                            warn!("TTS audio channel full, dropping chunk");
                        }
                    }
                }
                Ok(Message::Close(_)) => {
                    info!("WebSocket closed by server");
                    break;
                }
                Err(e) => {
                    error!("WebSocket recv error: {}", e);
                    break;
                }
                _ => {}
            }
        }
        let _ = app_recv.emit("connection-status", "disconnected");
    });

    // Task: send audio chunks and text messages to server
    let send_task = tokio::spawn(async move {
        loop {
            tokio::select! {
                Some(chunk) = audio_rx.recv() => {
                    // Already has 0x01 prefix from audio_capture
                    if let Err(e) = ws_tx.send(Message::Binary(chunk.into())).await {
                        error!("WebSocket send error: {}", e);
                        break;
                    }
                }
                Some(chunk) = mic_audio_rx.recv() => {
                    // Already has 0x02 prefix from mic_capture
                    if let Err(e) = ws_tx.send(Message::Binary(chunk.into())).await {
                        error!("WebSocket mic send error: {}", e);
                        break;
                    }
                }
                Some(text) = text_rx.recv() => {
                    if let Err(e) = ws_tx.send(Message::Text(text.into())).await {
                        error!("WebSocket text send error: {}", e);
                        break;
                    }
                }
                else => break,
            }
        }
        // Channels closed (capture stopped), close WebSocket
        info!("Channels closed, closing WebSocket");
        let _ = ws_tx.close().await;
    });

    // Wait for either task to finish
    tokio::select! {
        _ = recv_task => {},
        _ = send_task => {},
    }

    let _ = app.emit("connection-status", "disconnected");
    Ok(())
}
