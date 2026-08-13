pub mod audio_capture;
pub mod audio_output;
mod commands;
mod mic_capture;
pub mod permissions;
pub mod playback_lifecycle;
mod state;
mod ws_client;

use state::{AppState, SharedState};
use std::sync::Arc;
use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    env_logger::init();

    let shared_state: SharedState = Arc::new(AppState::new());

    tauri::Builder::default()
        // Second launch focuses the running window instead of starting a rival
        // instance: two clients would each open their own capture graph and
        // WebSocket session against one server, and the audio devices they
        // fight over give no useful error when they lose.
        .plugin(tauri_plugin_single_instance::init(|app, _argv, _cwd| {
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.show();
                let _ = window.unminimize();
                let _ = window.set_focus();
            }
        }))
        .plugin(tauri_plugin_shell::init())
        .manage(shared_state)
        .invoke_handler(tauri::generate_handler![
            commands::list_devices,
            commands::list_output_devices,
            commands::get_status,
            commands::set_server_url,
            commands::send_configure,
            commands::send_mic_active,
            commands::send_speak_text,
            commands::set_output_device,
            commands::start_capture,
            commands::stop_capture,
            commands::check_server_health,
            commands::start_server,
            commands::default_server_dir,
            permissions::permission_status,
            permissions::request_permissions,
            permissions::open_permission_settings,
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app_handle, event| {
            // Take the server down with the app. Without this the server it
            // spawned survives as an orphan holding port 8765 and answering
            // /health while resetting every WebSocket -- indistinguishable
            // from a broken app unless you read the client's stderr.
            if let tauri::RunEvent::Exit = event {
                let state = app_handle.state::<SharedState>();
                let child = state.server_child.lock().take();
                if let Some(child) = child {
                    match child.kill() {
                        Ok(()) => log::info!("Spawned server stopped with the app"),
                        Err(e) => log::warn!("Could not stop the spawned server: {e}"),
                    }
                }
            }
        });
}
