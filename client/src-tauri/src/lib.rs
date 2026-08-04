pub mod audio_capture;
pub mod audio_output;
mod commands;
mod mic_capture;
mod state;
mod ws_client;

use state::{AppState, SharedState};
use std::sync::Arc;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    env_logger::init();

    let shared_state: SharedState = Arc::new(AppState::new());

    tauri::Builder::default()
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
            commands::start_capture,
            commands::stop_capture,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
