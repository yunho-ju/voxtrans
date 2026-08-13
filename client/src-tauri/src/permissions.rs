//! macOS microphone permission, asked for up front instead of discovered on
//! failure.
//!
//! Only the microphone is checked here. System audio goes through a Core Audio
//! process tap, whose permission ("System Audio Recording Only") has no
//! preflight API, and the only way to test one is to build a tap -- which means
//! touching the audio graph on every launch. So it is not guessed at:
//! `sck_start_capture` reports its real failure and the capture-start path
//! surfaces that.
//!
//! `CGPreflightScreenCaptureAccess` is deliberately not used. It answers for
//! Screen Recording, which this app never requests, so its answer says nothing
//! about whether system audio will work.

use std::ffi::CString;

extern "C" {
    fn perm_check_microphone() -> bool;
    fn perm_request_microphone() -> bool;
    fn perm_open_settings(kind: *const std::os::raw::c_char);
}

/// What the app may do right now. System audio is deliberately absent: see the
/// module docs for why guessing at it did more harm than good.
#[derive(serde::Serialize, Clone, Debug, PartialEq)]
pub struct PermissionStatus {
    /// Microphone -- required to speak. Subtitles do not need it.
    pub microphone: bool,
}

pub fn status() -> PermissionStatus {
    PermissionStatus {
        microphone: unsafe { perm_check_microphone() },
    }
}

#[tauri::command]
pub fn permission_status() -> PermissionStatus {
    status()
}

/// Triggers the system dialogs for anything not yet granted, then reports the
/// resulting state. macOS shows each prompt only once per app signature; after
/// a denial only System Settings can change the answer, which is why the UI
/// pairs this with `open_permission_settings`.
#[tauri::command]
pub async fn request_permissions() -> Result<PermissionStatus, String> {
    // `perm_request_microphone` blocks on a semaphore until the user answers,
    // so this must not run on the UI thread.
    tauri::async_runtime::spawn_blocking(|| PermissionStatus {
        microphone: unsafe { perm_request_microphone() },
    })
    .await
    .map_err(|e| format!("Permission request failed: {}", e))
}

#[tauri::command]
pub fn open_permission_settings(kind: String) -> Result<(), String> {
    let which = match kind.as_str() {
        "microphone" | "system_audio" => kind,
        other => return Err(format!("Unknown permission kind: {}", other)),
    };
    let c_kind = CString::new(which).map_err(|e| e.to_string())?;
    unsafe { perm_open_settings(c_kind.as_ptr()) };
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_permission_settings_rejects_an_unknown_kind() {
        // Guards the FFI boundary: only the two strings the Swift side maps to
        // a settings anchor may cross it, so a typo in the UI fails loudly here
        // rather than silently opening the wrong pane.
        let err = open_permission_settings("bluetooth".to_string()).unwrap_err();
        assert!(err.contains("bluetooth"), "error should name the bad kind: {}", err);
    }
}
