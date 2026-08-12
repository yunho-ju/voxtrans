//! macOS capture permissions, asked for up front instead of discovered on failure.
//!
//! Without this, a missing Screen Recording grant surfaced only as
//! `sck_start_capture` returning false: the session ended immediately and the
//! UI bounced start back to stop with no explanation. Worse, a rebuilt bundle
//! can invalidate a grant while System Settings still shows it enabled (the TCC
//! row is bound to the previous signature), so "check the settings pane" is not
//! a reliable answer either. Preflight reports the effective state.

use std::ffi::CString;

extern "C" {
    fn perm_check_screen_capture() -> bool;
    fn perm_request_screen_capture() -> bool;
    fn perm_check_microphone() -> bool;
    fn perm_request_microphone() -> bool;
    fn perm_open_settings(kind: *const std::os::raw::c_char);
}

/// What the app may do right now, per capture path.
#[derive(serde::Serialize, Clone, Debug, PartialEq)]
pub struct PermissionStatus {
    /// Screen & System Audio Recording -- required for listen mode.
    pub screen: bool,
    /// Microphone -- required for speak mode.
    pub microphone: bool,
}

pub fn status() -> PermissionStatus {
    PermissionStatus {
        screen: unsafe { perm_check_screen_capture() },
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
        screen: unsafe { perm_request_screen_capture() },
        microphone: unsafe { perm_request_microphone() },
    })
    .await
    .map_err(|e| format!("Permission request failed: {}", e))
}

#[tauri::command]
pub fn open_permission_settings(kind: String) -> Result<(), String> {
    let which = match kind.as_str() {
        "microphone" | "screen" => kind,
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
