use crate::audio_capture;
use crate::audio_output;
use crate::mic_capture;
use crate::state::{ConnectionStatus, SharedState};
use crate::ws_client;
use log::{error, info};
use tauri::{AppHandle, Emitter, State};
use tokio::sync::mpsc;
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::PathBuf;
use std::sync::Mutex as StdMutex;
use std::time::{Duration, Instant};

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

/// Resolves the `(uid, name)` pair `start_capture` passes on to
/// `audio_output::start_tts_playback` from the two `Option<String>`
/// parameters `start_capture` receives over Tauri's IPC boundary. Pulled
/// out as its own function so this one-line pass-through -- silently
/// breaking here (e.g. discarding `output_device_name`) is invisible to
/// every test that calls `audio_output::start_tts_playback` directly,
/// since `start_capture` itself needs a live `AppHandle`/`State` no test in
/// this suite constructs -- has something narrow enough to unit test
/// (Critic round 2, US1 round 2 major finding 1, mutation (b)).
fn resolve_output_device_selection(
    output_device_uid: Option<String>,
    output_device_name: Option<String>,
) -> (String, String) {
    (
        output_device_uid.unwrap_or_default(),
        output_device_name.unwrap_or_default(),
    )
}

#[tauri::command]
pub async fn start_capture(
    app: AppHandle,
    state: State<'_, SharedState>,
    output_device_uid: Option<String>,
    output_device_name: Option<String>,
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
    // T026: tts_end segment_id notifications (ws_client -> TTS playback
    // thread) and the outbound channel `start_tts_playback` sends
    // `playback_complete` on -- a clone of the same sender other control
    // messages already use (`send_configure` etc.), not a second send path.
    let (tts_end_tx, tts_end_rx) = mpsc::channel::<i64>(8);

    // Store senders
    *state.stop_tx.lock() = Some(stop_audio_tx);
    *state.ws_text_tx.lock() = Some(text_tx.clone());
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
            tts_end_tx,
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

    // Spawn system audio capture on a blocking thread. `warning_tx` (T037
    // Critic round 2) carries non-fatal, `WARN:`-prefixed reports from a
    // device-change rebuild that has not yet produced audio -- forwarded
    // to the same `server-error` event the fatal path already uses (the
    // frontend banner needs no new event type, per the confirmed design),
    // but on its own async task so it can emit *while* the capture thread
    // below is still blocked polling, not only after the whole session ends.
    let (warning_tx, mut warning_rx) = mpsc::channel::<String>(8);
    let app_warning = app.clone();
    tokio::spawn(async move {
        while let Some(message) = warning_rx.recv().await {
            let _ = app_warning.emit("server-error", message);
        }
    });

    let state_audio = (*state).clone();
    let app_audio = app.clone();
    std::thread::spawn(move || {
        if let Err(e) = audio_capture::start_capture(audio_tx, stop_audio_rx, warning_tx) {
            error!("Audio capture error: {}", e);
            let _ = app_audio.emit(
                "permission-error",
                serde_json::json!({ "kind": "screen_capture", "message": e.clone() }),
            );
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
            let _ = app_mic.emit(
                "permission-error",
                serde_json::json!({ "kind": "microphone", "message": e.clone() }),
            );
            let _ = app_mic.emit("server-error", e);
        }
    });

    // Spawn TTS playback on a blocking thread
    let (device_uid, device_name) =
        resolve_output_device_selection(output_device_uid, output_device_name);
    let app_tts = app.clone();
    std::thread::spawn(move || {
        if let Err(e) = audio_output::start_tts_playback(
            device_uid,
            device_name,
            tts_rx,
            stop_tts_rx,
            tts_end_rx,
            text_tx,
        ) {
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

/// Pure TCP/HTTP probe for the server's `GET /health`, hand-rolled against
/// `std::net` (one request, one status line to parse) instead of pulling in
/// an HTTP client crate for it -- ticket instruction was "신규 크레이트 추가
/// 금지". Split from the `#[tauri::command]` wrapper below so a test can
/// point it at a mock `TcpListener` instead of a real server.
fn check_health_at(host: &str, port: u16) -> bool {
    let Ok(mut addrs) = format!("{host}:{port}").to_socket_addrs() else {
        return false;
    };
    let Some(addr) = addrs.next() else {
        return false;
    };
    let Ok(mut stream) = TcpStream::connect_timeout(&addr, Duration::from_millis(500)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_millis(500)));
    let request = "GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n";
    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }
    let mut buf = [0u8; 512];
    let Ok(n) = stream.read(&mut buf) else {
        return false;
    };
    let response = String::from_utf8_lossy(&buf[..n]);
    response.starts_with("HTTP/1.1 200") || response.starts_with("HTTP/1.0 200")
}

/// Whether the server on `127.0.0.1:port` is up and answering `/health` --
/// backs the client's "서버 켜기" flow (disabled button vs. spawn-and-poll).
/// `async fn` (Critic round 1, minor): Tauri offloads async commands onto
/// its runtime's blocking-safe pool rather than running the blocking
/// `TcpStream` I/O inline on the caller's dispatch thread.
#[tauri::command]
pub async fn check_server_health(port: u16) -> bool {
    check_health_at("127.0.0.1", port)
}

/// Resolves the server's virtualenv Python interpreter path and confirms it
/// exists, without spawning anything -- separated out of `start_server` so
/// this validation is unit-testable on its own (ticket: "스폰 자체는 실행하지
/// 않는 순수 검증 함수로 분리해 테스트").
fn resolve_server_python(server_dir: &str) -> Result<PathBuf, String> {
    let python_path = PathBuf::from(server_dir)
        .join(".venv")
        .join("bin")
        .join("python");
    if !python_path.is_file() {
        return Err(format!(
            "Python virtualenv not found at {}. Create it with `python3 -m venv .venv` \
             and install the server's dependencies inside {server_dir}.",
            python_path.display()
        ));
    }
    Ok(python_path)
}

/// How long a spawned server is given the benefit of the doubt before a
/// second `start_server` call is allowed to spawn again. The server loads
/// several GB of models inside FastAPI's lifespan before uvicorn binds its
/// socket at all -- `/health` is unreachable (not just unhealthy) for the
/// entire loading window, so a client poll timing out is not evidence the
/// first spawn failed. Critic round 1 [major]: without this guard, a
/// reclick of "서버 켜기" after the poll timeout spawns a *second*
/// `python -m src.main` while the first is still loading, and the loser of
/// the pair fails on bind with EADDRINUSE.
const SPAWN_GUARD_WINDOW: Duration = Duration::from_secs(180);

/// Last time `start_server` actually spawned a process, `None` before the
/// first call. Read/written only through `should_spawn`'s call sites in
/// `start_server` below.
static LAST_SPAWN: StdMutex<Option<Instant>> = StdMutex::new(None);

/// Pure decision behind the spawn guard: `now` and `last_spawn` are plain
/// parameters (not read from the static / `Instant::now()` directly) so
/// this is testable without depending on wall-clock timing ("시각 주입
/// 가능하게 분리").
fn should_spawn(last_spawn: Option<Instant>, now: Instant) -> bool {
    match last_spawn {
        None => true,
        Some(last) => now.duration_since(last) >= SPAWN_GUARD_WINDOW,
    }
}

/// Spawns `<server_dir>/.venv/bin/python -m src.main` (cwd=server_dir) for
/// the "서버 켜기" button, guarded by `should_spawn` against double-spawning
/// while a previous attempt is still (plausibly) loading. Returns
/// `"spawned"` when it actually launched a process, `"already_starting"`
/// when the guard held and nothing new was spawned -- the frontend treats
/// both as success and resumes polling either way.
///
/// The returned child handle is intentionally dropped rather than tracked
/// in app state: the frontend observes whether the spawn "took" by polling
/// `check_server_health` separately, and tauri-plugin-shell's
/// `CommandChild` does not kill its process on drop.
#[tauri::command]
pub async fn start_server(app: AppHandle, server_dir: String) -> Result<String, String> {
    let python_path = resolve_server_python(&server_dir)?;

    let now = Instant::now();
    let mut last_spawn = LAST_SPAWN.lock().map_err(|e| e.to_string())?;
    if !should_spawn(*last_spawn, now) {
        return Ok("already_starting".to_string());
    }

    use tauri_plugin_shell::ShellExt;
    app.shell()
        .command(python_path)
        .args(["-m", "src.main"])
        .current_dir(PathBuf::from(&server_dir))
        .spawn()
        .map_err(|e| format!("Failed to start server: {e}"))?;
    *last_spawn = Some(now);
    Ok("spawned".to_string())
}

/// Best-effort discovery of the server directory for a dev checkout: the
/// client and server live as siblings (`<repo>/client/src-tauri`,
/// `<repo>/server`), so two levels up from this crate's compile-time
/// location is the repo root. Returns `None` (never a guess) when that
/// layout doesn't hold -- packaged builds, or a differently-laid-out
/// checkout -- so the frontend falls back to asking the user instead of
/// spawning something that doesn't exist.
#[tauri::command]
pub fn default_server_dir() -> Option<String> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let candidate = PathBuf::from(manifest_dir).join("../../server");
    let canonical = candidate.canonicalize().ok()?;
    if canonical.is_dir() {
        Some(canonical.to_string_lossy().to_string())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    #[test]
    fn resolve_output_device_selection_threads_both_uid_and_name() {
        let (uid, name) = resolve_output_device_selection(
            Some("uid-123".to_string()),
            Some("Human Name".to_string()),
        );
        assert_eq!(uid, "uid-123");
        assert_eq!(name, "Human Name");
    }

    #[test]
    fn resolve_output_device_selection_defaults_missing_uid_and_name_to_empty() {
        let (uid, name) = resolve_output_device_selection(None, None);
        assert_eq!(uid, "");
        assert_eq!(name, "");
    }

    // C6: check_health_at (the pure function behind check_server_health)
    // against mock TCP sockets only -- no real server dependency.

    #[test]
    fn check_health_at_false_when_nothing_is_listening() {
        // Bind then immediately drop, so the port was briefly valid but
        // nothing answers a real connection attempt.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);

        assert!(!check_health_at("127.0.0.1", port));
    }

    /// Spins up a one-shot mock HTTP server on an OS-assigned port that
    /// replies with `response` to the first connection, then returns the
    /// port so a test can point `check_health_at` at it.
    fn spawn_mock_http_server(response: &'static str) -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buf = [0u8; 512];
                let _ = stream.read(&mut buf);
                let _ = stream.write_all(response.as_bytes());
            }
        });
        port
    }

    #[test]
    fn check_health_at_true_for_200_response() {
        let port =
            spawn_mock_http_server("HTTP/1.1 200 OK\r\nContent-Length: 15\r\n\r\n{\"status\":\"ok\"}");

        assert!(check_health_at("127.0.0.1", port));
    }

    #[test]
    fn check_health_at_false_for_404_response() {
        let port = spawn_mock_http_server("HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n");

        assert!(!check_health_at("127.0.0.1", port));
    }

    // C7: resolve_server_python -- the pure validation `start_server` runs
    // before ever spawning a process.

    #[test]
    fn resolve_server_python_errs_with_the_missing_path_when_venv_is_absent() {
        let dir = tempfile::tempdir().unwrap();

        let err = resolve_server_python(dir.path().to_str().unwrap()).unwrap_err();

        let expected_path = dir.path().join(".venv").join("bin").join("python");
        assert!(
            err.contains(&expected_path.to_string_lossy().to_string()),
            "expected error to mention the missing path {expected_path:?}, got: {err}"
        );
    }

    #[test]
    fn resolve_server_python_ok_when_venv_python_exists() {
        let dir = tempfile::tempdir().unwrap();
        let venv_bin = dir.path().join(".venv").join("bin");
        std::fs::create_dir_all(&venv_bin).unwrap();
        let python_path = venv_bin.join("python");
        std::fs::write(&python_path, "").unwrap();

        let resolved = resolve_server_python(dir.path().to_str().unwrap()).unwrap();

        assert_eq!(resolved, python_path);
    }

    // default_server_dir: pure filesystem detection of the dev checkout's
    // sibling `server/` directory from this crate's compile-time location.
    // Guarded (Critic round 1, "참고" item): an isolated checkout copy of
    // this crate without the sibling `server/` directory must not fail this
    // test on ambient repo layout the code under test has no control over
    // -- skip rather than assert when the directory this test itself
    // expects to find isn't actually there.
    #[test]
    fn default_server_dir_detects_the_sibling_server_directory_when_present() {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let expected = PathBuf::from(manifest_dir).join("../../server");
        if !expected.is_dir() {
            eprintln!(
                "skipping default_server_dir_detects_the_sibling_server_directory_when_present: \
                 no sibling server/ directory in this checkout ({expected:?})"
            );
            return;
        }

        let dir = default_server_dir()
            .expect("expected to detect ../../server since it exists in this checkout");

        let path = PathBuf::from(dir);
        assert!(path.is_dir(), "{path:?} should be a directory");
        assert!(path.ends_with("server"), "{path:?} should end with server");
    }

    // Critic round 1, [major]: a second start_server call within the guard
    // window must not spawn a second server process while the first is
    // still (plausibly) loading models -- should_spawn is the pure
    // decision behind that guard, with both times passed in as parameters
    // so tests don't depend on wall-clock timing ("시각 주입 가능하게 분리").

    #[test]
    fn should_spawn_true_when_nothing_has_been_spawned_yet() {
        assert!(should_spawn(None, Instant::now()));
    }

    #[test]
    fn should_spawn_false_within_the_guard_window_of_a_previous_spawn() {
        let now = Instant::now();
        let ten_seconds_ago = now - Duration::from_secs(10);

        assert!(!should_spawn(Some(ten_seconds_ago), now));
    }

    #[test]
    fn should_spawn_true_once_the_guard_window_has_fully_elapsed() {
        let now = Instant::now();
        let long_ago = now - (SPAWN_GUARD_WINDOW + Duration::from_secs(1));

        assert!(should_spawn(Some(long_ago), now));
    }
}
