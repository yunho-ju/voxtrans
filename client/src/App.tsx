import React, { useState, useCallback, useEffect, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-shell";
import TranscriptionOverlay from "./components/TranscriptionOverlay";
import StatusBar from "./components/StatusBar";
import Controls from "./components/Controls";
import TrafficLights from "./components/TrafficLights";
import SettingsDrawer from "./components/SettingsDrawer";
import { useTranscription } from "./hooks/useTranscription";

interface OutputDevice {
  name: string;
  uid: string;
}

// "system_audio" is the Core Audio process tap's permission. macOS lists it
// under Screen Recording as "System Audio Recording Only"; this app never
// records the screen and does not need full Screen Recording.
type PermissionKind = "system_audio" | "microphone";

interface PermissionErrorPayload {
  kind: PermissionKind;
  message: string;
}

// macOS System Settings deep links for each permission kind `commands.rs`'s
// two capture-start catch sites can report via the `permission-error` event.
const PERMISSION_SETTINGS_URLS: Record<PermissionKind, string> = {
  system_audio:
    "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture",
  microphone:
    "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone",
};

const PERMISSION_LABELS: Record<PermissionKind, string> = {
  system_audio: "시스템 오디오 녹음",
  microphone: "마이크",
};

// Module-scoped (not component state): System Settings should deep-link
// open automatically only once per app launch, regardless of how many times
// App re-renders or how many permission-error events arrive afterward.
let hasAutoOpenedPermissionSettings = false;

interface InputDevice {
  name: string;
  uid: string;
}

const DEFAULT_URL = "ws://localhost:8765/ws";

const App: React.FC = () => {
  const { entries, connectionStatus, isTtsSpeaking, clearEntries } = useTranscription();
  const [serverUrl, setServerUrl] = useState(
    () => localStorage.getItem("voxtrans_server_url") || DEFAULT_URL
  );
  const [sourceLanguage, setSourceLanguage] = useState("auto");
  const [targetLanguage, setTargetLanguage] = useState("Korean");
  // Sessions always run mode "both" (2026-08-06 user decision): listening
  // never stops; speaking and typing are actions inside the same session,
  // not modes to switch into. The server has supported "both" all along
  // (ws_handler routes system audio and mic audio independently).
  const mode = "both";
  const [ttsVoice, setTtsVoice] = useState("Ryan");
  // Typing-to-speak is a per-user preference, not a mode — some users only
  // ever speak, so the row can be hidden entirely from Settings.
  const [showTypeInput, setShowTypeInput] = useState(
    () => localStorage.getItem("voxtrans_show_type_input") !== "false"
  );
  // Persisted like the input device: the meeting-routing setup (Output →
  // BlackHole) should survive restarts instead of being re-picked each run.
  const [outputDevice, setOutputDevice] = useState(
    () => localStorage.getItem("voxtrans_output_device") || ""
  );
  const [outputDevices, setOutputDevices] = useState<OutputDevice[]>([]);
  const [inputDevice, setInputDevice] = useState(
    () => localStorage.getItem("voxtrans_input_device") || ""
  );
  const [inputDevices, setInputDevices] = useState<InputDevice[]>([]);
  // Server-launch priority ① localStorage override ② Rust-detected dev
  // default (../../server from src-tauri, populated below) ③ null, which
  // Controls.tsx renders as a hint to fill in Settings rather than a
  // "서버 켜기" button pointed at nothing.
  const [serverDirOverride, setServerDirOverride] = useState(
    () => localStorage.getItem("voxtrans_server_dir") || ""
  );
  const [detectedServerDir, setDetectedServerDir] = useState<string | null>(null);
  const serverDir = serverDirOverride || detectedServerDir;
  const [speakTargetLanguage, setSpeakTargetLanguage] = useState("English");
  const [voiceClone, setVoiceClone] = useState(true);
  const [micMode, setMicMode] = useState("auto");
  // Empty = server default template. Persisted so a tuned prompt survives a
  // restart, which is the whole point of making it editable.
  const [translationPrompt, setTranslationPrompt] = useState(
    () => localStorage.getItem("voxtrans_translation_prompt") || ""
  );
  const [micActive, setMicActive] = useState(false);
  const [opacity, setOpacity] = useState(0.4);
  const [showSettings, setShowSettings] = useState(false);
  const [textInput, setTextInput] = useState("");
  const [lastError, setLastError] = useState<string | null>(null);
  const [permissionErrorKind, setPermissionErrorKind] = useState<PermissionKind | null>(
    null
  );
  // Startup permission check (null until it has run once). Distinct from
  // `permissionErrorKind`, which is the reactive path: this one asks *before*
  // the user presses start, so a missing grant is a dialog rather than a
  // session that ends the instant it begins.
  const [missingPermissions, setMissingPermissions] = useState<
    PermissionKind[] | null
  >(null);

  const sourceLanguageRef = useRef(sourceLanguage);
  sourceLanguageRef.current = sourceLanguage;
  const targetLanguageRef = useRef(targetLanguage);
  targetLanguageRef.current = targetLanguage;
  const modeRef = useRef(mode);
  modeRef.current = mode;
  const ttsVoiceRef = useRef(ttsVoice);
  ttsVoiceRef.current = ttsVoice;
  const speakTargetLanguageRef = useRef(speakTargetLanguage);
  speakTargetLanguageRef.current = speakTargetLanguage;
  const voiceCloneRef = useRef(voiceClone);
  voiceCloneRef.current = voiceClone;
  const micModeRef = useRef(micMode);
  micModeRef.current = micMode;
  const translationPromptRef = useRef(translationPrompt);
  translationPromptRef.current = translationPrompt;

  // Load output devices on mount
  useEffect(() => {
    invoke<string[][]>("list_output_devices")
      .then((devices) => {
        const parsed = devices.map(([name, uid]) => ({ name, uid }));
        setOutputDevices(parsed);
      })
      .catch(() => {});
  }, []);

  // Load input (microphone) devices on mount (T008/T010: real UID
  // enumeration, no longer a hardcoded single entry).
  useEffect(() => {
    invoke<string[][]>("list_devices")
      .then((devices) => {
        const parsed = devices.map(([name, uid]) => ({ name, uid }));
        setInputDevices(parsed);
      })
      .catch(() => {});
  }, []);

  // Detect the sibling server/ directory in a dev checkout (only used when
  // the user hasn't overridden voxtrans_server_dir in Settings).
  useEffect(() => {
    invoke<string | null>("default_server_dir")
      .then(setDetectedServerDir)
      .catch(() => setDetectedServerDir(null));
  }, []);

  // FR-009 / spec.md US3 acceptance scenario 2: a selected device that
  // fails to open (missing, removed, or a fabricated UID) must notify the
  // user rather than silently doing nothing. `commands.rs` already emits
  // `server-error` naming the failed UID when `mic_capture::start_mic_capture`
  // returns `Err` -- this is the first (and, before this, only missing)
  // listener for that event anywhere in `client/src/`.
  useEffect(() => {
    const unlistenServerError = listen<string>("server-error", (event) => {
      setLastError(event.payload);
    });
    return () => {
      unlistenServerError.then((fn) => fn());
    };
  }, []);

  // Permission errors (screen recording / microphone) deep-link to the
  // relevant System Settings pane: automatically once per app launch, and
  // on every click of the banner's "설정 열기" button thereafter (design
  // decisions 1-2 of the permission-error deep-link ticket). This does not
  // raise its own banner -- `server-error` (above) already does that from
  // the same Rust catch site; this only remembers which pane to deep-link.
  useEffect(() => {
    const unlistenPermissionError = listen<PermissionErrorPayload>(
      "permission-error",
      (event) => {
        const { kind } = event.payload;
        setPermissionErrorKind(kind);
        if (!hasAutoOpenedPermissionSettings) {
          hasAutoOpenedPermissionSettings = true;
          open(PERMISSION_SETTINGS_URLS[kind]).catch(() => {});
        }
      }
    );
    return () => {
      unlistenPermissionError.then((fn) => fn());
    };
  }, []);

  // Ask for the microphone on launch, before speaking needs it. System audio
  // is deliberately not preflighted: it goes through a Core Audio tap whose
  // permission has no preflight API, and the previous attempt checked Screen
  // Recording instead — warning about a permission this app never uses while
  // capture worked fine. System-audio failures surface through
  // `permission-error` at capture-start time, where they are real.
  useEffect(() => {
    let cancelled = false;
    invoke<{ microphone: boolean }>("request_permissions")
      .then((status) => {
        if (cancelled) return;
        const missing: PermissionKind[] = [];
        if (!status.microphone) missing.push("microphone");
        setMissingPermissions(missing);
      })
      .catch(() => {
        // A failed check must not block the app: the reactive
        // `permission-error` path still catches it at capture-start time.
        if (!cancelled) setMissingPermissions([]);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Send configure automatically when connected
  useEffect(() => {
    if (connectionStatus === "connected") {
      invoke("send_configure", {
        language: sourceLanguageRef.current,
        targetLanguage: targetLanguageRef.current,
        mode: modeRef.current,
        ttsVoice: ttsVoiceRef.current,
        speakTargetLanguage: speakTargetLanguageRef.current,
        voiceClone: voiceCloneRef.current,
        micMode: micModeRef.current,
        translationPrompt: translationPromptRef.current,
      }).catch(() => {});
    }
  }, [connectionStatus]);

  const sendConfigure = useCallback(
    async (overrides: Record<string, string | boolean> = {}) => {
      try {
        await invoke("send_configure", {
          language: overrides?.language ?? sourceLanguageRef.current,
          targetLanguage: overrides.targetLanguage ?? targetLanguageRef.current,
          mode: overrides.mode ?? modeRef.current,
          ttsVoice: overrides.ttsVoice ?? ttsVoiceRef.current,
          speakTargetLanguage:
            overrides.speakTargetLanguage ?? speakTargetLanguageRef.current,
          voiceClone: overrides.voiceClone ?? voiceCloneRef.current,
          micMode: overrides.micMode ?? micModeRef.current,
          translationPrompt:
            overrides.translationPrompt ?? translationPromptRef.current,
        });
      } catch {
        // Not connected yet
      }
    },
    []
  );

  // Push-to-talk: spacebar handler
  useEffect(() => {
    if (micModeRef.current !== "push_to_talk") return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code === "Space" && !e.repeat) {
        // Don't intercept space when text input is focused
        if (document.activeElement?.tagName === "INPUT") return;
        e.preventDefault();
        setMicActive(true);
        invoke("send_mic_active", { active: true }).catch(() => {});
      }
    };
    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.code === "Space") {
        if (document.activeElement?.tagName === "INPUT") return;
        e.preventDefault();
        setMicActive(false);
        invoke("send_mic_active", { active: false }).catch(() => {});
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, [micMode]);

  const handleServerUrlChange = useCallback((url: string) => {
    setServerUrl(url);
    localStorage.setItem("voxtrans_server_url", url);
  }, []);

  const handleTranslationPromptChange = useCallback(
    (prompt: string) => {
      setTranslationPrompt(prompt);
      localStorage.setItem("voxtrans_translation_prompt", prompt);
      // Applies to the next segment; no stop/start needed.
      sendConfigure({ translationPrompt: prompt });
    },
    [sendConfigure]
  );

  const handleInputDeviceChange = useCallback((uid: string) => {
    setInputDevice(uid);
    localStorage.setItem("voxtrans_input_device", uid);
  }, []);

  const handleOutputDeviceChange = useCallback(
    (uid: string) => {
      setOutputDevice(uid);
      localStorage.setItem("voxtrans_output_device", uid);
      // Retarget the running playback thread too. Without this the change only
      // took effect on the next stop/start, while the UI showed the new device
      // as if it were already live.
      invoke("set_output_device", {
        outputDeviceUid: uid,
        outputDeviceName: outputDevices.find((d) => d.uid === uid)?.name ?? "",
      }).catch(() => {});
    },
    [outputDevices]
  );

  /** One-click meeting routing (2026-08-06 user request): TTS output →
   * BlackHole 2ch, capture mic → a real (non-virtual) input. The remaining
   * leg — the meeting app's own mic picker → BlackHole — lives inside
   * Zoom/Meet and cannot be set from here. Returns a human-readable result
   * so Settings can show what happened; never silently half-applies. */
  const applyMeetingPreset = useCallback((): string => {
    const blackhole = outputDevices.find((d) => d.name.includes("BlackHole"));
    if (!blackhole) {
      return "BlackHole 2ch가 없습니다 — 설치 후 다시 시도하세요 (existential.audio/blackhole)";
    }
    const realMic = inputDevices.find((d) => !d.name.includes("BlackHole"));
    handleOutputDeviceChange(blackhole.uid);
    if (realMic) {
      handleInputDeviceChange(realMic.uid);
    }
    return realMic
      ? `Output → ${blackhole.name}, Microphone → ${realMic.name} 설정됨. Zoom/Meet의 마이크는 BlackHole 2ch로 직접 선택하세요.`
      : `Output → ${blackhole.name} 설정됨. 실제 마이크를 찾지 못해 Microphone은 그대로입니다.`;
  }, [outputDevices, inputDevices, handleOutputDeviceChange, handleInputDeviceChange]);

  const handleServerDirChange = useCallback((dir: string) => {
    setServerDirOverride(dir);
    localStorage.setItem("voxtrans_server_dir", dir);
  }, []);

  const handleSourceLanguageChange = useCallback(
    async (lang: string) => {
      setSourceLanguage(lang);
      await sendConfigure({ language: lang });
    },
    [sendConfigure]
  );

  const handleTargetLanguageChange = useCallback(
    async (lang: string) => {
      setTargetLanguage(lang);
      await sendConfigure({ targetLanguage: lang });
    },
    [sendConfigure]
  );

  const handleShowTypeInputChange = useCallback((show: boolean) => {
    setShowTypeInput(show);
    localStorage.setItem("voxtrans_show_type_input", String(show));
  }, []);

  const handleTtsVoiceChange = useCallback(
    async (v: string) => {
      setTtsVoice(v);
      await sendConfigure({ ttsVoice: v });
    },
    [sendConfigure]
  );

  const handleSpeakTargetLanguageChange = useCallback(
    async (lang: string) => {
      setSpeakTargetLanguage(lang);
      await sendConfigure({ speakTargetLanguage: lang });
    },
    [sendConfigure]
  );

  const handleVoiceCloneChange = useCallback(
    async (clone: boolean) => {
      setVoiceClone(clone);
      await sendConfigure({ voiceClone: clone });
    },
    [sendConfigure]
  );

  const handleMicModeChange = useCallback(
    async (m: string) => {
      setMicMode(m);
      setMicActive(false);
      await sendConfigure({ micMode: m });
    },
    [sendConfigure]
  );

  const handleOpenPermissionSettings = useCallback(() => {
    if (permissionErrorKind) {
      open(PERMISSION_SETTINGS_URLS[permissionErrorKind]).catch(() => {});
    }
  }, [permissionErrorKind]);

  const handleSendText = useCallback(async () => {
    if (!textInput.trim() || isTtsSpeaking) return;
    try {
      await invoke("send_speak_text", { text: textInput });
      setTextInput("");
    } catch {
      // Not connected
    }
  }, [textInput, isTtsSpeaking]);

  // Glass background
  const bgAlpha = opacity * 0.6;
  const whiteAlpha = 0.04 + opacity * 0.04;

  return (
    <div
      style={{
        ...styles.root,
        borderColor: `rgba(255, 255, 255, ${0.1 + opacity * 0.1})`,
      }}
    >
      <div
        className="glass-backdrop"
        style={{
          background: `linear-gradient(
            135deg,
            rgba(255, 255, 255, ${whiteAlpha}),
            rgba(0, 0, 0, ${bgAlpha})
          )`,
          borderRadius: "12px",
        }}
      />
      <div style={styles.header} data-tauri-drag-region>
        <div style={styles.headerLeft}>
          <TrafficLights />
          <StatusBar status={connectionStatus} />
        </div>
        <div style={styles.headerRight} data-tauri-drag-region>
          <span style={styles.title}>VoxTrans</span>
          <button
            onClick={() => setShowSettings((s) => !s)}
            style={{
              ...styles.gearBtn,
              ...(showSettings ? styles.gearBtnActive : {}),
            }}
            title="Settings"
          >
            ⚙
          </button>
          <div style={styles.opacityControl}>
            <span style={styles.opacityLabel}>
              {Math.round(opacity * 100)}%
            </span>
            <input
              type="range"
              min="0.1"
              max="1"
              step="0.05"
              value={opacity}
              onChange={(e) => setOpacity(parseFloat(e.target.value))}
              style={styles.opacitySlider}
            />
          </div>
        </div>
      </div>
      {missingPermissions !== null && missingPermissions.length > 0 && (
        <div style={styles.errorBanner} role="alert">
          <span style={styles.errorBannerText}>
            {PERMISSION_LABELS[missingPermissions[0]]} 권한이 없어
            {missingPermissions[0] === "system_audio"
              ? " 회의 소리를 못 듣습니다."
              : " 말하기를 쓸 수 없습니다."}
            {" 허용 후 앱을 다시 시작하세요."}
          </span>
          <button
            onClick={() => {
              invoke("open_permission_settings", {
                kind:
                  missingPermissions[0] === "system_audio"
                    ? "system_audio"
                    : "microphone",
              }).catch(() => {});
            }}
            style={styles.errorBannerAction}
          >
            설정 열기
          </button>
          <button
            onClick={() => setMissingPermissions([])}
            style={styles.errorBannerClose}
            aria-label="Dismiss permission warning"
          >
            ✕
          </button>
        </div>
      )}
      {lastError && (
        <div style={styles.errorBanner} role="alert">
          <span style={styles.errorBannerText}>{lastError}</span>
          {permissionErrorKind && (
            <button
              onClick={handleOpenPermissionSettings}
              style={styles.errorBannerAction}
            >
              설정 열기
            </button>
          )}
          <button
            onClick={() => {
              setLastError(null);
              setPermissionErrorKind(null);
            }}
            style={styles.errorBannerClose}
            aria-label="Dismiss error"
          >
            ✕
          </button>
        </div>
      )}
      <div style={styles.contentArea}>
        <TranscriptionOverlay entries={entries} />
        {showSettings && (
          <SettingsDrawer
            onClose={() => setShowSettings(false)}
            serverUrl={serverUrl}
            onServerUrlChange={handleServerUrlChange}
            sourceLanguage={sourceLanguage}
            onSourceLanguageChange={handleSourceLanguageChange}
            targetLanguage={targetLanguage}
            onTargetLanguageChange={handleTargetLanguageChange}
            showTypeInput={showTypeInput}
            onShowTypeInputChange={handleShowTypeInputChange}
            ttsVoice={ttsVoice}
            onTtsVoiceChange={handleTtsVoiceChange}
            outputDevice={outputDevice}
            onOutputDeviceChange={handleOutputDeviceChange}
            onApplyMeetingPreset={applyMeetingPreset}
            outputDevices={outputDevices}
            inputDevice={inputDevice}
            onInputDeviceChange={handleInputDeviceChange}
            inputDevices={inputDevices}
            serverDir={serverDirOverride}
            onServerDirChange={handleServerDirChange}
            detectedServerDir={detectedServerDir}
            speakTargetLanguage={speakTargetLanguage}
            onSpeakTargetLanguageChange={handleSpeakTargetLanguageChange}
            voiceClone={voiceClone}
            onVoiceCloneChange={handleVoiceCloneChange}
            translationPrompt={translationPrompt}
            onTranslationPromptChange={handleTranslationPromptChange}
            micMode={micMode}
            onMicModeChange={handleMicModeChange}
            micActive={micActive}
          />
        )}
      </div>
      <Controls
        status={connectionStatus}
        onClear={clearEntries}
        serverUrl={serverUrl}
        outputDevice={outputDevice}
        outputDeviceName={
          outputDevices.find((d) => d.uid === outputDevice)?.name ?? ""
        }
        inputDevice={inputDevice}
        serverDir={serverDir}
        showTypeInput={showTypeInput}
        micMode={micMode}
        micActive={micActive}
        textInput={textInput}
        onTextInputChange={setTextInput}
        onSendText={handleSendText}
        isTtsSpeaking={isTtsSpeaking}
      />
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  root: {
    position: "relative" as const,
    width: "100%",
    height: "100%",
    display: "flex",
    flexDirection: "column",
    borderRadius: "12px",
    overflow: "hidden",
    border: "1px solid rgba(255,255,255,0.15)",
    boxShadow: "inset 0 1px 0 rgba(255,255,255,0.1), 0 0 0 0.5px rgba(0,0,0,0.3)",
  },
  header: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "6px 0",
    borderBottom: "1px solid rgba(255,255,255,0.08)",
    background: "rgba(0,0,0,0.3)",
    cursor: "grab",
    flexShrink: 0,
    position: "relative" as const,
    zIndex: 1,
  },
  headerLeft: {
    display: "flex",
    alignItems: "center",
    gap: "2px",
  },
  headerRight: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
    paddingRight: "12px",
  },
  title: {
    color: "rgba(255,255,255,0.5)",
    fontSize: "11px",
    userSelect: "none" as const,
    whiteSpace: "nowrap" as const,
  },
  gearBtn: {
    padding: "2px 5px",
    border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: "4px",
    backgroundColor: "transparent",
    color: "rgba(255,255,255,0.4)",
    fontSize: "13px",
    cursor: "pointer",
    lineHeight: 1,
  },
  gearBtnActive: {
    backgroundColor: "rgba(255,255,255,0.1)",
    borderColor: "rgba(255,255,255,0.2)",
    color: "rgba(255,255,255,0.8)",
  },
  opacityControl: {
    display: "flex",
    alignItems: "center",
    gap: "6px",
  },
  opacityLabel: {
    color: "rgba(255,255,255,0.4)",
    fontSize: "10px",
    minWidth: "28px",
    textAlign: "right" as const,
    userSelect: "none" as const,
  },
  opacitySlider: {
    width: "50px",
    cursor: "pointer",
  },
  contentArea: {
    position: "relative" as const,
    flex: 1,
    display: "flex",
    flexDirection: "column",
    overflow: "hidden",
    zIndex: 1,
  },
  errorBanner: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: "8px",
    padding: "6px 10px",
    backgroundColor: "rgba(248,113,113,0.18)",
    borderBottom: "1px solid rgba(248,113,113,0.35)",
    color: "#fecaca",
    fontSize: "11px",
    position: "relative" as const,
    zIndex: 2,
  },
  errorBannerText: {
    flex: 1,
    overflowWrap: "break-word" as const,
  },
  errorBannerAction: {
    flexShrink: 0,
    padding: "1px 7px",
    border: "1px solid rgba(254,202,202,0.5)",
    borderRadius: "3px",
    backgroundColor: "transparent",
    color: "#fecaca",
    fontSize: "11px",
    cursor: "pointer",
  },
  errorBannerClose: {
    flexShrink: 0,
    padding: "1px 5px",
    border: "none",
    borderRadius: "3px",
    backgroundColor: "transparent",
    color: "#fecaca",
    fontSize: "11px",
    cursor: "pointer",
  },
};

export default App;
