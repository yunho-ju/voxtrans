import React, { useState, useCallback, useEffect, useMemo, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-shell";
import TranscriptionOverlay from "./components/TranscriptionOverlay";
import StatusBar from "./components/StatusBar";
import Controls, { portFromServerUrl } from "./components/Controls";
import TrafficLights from "./components/TrafficLights";
import SettingsDrawer from "./components/SettingsDrawer";
import { useTranscription } from "./hooks/useTranscription";
import {
  DEFAULT_SUBTITLE_PACE,
  DEFAULT_TRANSLATION_PROMPT,
  type SubtitlePace,
} from "./constants";

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

const DEFAULT_URL = "ws://localhost:18765/ws";
const LEGACY_DEFAULT_URLS = new Set([
  "ws://localhost:8765/ws",
  "ws://127.0.0.1:8765/ws",
]);

/** What `GET /voices` answers, as the Rust command hands it over. */
type VoicesInfo = {
  backend: string;
  active: string;
  default: string;
  backends: { value: string; label: string }[];
  voices: { value: string; label: string }[];
};

const App: React.FC = () => {
  const { entries, connectionStatus, isTtsSpeaking, clearEntries } = useTranscription();
  const [serverUrl, setServerUrl] = useState(() => {
    const saved = localStorage.getItem("voxtrans_server_url");
    if (saved && LEGACY_DEFAULT_URLS.has(saved)) {
      localStorage.setItem("voxtrans_server_url", DEFAULT_URL);
      return DEFAULT_URL;
    }
    return saved || DEFAULT_URL;
  });
  const [sourceLanguage, setSourceLanguage] = useState("auto");
  const [targetLanguage, setTargetLanguage] = useState("Korean");
  // Sessions always run mode "both" (2026-08-06 user decision): listening
  // never stops; speaking and typing are actions inside the same session,
  // not modes to switch into. The server has supported "both" all along
  // (ws_handler routes system audio and mic audio independently).
  const mode = "both";
  const [ttsVoice, setTtsVoice] = useState("Ryan");
  // Which voices exist is the *server's* answer, not ours: the TTS backend is
  // chosen server-side (VOXTRANS_TTS_BACKEND), and each backend has its own
  // inventory. Empty until the server answers, which is when SettingsDrawer
  // falls back to its built-in list.
  const [voiceOptions, setVoiceOptions] = useState<
    { value: string; label: string }[]
  >([]);
  // The loaded backend belongs to the server. A picker change requests a
  // switch; the configured status confirms which backend actually won.
  const [ttsBackend, setTtsBackend] = useState("qwen3");
  const [backendOptions, setBackendOptions] = useState<
    { value: string; label: string }[]
  >([]);
  const pendingBackend = useRef<string | null>(null);
  // Picker requests can finish out of order because each previews /voices.
  // Only the newest choice may update the voice or send a backend switch.
  const backendChoiceGeneration = useRef(0);
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
  // Pre-filled with the default so the field can be edited rather than
  // retyped; `??` not `||` so a deliberately emptied field stays empty on the
  // next launch instead of springing back. An empty prompt tells the server to
  // use its own default.
  const [translationPrompt, setTranslationPrompt] = useState(
    () =>
      localStorage.getItem("voxtrans_translation_prompt") ??
      DEFAULT_TRANSLATION_PROMPT
  );
  const [subtitlePace, setSubtitlePace] = useState<SubtitlePace>(() => {
    const saved = localStorage.getItem("voxtrans_subtitle_pace");
    if (!saved) return DEFAULT_SUBTITLE_PACE;
    try {
      return { ...DEFAULT_SUBTITLE_PACE, ...JSON.parse(saved) };
    } catch {
      return DEFAULT_SUBTITLE_PACE;
    }
  });
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
  const subtitlePaceRef = useRef(subtitlePace);
  subtitlePaceRef.current = subtitlePace;

  // Load output devices on mount
  useEffect(() => {
    invoke<string[][]>("list_output_devices")
      .then((devices) => {
        const parsed = devices.map(([name, uid]) => ({ name, uid }));
        setOutputDevices(parsed);
      })
      .catch(() => {});
  }, []);

  // Load the server's voice list. Re-fetched whenever the connection comes
  // up, because on a cold start the app is running well before the server is
  // -- and a server started later may be a different backend than the one
  // that answered last time.
  useEffect(() => {
    const port = portFromServerUrl(serverUrl);
    let cancelled = false;
    invoke<VoicesInfo | null>("list_voices", { port })
      .then((info) => {
        if (cancelled || !info) return;
        if (info.voices.length > 0) setVoiceOptions(info.voices);
        if (info.backends.length > 0) setBackendOptions(info.backends);
        if (!pendingBackend.current) setTtsBackend(info.active);
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [serverUrl, connectionStatus]);

  // Configure acknowledgements report the actual globally loaded backend.
  // A switch can fail or another session can change it; either way the picker
  // must follow the server's answer rather than retain an optimistic choice.
  useEffect(() => {
    const unlisten = listen<string>("server-status", (event) => {
      try {
        const status = JSON.parse(event.payload) as {
          status?: string;
          tts_backend?: string;
        };
        if (status.status !== "configured" || !status.tts_backend) return;
        pendingBackend.current = null;
        setTtsBackend(status.tts_backend);
        const port = portFromServerUrl(serverUrl);
        invoke<VoicesInfo | null>("list_voices", { port, backend: status.tts_backend })
          .then((info) => {
            if (!info || info.voices.length === 0) return;
            setVoiceOptions(info.voices);
            if (!info.voices.some((v) => v.value === ttsVoiceRef.current)) {
              setTtsVoice(info.voices.find((v) => v.value === info.default)?.value ?? info.voices[0].value);
            }
          })
          .catch(() => {});
      } catch {
        // Ignore statuses from older servers that do not carry JSON here.
      }
    });
    return () => { unlisten.then((fn) => fn()); };
  }, [serverUrl]);

  /** Switching backend reloads the server's TTS worker, and the voice that
   * was selected may not exist on the other side — a Kokoro voicepack means
   * nothing to Qwen3-TTS, which would fall back to an unconditioned voice
   * that sounds wrong without saying why. So the voice resets with it. */
  const handleTtsBackendChange = useCallback(
    async (backend: string) => {
      const generation = ++backendChoiceGeneration.current;
      pendingBackend.current = backend;
      setTtsBackend(backend);
      const port = portFromServerUrl(serverUrl);
      const info = await invoke<VoicesInfo | null>("list_voices", {
        port,
        backend,
      }).catch(() => null);
      if (generation !== backendChoiceGeneration.current) return;
      const overrides: Record<string, string> = { ttsBackend: backend };
      if (info && info.voices.length > 0) {
        setVoiceOptions(info.voices);
        if (!info.voices.some((v) => v.value === ttsVoiceRef.current)) {
          const fallback =
            info.voices.find((v) => v.value === info.default)?.value ??
            info.voices[0].value;
          setTtsVoice(fallback);
          overrides.ttsVoice = fallback;
        }
      }
      const sent = await sendConfigure(overrides);
      if (!sent && generation === backendChoiceGeneration.current) {
        pendingBackend.current = null;
        if (info) setTtsBackend(info.active);
      }
    },
    [serverUrl]
  );

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
        subtitlePace: [
          subtitlePaceRef.current.speechS,
          subtitlePaceRef.current.silenceMs,
          subtitlePaceRef.current.maxS,
        ],
      }).catch(() => {});
    }
  }, [connectionStatus]);

  const sendConfigure = useCallback(
    async (overrides: Record<string, string | boolean | number[]> = {}) => {
      try {
        await invoke("send_configure", {
          language: overrides?.language ?? sourceLanguageRef.current,
          targetLanguage: overrides.targetLanguage ?? targetLanguageRef.current,
          mode: overrides.mode ?? modeRef.current,
          ttsVoice: overrides.ttsVoice ?? ttsVoiceRef.current,
          ...(overrides.ttsBackend ? { ttsBackend: overrides.ttsBackend } : {}),
          speakTargetLanguage:
            overrides.speakTargetLanguage ?? speakTargetLanguageRef.current,
          voiceClone: overrides.voiceClone ?? voiceCloneRef.current,
          micMode: overrides.micMode ?? micModeRef.current,
          translationPrompt:
            overrides.translationPrompt ?? translationPromptRef.current,
          subtitlePace:
            overrides.subtitlePace ?? [
              subtitlePaceRef.current.speechS,
              subtitlePaceRef.current.silenceMs,
              subtitlePaceRef.current.maxS,
            ],
        });
        return true;
      } catch {
        // Not connected yet
        return false;
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

  const handleSubtitlePaceChange = useCallback(
    (pace: SubtitlePace) => {
      setSubtitlePace(pace);
      localStorage.setItem("voxtrans_subtitle_pace", JSON.stringify(pace));
      // Applies at the next segment boundary; no stop/start needed.
      sendConfigure({
        subtitlePace: [pace.speechS, pace.silenceMs, pace.maxS],
      });
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

  // Keep the panel tint on the root so subtitle updates cannot recompose a
  // full-window backdrop blur. The slider is the only source of tint changes.
  const rootStyle = useMemo(
    () => ({
      ...styles.root,
      backgroundColor: `rgba(10, 14, 24, ${opacity})`,
      borderColor: `rgba(255, 255, 255, ${0.1 + opacity * 0.1})`,
    }),
    [opacity]
  );
  // The header rides the same slider as the panel instead of a fixed black:
  // at low opacity it used to stay a solid bar while everything below it went
  // clear.
  const headerStyle = useMemo(
    () => ({
      ...styles.header,
      background: `rgba(0, 0, 0, ${opacity * 0.18})`,
      borderBottomColor: `rgba(255, 255, 255, ${0.03 + opacity * 0.05})`,
    }),
    [opacity]
  );

  return (
    <div style={rootStyle}>
      <div style={headerStyle} data-tauri-drag-region>
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
            voiceOptions={voiceOptions}
            ttsBackend={ttsBackend}
            backendOptions={backendOptions}
            onTtsBackendChange={handleTtsBackendChange}
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
            subtitlePace={subtitlePace}
            onSubtitlePaceChange={handleSubtitlePaceChange}
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
