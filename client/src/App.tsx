import React, { useState, useCallback, useEffect, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
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
  const [mode, setMode] = useState("listen");
  const [ttsVoice, setTtsVoice] = useState("Ryan");
  const [outputDevice, setOutputDevice] = useState("");
  const [outputDevices, setOutputDevices] = useState<OutputDevice[]>([]);
  const [inputDevice, setInputDevice] = useState(
    () => localStorage.getItem("voxtrans_input_device") || ""
  );
  const [inputDevices, setInputDevices] = useState<InputDevice[]>([]);
  const [speakTargetLanguage, setSpeakTargetLanguage] = useState("English");
  const [voiceClone, setVoiceClone] = useState(true);
  const [micMode, setMicMode] = useState("auto");
  const [micActive, setMicActive] = useState(false);
  const [opacity, setOpacity] = useState(0.4);
  const [showSettings, setShowSettings] = useState(false);
  const [textInput, setTextInput] = useState("");
  const [lastError, setLastError] = useState<string | null>(null);

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

  const handleInputDeviceChange = useCallback((uid: string) => {
    setInputDevice(uid);
    localStorage.setItem("voxtrans_input_device", uid);
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

  const handleModeChange = useCallback(
    async (m: string) => {
      setMode(m);
      await sendConfigure({ mode: m });
    },
    [sendConfigure]
  );

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
      {lastError && (
        <div style={styles.errorBanner} role="alert">
          <span style={styles.errorBannerText}>{lastError}</span>
          <button
            onClick={() => setLastError(null)}
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
            mode={mode}
            onModeChange={handleModeChange}
            ttsVoice={ttsVoice}
            onTtsVoiceChange={handleTtsVoiceChange}
            outputDevice={outputDevice}
            onOutputDeviceChange={setOutputDevice}
            outputDevices={outputDevices}
            inputDevice={inputDevice}
            onInputDeviceChange={handleInputDeviceChange}
            inputDevices={inputDevices}
            speakTargetLanguage={speakTargetLanguage}
            onSpeakTargetLanguageChange={handleSpeakTargetLanguageChange}
            voiceClone={voiceClone}
            onVoiceCloneChange={handleVoiceCloneChange}
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
        inputDevice={inputDevice}
        mode={mode}
        onModeChange={handleModeChange}
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
