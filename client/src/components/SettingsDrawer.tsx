import React from "react";

interface OutputDevice {
  name: string;
  uid: string;
}

interface InputDevice {
  name: string;
  uid: string;
}

interface Props {
  onClose: () => void;
  serverUrl: string;
  onServerUrlChange: (url: string) => void;
  sourceLanguage: string;
  onSourceLanguageChange: (lang: string) => void;
  targetLanguage: string;
  onTargetLanguageChange: (lang: string) => void;
  /** Whether the typing-to-speak input row is shown in Controls. Sessions
   * always run mode "both" (2026-08-06 user decision) — typing is an action
   * some users never need, so its visibility is a preference, not a mode. */
  showTypeInput: boolean;
  onShowTypeInputChange: (show: boolean) => void;
  ttsVoice: string;
  onTtsVoiceChange: (voice: string) => void;
  outputDevice: string;
  onOutputDeviceChange: (uid: string) => void;
  outputDevices: OutputDevice[];
  inputDevice: string;
  onInputDeviceChange: (uid: string) => void;
  inputDevices: InputDevice[];
  /** localStorage override only (empty string means "not set") -- App.tsx
   * combines this with `detectedServerDir` to get the effective path the
   * "서버 켜기" button uses. */
  serverDir: string;
  onServerDirChange: (dir: string) => void;
  /** Rust `default_server_dir()` result, shown as the field's placeholder
   * so the user can see what would be used if they leave it blank. */
  detectedServerDir: string | null;
  speakTargetLanguage: string;
  onSpeakTargetLanguageChange: (lang: string) => void;
  voiceClone: boolean;
  onVoiceCloneChange: (clone: boolean) => void;
  micMode: string;
  onMicModeChange: (mode: string) => void;
  micActive: boolean;
}

const SOURCE_LANGUAGE_OPTIONS = [
  { value: "auto", label: "Auto" },
  { value: "Korean", label: "Korean" },
  { value: "English", label: "English" },
  { value: "Japanese", label: "Japanese" },
  { value: "Chinese", label: "Chinese" },
  { value: "Spanish", label: "Spanish" },
];

const LANGUAGE_OPTIONS = [
  { value: "", label: "Off" },
  { value: "Korean", label: "Korean" },
  { value: "English", label: "English" },
  { value: "Japanese", label: "Japanese" },
  { value: "Chinese", label: "Chinese" },
  { value: "Spanish", label: "Spanish" },
];

const VOICE_OPTIONS = [
  { value: "Ryan", label: "Ryan (EN)" },
  { value: "Aiden", label: "Aiden (EN)" },
  { value: "Vivian", label: "Vivian (ZH)" },
  { value: "Serena", label: "Serena (ZH)" },
];

const MIC_MODE_OPTIONS = [
  { value: "auto", label: "Auto (VAD)" },
  { value: "push_to_talk", label: "Push-to-Talk" },
];

const SettingsDrawer: React.FC<Props> = ({
  onClose,
  serverUrl,
  onServerUrlChange,
  sourceLanguage,
  onSourceLanguageChange,
  targetLanguage,
  onTargetLanguageChange,
  showTypeInput,
  onShowTypeInputChange,
  ttsVoice,
  onTtsVoiceChange,
  outputDevice,
  onOutputDeviceChange,
  outputDevices,
  inputDevice,
  onInputDeviceChange,
  inputDevices,
  serverDir,
  onServerDirChange,
  detectedServerDir,
  speakTargetLanguage,
  onSpeakTargetLanguageChange,
  voiceClone,
  onVoiceCloneChange,
  micMode,
  onMicModeChange,
  micActive,
}) => {
  return (
    <div style={styles.overlay}>
      <div style={styles.header}>
        <span style={styles.title}>Settings</span>
        <button onClick={onClose} style={styles.closeBtn}>✕</button>
      </div>
      <div style={styles.content}>
        <input
          type="text"
          value={serverUrl}
          onChange={(e) => onServerUrlChange(e.target.value)}
          style={styles.input}
          placeholder="wss://host/ws"
        />
        <input
          type="text"
          value={serverDir}
          onChange={(e) => onServerDirChange(e.target.value)}
          style={styles.input}
          placeholder={
            detectedServerDir
              ? `자동 탐지됨: ${detectedServerDir}`
              : "서버 경로 (자동 탐지 실패, 직접 입력)"
          }
          aria-label="Server directory"
        />
        <div style={styles.settingRow}>
          <label style={styles.label}>Source</label>
          <select
            value={sourceLanguage}
            onChange={(e) => onSourceLanguageChange(e.target.value)}
            style={styles.select}
          >
            {SOURCE_LANGUAGE_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>
        <div style={styles.settingRow}>
          <label style={styles.label}>Translation</label>
          <select
            value={targetLanguage}
            onChange={(e) => onTargetLanguageChange(e.target.value)}
            style={styles.select}
          >
            {LANGUAGE_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>
        <div style={styles.settingRow}>
          <label style={styles.label}>Speak To</label>
          <select
            value={speakTargetLanguage}
            onChange={(e) => onSpeakTargetLanguageChange(e.target.value)}
            style={styles.select}
          >
            {LANGUAGE_OPTIONS.filter((o) => o.value !== "").map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>
        <div style={styles.settingRow}>
          <label style={styles.label}>Output</label>
          <select
            aria-label="Output"
            value={outputDevice}
            onChange={(e) => onOutputDeviceChange(e.target.value)}
            style={styles.select}
          >
            <option value="">Default (Speakers)</option>
            {outputDevices.map((dev) => (
              <option key={dev.uid} value={dev.uid}>
                {dev.name}
              </option>
            ))}
          </select>
        </div>
        <div style={styles.settingRow}>
          <label style={styles.label}>Microphone</label>
          <select
            aria-label="Microphone"
            value={inputDevice}
            onChange={(e) => onInputDeviceChange(e.target.value)}
            style={styles.select}
          >
            <option value="">Default (System)</option>
            {inputDevices.map((dev) => (
              <option key={dev.uid} value={dev.uid}>
                {dev.name}
              </option>
            ))}
          </select>
        </div>
        <div style={styles.settingRow}>
          <label style={styles.label}>타이핑 입력</label>
          <div style={{ display: "flex", gap: "4px", flex: 1 }}>
            <button
              onClick={() => onShowTypeInputChange(true)}
              style={{
                ...styles.toggleBtn,
                ...(showTypeInput ? styles.toggleActive : {}),
              }}
            >
              표시
            </button>
            <button
              onClick={() => onShowTypeInputChange(false)}
              style={{
                ...styles.toggleBtn,
                ...(!showTypeInput ? styles.toggleActive : {}),
              }}
            >
              숨김
            </button>
          </div>
        </div>
        {(
          <>
            <div style={styles.settingRow}>
              <label style={styles.label}>Voice</label>
              <div style={{ display: "flex", gap: "4px", flex: 1 }}>
                <button
                  onClick={() => onVoiceCloneChange(true)}
                  style={{
                    ...styles.toggleBtn,
                    ...(voiceClone ? styles.toggleActive : {}),
                  }}
                >
                  My Voice
                </button>
                <button
                  onClick={() => onVoiceCloneChange(false)}
                  style={{
                    ...styles.toggleBtn,
                    ...(!voiceClone ? styles.toggleActive : {}),
                  }}
                >
                  Preset
                </button>
              </div>
            </div>
            {!voiceClone && (
              <div style={styles.settingRow}>
                <label style={styles.label}>TTS Voice</label>
                <select
                  value={ttsVoice}
                  onChange={(e) => onTtsVoiceChange(e.target.value)}
                  style={styles.select}
                >
                  {VOICE_OPTIONS.map((opt) => (
                    <option key={opt.value} value={opt.value}>
                      {opt.label}
                    </option>
                  ))}
                </select>
              </div>
            )}
            <div style={styles.settingRow}>
              <label style={styles.label}>Mic</label>
              <div style={{ display: "flex", gap: "4px", flex: 1 }}>
                {MIC_MODE_OPTIONS.map((opt) => (
                  <button
                    key={opt.value}
                    onClick={() => onMicModeChange(opt.value)}
                    style={{
                      ...styles.toggleBtn,
                      ...(micMode === opt.value ? styles.toggleActive : {}),
                    }}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            </div>
            {micMode === "push_to_talk" && (
              <div
                style={{
                  ...styles.pttIndicator,
                  backgroundColor: micActive
                    ? "rgba(76, 175, 80, 0.25)"
                    : "rgba(255,255,255,0.04)",
                  borderColor: micActive
                    ? "rgba(76, 175, 80, 0.5)"
                    : "rgba(255,255,255,0.1)",
                }}
              >
                {micActive ? "Recording..." : "Hold Space to talk"}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  overlay: {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    background: "rgba(0,0,0,0.88)",
    backdropFilter: "blur(12px)",
    WebkitBackdropFilter: "blur(12px)",
    zIndex: 10,
    display: "flex",
    flexDirection: "column",
    animation: "slideDown 0.2s ease",
  },
  header: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "8px 12px 6px",
    borderBottom: "1px solid rgba(255,255,255,0.08)",
    flexShrink: 0,
  },
  title: {
    color: "rgba(255,255,255,0.6)",
    fontSize: "11px",
    fontWeight: 500,
    userSelect: "none",
  },
  closeBtn: {
    padding: "2px 6px",
    border: "1px solid rgba(255,255,255,0.1)",
    borderRadius: "4px",
    backgroundColor: "rgba(255,255,255,0.05)",
    color: "rgba(255,255,255,0.5)",
    fontSize: "11px",
    cursor: "pointer",
  },
  content: {
    flex: 1,
    overflowY: "auto",
    padding: "8px 12px",
    display: "flex",
    flexDirection: "column",
    gap: "5px",
  },
  settingRow: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
  },
  label: {
    color: "rgba(255,255,255,0.5)",
    fontSize: "11px",
    flexShrink: 0,
    minWidth: "65px",
  },
  input: {
    width: "100%",
    padding: "5px 8px",
    border: "1px solid rgba(255,255,255,0.12)",
    borderRadius: "5px",
    backgroundColor: "rgba(255,255,255,0.06)",
    color: "#fff",
    fontSize: "12px",
    outline: "none",
    boxSizing: "border-box",
  },
  select: {
    flex: 1,
    padding: "4px 6px",
    border: "1px solid rgba(255,255,255,0.12)",
    borderRadius: "5px",
    backgroundColor: "rgba(255,255,255,0.06)",
    color: "#fff",
    fontSize: "12px",
    outline: "none",
  },
  toggleBtn: {
    flex: 1,
    padding: "4px 6px",
    border: "1px solid rgba(255,255,255,0.12)",
    borderRadius: "5px",
    backgroundColor: "rgba(255,255,255,0.05)",
    color: "rgba(255,255,255,0.5)",
    fontSize: "11px",
    cursor: "pointer",
  },
  toggleActive: {
    backgroundColor: "rgba(76, 175, 80, 0.2)",
    borderColor: "rgba(76, 175, 80, 0.35)",
    color: "#4ade80",
  },
  pttIndicator: {
    padding: "6px 10px",
    border: "1px solid",
    borderRadius: "6px",
    textAlign: "center" as const,
    fontSize: "12px",
    color: "rgba(255,255,255,0.7)",
    userSelect: "none" as const,
    transition: "all 0.15s ease",
  },
};

export default SettingsDrawer;
