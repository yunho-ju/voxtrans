import React, { useRef, useState } from "react";
import {
  clampSubtitlePace,
  DEFAULT_TRANSLATION_PROMPT,
  SUBTITLE_PACE_PRESETS,
  type SubtitlePace,
} from "../constants";

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
  /** Applies the one-click meeting routing (Output → BlackHole, Mic → real
   * input) and returns a message describing what it did or why it couldn't. */
  onApplyMeetingPreset: () => string;
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
  /** Prompt template sent to the translation model. Pre-filled with the
   * default so it can be edited in place; an emptied field means "use the
   * server's own default". `{lang}` and `{text}` are filled in per segment. */
  translationPrompt: string;
  onTranslationPromptChange: (prompt: string) => void;
  /** How eagerly listen mode cuts speech into segments. Entered as numbers;
   * the preset buttons only fill them in. */
  subtitlePace: SubtitlePace;
  onSubtitlePaceChange: (pace: SubtitlePace) => void;
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

const PACE_FIELDS = [
  { key: "speechS", label: "분할 시작", unit: "초" },
  { key: "silenceMs", label: "침묵", unit: "ms" },
  { key: "maxS", label: "상한", unit: "초" },
] as const;

const toDraft = (pace: SubtitlePace): Record<string, string> => ({
  speechS: String(pace.speechS),
  silenceMs: String(pace.silenceMs),
  maxS: String(pace.maxS),
});

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
  onApplyMeetingPreset,
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
  translationPrompt,
  onTranslationPromptChange,
  subtitlePace,
  onSubtitlePaceChange,
}) => {
  const [presetResult, setPresetResult] = useState<string | null>(null);
  // The pace fields are edited as text and committed on blur or Enter.
  // Parsing every keystroke made them unusable: clearing "300" to type "350"
  // produced Number("") === 0, which came straight back as the field's value,
  // and each intermediate digit was sent to the server and clamped.
  const [paceDraft, setPaceDraft] = useState(() => toDraft(subtitlePace));
  const lastPaceRef = useRef(subtitlePace);
  if (lastPaceRef.current !== subtitlePace) {
    // Changed from outside (a preset button): adopt it, discarding any draft.
    lastPaceRef.current = subtitlePace;
    setPaceDraft(toDraft(subtitlePace));
  }

  const commitPace = (key: keyof SubtitlePace) => {
    const parsed = Number(paceDraft[key]);
    if (paceDraft[key].trim() === "" || !Number.isFinite(parsed)) {
      // Unparseable: put the field back rather than guessing at a number.
      setPaceDraft(toDraft(subtitlePace));
      return;
    }
    const next = clampSubtitlePace({ ...subtitlePace, [key]: parsed });
    // Show the clamped result even when nothing else changed, so a rejected
    // entry visibly snaps to the bound rather than lingering on screen.
    setPaceDraft(toDraft(next));
    if (
      next.speechS === subtitlePace.speechS &&
      next.silenceMs === subtitlePace.silenceMs &&
      next.maxS === subtitlePace.maxS
    ) {
      return;
    }
    onSubtitlePaceChange(next);
  };

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
        <button
          onClick={() => setPresetResult(onApplyMeetingPreset())}
          style={styles.presetBtn}
          data-testid="meeting-preset-btn"
        >
          회의 프리셋 — Output·마이크 자동 설정
        </button>
        {presetResult && (
          <div style={styles.presetResult} data-testid="meeting-preset-result">
            {presetResult}
          </div>
        )}
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
            <div style={styles.settingRow}>
              <label style={styles.label}>자막 속도</label>
              <div style={{ display: "flex", gap: "4px", flex: 1 }}>
                {SUBTITLE_PACE_PRESETS.map((preset) => {
                  const active =
                    preset.value.speechS === subtitlePace.speechS &&
                    preset.value.silenceMs === subtitlePace.silenceMs &&
                    preset.value.maxS === subtitlePace.maxS;
                  return (
                    <button
                      key={preset.label}
                      onClick={() => onSubtitlePaceChange(preset.value)}
                      style={{
                        ...styles.toggleBtn,
                        ...(active ? styles.toggleActive : {}),
                      }}
                    >
                      {preset.label}
                    </button>
                  );
                })}
              </div>
            </div>
            <div style={styles.paceRow}>
              {PACE_FIELDS.map((field) => (
                <label key={field.key} style={styles.paceField}>
                  <span style={styles.paceLabel}>{field.label}</span>
                  <input
                    type="text"
                    inputMode="numeric"
                    aria-label={field.label}
                    value={paceDraft[field.key]}
                    onChange={(e) =>
                      setPaceDraft((d) => ({ ...d, [field.key]: e.target.value }))
                    }
                    onBlur={() => commitPace(field.key)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") e.currentTarget.blur();
                      if (e.key === "Escape") {
                        setPaceDraft(toDraft(subtitlePace));
                        e.currentTarget.blur();
                      }
                    }}
                    style={styles.paceInput}
                  />
                  <span style={styles.paceUnit}>{field.unit}</span>
                </label>
              ))}
            </div>
            <div style={styles.hint}>
              {subtitlePace.speechS}초 넘게 말하는 중에 {subtitlePace.silenceMs}ms
              쉬면 거기서 끊고, 안 쉬면 {subtitlePace.maxS}초에서 강제로 끊습니다.
              짧게 끊을수록 자막이 빨리 뜨지만 문장이 갈리고, ASR·번역이 한 번에
              하나씩만 도는 구조라 너무 잘게 끊으면 번역이 밀립니다.
            </div>
            <div style={styles.settingRow}>
              <label style={styles.label}>Prompt</label>
              <textarea
                aria-label="Prompt"
                value={translationPrompt}
                onChange={(e) => onTranslationPromptChange(e.target.value)}
                rows={5}
                style={styles.textarea}
              />
            </div>
            <div style={styles.hint}>
              {"{lang}"} = 목표 언어, {"{text}"} = 인식된 문장. 다음 문장부터 바로
              적용됩니다.
              {translationPrompt !== DEFAULT_TRANSLATION_PROMPT && (
                <button
                  onClick={() =>
                    onTranslationPromptChange(DEFAULT_TRANSLATION_PROMPT)
                  }
                  style={styles.linkBtn}
                >
                  기본값으로
                </button>
              )}
            </div>
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
  textarea: {
    flex: 1,
    padding: "5px 8px",
    border: "1px solid rgba(255,255,255,0.12)",
    borderRadius: "5px",
    backgroundColor: "rgba(255,255,255,0.06)",
    color: "#fff",
    fontSize: "11px",
    lineHeight: 1.4,
    outline: "none",
    boxSizing: "border-box",
    resize: "vertical",
    fontFamily: "inherit",
  },
  paceRow: {
    display: "flex",
    gap: "6px",
    paddingLeft: "73px",
  },
  paceField: {
    display: "flex",
    alignItems: "center",
    gap: "3px",
    flex: 1,
  },
  paceLabel: {
    color: "rgba(255,255,255,0.4)",
    fontSize: "10px",
    flexShrink: 0,
  },
  paceInput: {
    width: "100%",
    minWidth: 0,
    padding: "3px 5px",
    border: "1px solid rgba(255,255,255,0.12)",
    borderRadius: "4px",
    backgroundColor: "rgba(255,255,255,0.06)",
    color: "#fff",
    fontSize: "11px",
    outline: "none",
  },
  paceUnit: {
    color: "rgba(255,255,255,0.35)",
    fontSize: "10px",
    flexShrink: 0,
  },
  linkBtn: {
    marginLeft: "6px",
    padding: 0,
    border: "none",
    background: "none",
    color: "rgba(120,180,255,0.85)",
    fontSize: "10px",
    cursor: "pointer",
    textDecoration: "underline",
  },
  hint: {
    color: "rgba(255,255,255,0.35)",
    fontSize: "10px",
    lineHeight: 1.5,
    paddingLeft: "73px",
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
  presetBtn: {
    padding: "6px 10px",
    border: "1px solid rgba(255,255,255,0.15)",
    borderRadius: "5px",
    backgroundColor: "rgba(255,255,255,0.08)",
    color: "rgba(255,255,255,0.85)",
    fontSize: "12px",
    cursor: "pointer",
    fontWeight: 500,
  },
  presetResult: {
    fontSize: "11px",
    color: "rgba(255,255,255,0.6)",
    lineHeight: 1.4,
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
