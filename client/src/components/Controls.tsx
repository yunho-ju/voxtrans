import React, { useState } from "react";
import { invoke } from "@tauri-apps/api/core";

interface Props {
  status: string;
  onClear: () => void;
  serverUrl: string;
  outputDevice: string;
  mode: string;
  onModeChange: (mode: string) => void;
  micMode: string;
  micActive: boolean;
  textInput: string;
  onTextInputChange: (text: string) => void;
  onSendText: () => void;
  isTtsSpeaking: boolean;
}

type ActiveTab = "listen" | "speak" | "type";

const Controls: React.FC<Props> = ({
  status,
  onClear,
  serverUrl,
  outputDevice,
  mode,
  onModeChange,
  micMode,
  micActive,
  textInput,
  onTextInputChange,
  onSendText,
  isTtsSpeaking,
}) => {
  const isCapturing = status === "connected" || status === "connecting";

  // Derive initial tab from mode
  const modeToTab = (m: string): ActiveTab =>
    m === "speak" ? "speak" : "listen";
  const [activeTab, setActiveTab] = useState<ActiveTab>(() =>
    modeToTab(mode)
  );

  const handleToggle = async () => {
    try {
      if (isCapturing) {
        await invoke("stop_capture");
      } else {
        await invoke("set_server_url", { url: serverUrl });
        await invoke("start_capture", {
          outputDeviceUid: outputDevice || null,
        });
      }
    } catch (e) {
      console.error("Command error:", e);
    }
  };

  const handleTabChange = (tab: ActiveTab) => {
    setActiveTab(tab);
    if (tab === "listen") {
      onModeChange("listen");
    } else if (tab === "speak") {
      onModeChange("speak");
    } else if (tab === "type") {
      onModeChange("type");
    }
  };

  return (
    <div style={styles.container}>
      {/* Mode tabs */}
      <div style={styles.tabs}>
        {(["listen", "speak", "type"] as ActiveTab[]).map((tab) => (
          <button
            key={tab}
            onClick={() => handleTabChange(tab)}
            style={{
              ...styles.tab,
              ...(activeTab === tab ? styles.tabActive : {}),
            }}
          >
            {tab === "listen" ? "듣기" : tab === "speak" ? "말하기" : "타이핑"}
          </button>
        ))}
      </div>

      {/* Action area */}
      <div style={styles.actionArea}>
        {activeTab === "listen" && (
          <div style={styles.actionRow}>
            <button
              onClick={handleToggle}
              style={{
                ...styles.mainBtn,
                ...(isCapturing ? styles.mainBtnActive : {}),
              }}
            >
              {isCapturing ? "■ Stop" : "▶ Start"}
            </button>
            <button onClick={onClear} style={styles.auxBtn}>
              Clear
            </button>
          </div>
        )}

        {activeTab === "speak" && (
          <div style={styles.actionCol}>
            <div style={styles.actionRow}>
              <button
                onClick={handleToggle}
                style={{
                  ...styles.mainBtn,
                  ...(isCapturing ? styles.mainBtnActive : {}),
                }}
              >
                {isCapturing ? "■ Stop" : "▶ Start"}
              </button>
            </div>
            {isCapturing && micMode === "push_to_talk" && (
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
                {micActive ? "● Recording..." : "SPACE — Push to Talk"}
              </div>
            )}
            {isCapturing && micMode === "auto" && (
              <div style={styles.pttIndicator}>Auto VAD active</div>
            )}
          </div>
        )}

        {activeTab === "type" && (
          <div style={styles.actionCol}>
            <div style={styles.typeRow}>
              <input
                value={textInput}
                onChange={(e) => onTextInputChange(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !isTtsSpeaking && textInput.trim()) {
                    onSendText();
                  }
                }}
                placeholder="번역할 텍스트 입력..."
                disabled={isTtsSpeaking}
                style={{
                  ...styles.typeInput,
                  ...(isTtsSpeaking ? styles.typeInputDisabled : {}),
                }}
              />
              <button
                onClick={onSendText}
                disabled={isTtsSpeaking || !textInput.trim()}
                style={{
                  ...styles.sendBtn,
                  ...(isTtsSpeaking || !textInput.trim()
                    ? styles.sendBtnDisabled
                    : {}),
                }}
              >
                →
              </button>
            </div>
            <div style={styles.actionRow}>
              <button
                onClick={handleToggle}
                style={{
                  ...styles.auxBtn,
                  flex: 1,
                  ...(isCapturing ? styles.auxBtnActive : {}),
                }}
              >
                {isCapturing ? "■ Disconnect" : "▶ Connect"}
              </button>
              <button onClick={onClear} style={styles.auxBtn}>
                Clear
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    borderTop: "1px solid rgba(255,255,255,0.08)",
    background: "rgba(0,0,0,0.3)",
    flexShrink: 0,
    position: "relative" as const,
    zIndex: 1,
  },
  tabs: {
    display: "flex",
    borderBottom: "1px solid rgba(255,255,255,0.06)",
  },
  tab: {
    flex: 1,
    padding: "8px 4px",
    border: "none",
    borderBottom: "2px solid transparent",
    borderRadius: 0,
    backgroundColor: "transparent",
    color: "rgba(255,255,255,0.55)",
    fontSize: "12px",
    cursor: "pointer",
    fontWeight: 400,
    transition: "all 0.15s ease",
  },
  tabActive: {
    color: "rgba(255,255,255,0.85)",
    borderBottomColor: "rgba(255,255,255,0.5)",
    fontWeight: 500,
  },
  actionArea: {
    padding: "10px 12px 12px",
    minHeight: "80px",
    display: "flex",
    flexDirection: "column",
    justifyContent: "flex-start",
    gap: "6px",
  },
  actionRow: {
    display: "flex",
    gap: "6px",
  },
  actionCol: {
    display: "flex",
    flexDirection: "column",
    gap: "6px",
  },
  mainBtn: {
    flex: 1,
    padding: "7px 12px",
    border: "1px solid rgba(255,255,255,0.15)",
    borderRadius: "6px",
    backgroundColor: "rgba(255,255,255,0.08)",
    color: "rgba(255,255,255,0.9)",
    fontSize: "13px",
    cursor: "pointer",
    fontWeight: 500,
  },
  mainBtnActive: {
    backgroundColor: "rgba(248,113,113,0.2)",
    borderColor: "rgba(248,113,113,0.35)",
    color: "#f87171",
  },
  auxBtn: {
    padding: "7px 12px",
    border: "1px solid rgba(255,255,255,0.1)",
    borderRadius: "6px",
    backgroundColor: "rgba(255,255,255,0.05)",
    color: "rgba(255,255,255,0.75)",
    fontSize: "12px",
    cursor: "pointer",
  },
  auxBtnActive: {
    backgroundColor: "rgba(248,113,113,0.15)",
    borderColor: "rgba(248,113,113,0.3)",
    color: "#f87171",
  },
  pttIndicator: {
    padding: "6px 10px",
    border: "1px solid rgba(255,255,255,0.1)",
    borderRadius: "6px",
    textAlign: "center" as const,
    fontSize: "12px",
    color: "rgba(255,255,255,0.5)",
    userSelect: "none" as const,
    transition: "all 0.15s ease",
    backgroundColor: "rgba(255,255,255,0.04)",
  },
  typeRow: {
    display: "flex",
    gap: "6px",
    alignItems: "center",
  },
  typeInput: {
    flex: 1,
    padding: "7px 10px",
    border: "1px solid rgba(255,255,255,0.15)",
    borderRadius: "6px",
    backgroundColor: "rgba(255,255,255,0.08)",
    color: "#fff",
    fontSize: "13px",
    outline: "none",
  },
  typeInputDisabled: {
    opacity: 0.4,
    cursor: "not-allowed",
  },
  sendBtn: {
    width: "38px",
    height: "34px",
    border: "1px solid rgba(255,255,255,0.15)",
    borderRadius: "6px",
    backgroundColor: "rgba(100,180,255,0.15)",
    borderColor: "rgba(100,180,255,0.3)",
    color: "rgba(100,180,255,0.9)",
    fontSize: "16px",
    cursor: "pointer",
    flexShrink: 0,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  sendBtnDisabled: {
    opacity: 0.3,
    cursor: "not-allowed",
  },
};

export default Controls;
