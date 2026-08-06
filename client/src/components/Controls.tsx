import React, { useCallback, useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";

interface Props {
  status: string;
  onClear: () => void;
  serverUrl: string;
  outputDevice: string;
  outputDeviceName: string;
  inputDevice: string;
  mode: string;
  onModeChange: (mode: string) => void;
  micMode: string;
  micActive: boolean;
  textInput: string;
  onTextInputChange: (text: string) => void;
  onSendText: () => void;
  isTtsSpeaking: boolean;
  /** Resolved server directory (localStorage override, then Rust-detected
   * dev default, else null) -- see App.tsx. Null means neither source
   * produced a usable path, so the "서버 켜기" flow can't spawn anything
   * and shows a hint pointing at Settings instead of a button. */
  serverDir: string | null;
}

/** Extracts the port from a `ws(s)://host[:port]/path` server URL, matching
 * the server's actual `/health` listener; falls back to the server's
 * documented default when the URL has no explicit port. Exported (Critic
 * round 1 minor: "무테스트") so it has its own direct unit tests instead of
 * only being exercised indirectly through component behaviour. */
export function portFromServerUrl(url: string): number {
  const match = url.match(/:(\d+)(?:\/|$)/);
  return match ? parseInt(match[1], 10) : 8765;
}

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

// Critic round 1 [major]: the server loads several GB of models inside
// FastAPI's lifespan before uvicorn binds `/health` at all, so 30s was not
// remotely long enough -- a normal, successful startup was routinely
// mistaken for a failure. 180 x 1s gives loading several minutes to finish
// before the poll gives up.
const HEALTH_POLL_MAX_ATTEMPTS = 180;
const HEALTH_POLL_INTERVAL_MS = 1000;
// Low-frequency probe used only while the banner is already showing (not
// during the button's own active poll above) -- Critic round 1 minor
// ("타임아웃 이후 배너 stale"): if the server comes up on its own after the
// poll gave up, the banner has to notice and clear itself.
const STALE_BANNER_REPROBE_INTERVAL_MS = 5000;

/** Polls `check_server_health` once per second (up to
 * `HEALTH_POLL_MAX_ATTEMPTS` times) until it reports healthy, or gives up. */
async function pollUntilHealthy(port: number): Promise<boolean> {
  for (let attempt = 0; attempt < HEALTH_POLL_MAX_ATTEMPTS; attempt++) {
    await sleep(HEALTH_POLL_INTERVAL_MS);
    const healthy = await invoke<boolean>("check_server_health", { port }).catch(
      () => false
    );
    if (healthy) return true;
  }
  return false;
}

type ActiveTab = "listen" | "speak" | "type";

const Controls: React.FC<Props> = ({
  status,
  onClear,
  serverUrl,
  outputDevice,
  outputDeviceName,
  inputDevice,
  mode,
  onModeChange,
  micMode,
  micActive,
  textInput,
  onTextInputChange,
  onSendText,
  isTtsSpeaking,
  serverDir,
}) => {
  const isCapturing = status === "connected" || status === "connecting";

  // Derive initial tab from mode
  const modeToTab = (m: string): ActiveTab =>
    m === "speak" ? "speak" : "listen";
  const [activeTab, setActiveTab] = useState<ActiveTab>(() =>
    modeToTab(mode)
  );

  // null = not checked yet, so the button/hint stay hidden until the first
  // check resolves rather than flashing on for a healthy server (C1).
  const [serverHealthy, setServerHealthy] = useState<boolean | null>(null);
  const [startingServer, setStartingServer] = useState(false);
  const [serverError, setServerError] = useState<string | null>(null);

  const checkHealth = useCallback(async () => {
    const port = portFromServerUrl(serverUrl);
    const healthy = await invoke<boolean>("check_server_health", { port }).catch(
      () => false
    );
    setServerHealthy(healthy);
    return healthy;
  }, [serverUrl]);

  // (b) "마운트/상태 전환 시": re-probes whenever the UI is (or becomes)
  // disconnected, including the initial mount.
  useEffect(() => {
    if (status === "disconnected") {
      checkHealth();
    }
  }, [status, checkHealth]);

  // Critic round 1 minor ("타임아웃 이후 배너 stale"): while the banner is
  // showing (disconnected + confirmed unhealthy) and no active poll is
  // already covering it, keep re-probing at a low frequency so a server
  // that finishes starting up on its own -- no further click needed --
  // makes the banner disappear by itself instead of staying stale forever.
  useEffect(() => {
    if (status !== "disconnected" || serverHealthy !== false || startingServer) {
      return;
    }
    const intervalId = setInterval(() => {
      checkHealth();
    }, STALE_BANNER_REPROBE_INTERVAL_MS);
    return () => clearInterval(intervalId);
  }, [status, serverHealthy, startingServer, checkHealth]);

  const handleToggle = async () => {
    try {
      if (isCapturing) {
        await invoke("stop_capture");
      } else {
        await invoke("set_server_url", { url: serverUrl });
        // An empty UID is the explicit "use the system default" choice,
        // not an accidental fallback (contracts/audio-devices.md "Device
        // selection"; research.md D0 -- the old behaviour was silently
        // defaulting with no way to choose otherwise, which is exactly
        // the bug this makes impossible: the default is now one option
        // among the picker's choices, reached deliberately, same as
        // outputDeviceUid already does for the output side).
        // The output device's name, captured at selection time from the
        // picker's device list (App.tsx `outputDevices`) rather than
        // re-queried later: once the device is actually gone, a fresh
        // enumeration can no longer find a name for it, so "last known
        // name" has to be threaded through now (specs/002-audio-routing
        // briefs/US1-round2-T017-T017b.md).
        await invoke("start_capture", {
          outputDeviceUid: outputDevice || null,
          outputDeviceName: outputDeviceName || null,
          inputDeviceUid: inputDevice || null,
        });
      }
    } catch (e) {
      console.error("Command error:", e);
      // (a) "Connect 실패 시": a down server is the most common reason
      // start_capture's set_server_url/start_capture pair fails, so
      // re-probe immediately rather than waiting for the next
      // disconnected-status transition (there may not be one).
      if (!isCapturing) {
        checkHealth();
      }
    }
  };

  const handleStartServer = async () => {
    // No `if (!serverDir)` guard here (Critic round 1 "정리"): this handler
    // is only ever wired to the button rendered in the `serverDir ? ... :
    // ...` branch below, so `serverDir` is already guaranteed truthy at
    // every real call site -- a runtime check here was unreachable.
    setServerError(null);
    setStartingServer(true);
    const port = portFromServerUrl(serverUrl);
    try {
      // Design decision: "스폰 전 이미 healthy면 no-op 성공" -- covers the
      // race where the server came up between the last probe and the
      // click, without ever calling start_server.
      const alreadyHealthy = await checkHealth();
      if (!alreadyHealthy) {
        await invoke("start_server", { serverDir });
        const becameHealthy = await pollUntilHealthy(port);
        if (!becameHealthy) {
          // Critic round 1 [major]: no longer implies a fresh click would
          // restart anything -- the backend's spawn guard means a retry
          // just resumes waiting on the same still-loading process.
          setServerError(
            "서버가 아직 응답하지 않습니다. 모델 로딩에 수 분 정도 걸릴 수 있습니다 " +
              "— 잠시 후 다시 시도하면 이어서 대기합니다."
          );
          setServerHealthy(false);
          setStartingServer(false);
          return;
        }
        setServerHealthy(true);
      }
      setStartingServer(false);
      await handleToggle();
    } catch (e) {
      setServerError(typeof e === "string" ? e : "서버를 시작하지 못했습니다");
      setServerHealthy(false);
      setStartingServer(false);
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

      {/* Server offline banner: shown only once the health probe has come
          back unhealthy while disconnected (C1: healthy or not-yet-checked
          renders nothing here). */}
      {status === "disconnected" && serverHealthy === false && (
        <div style={styles.serverBanner}>
          {serverDir ? (
            <>
              <button
                onClick={handleStartServer}
                disabled={startingServer}
                style={{
                  ...styles.serverBtn,
                  ...(startingServer ? styles.serverBtnDisabled : {}),
                }}
                data-testid="start-server-btn"
              >
                {startingServer ? "서버 시작 중…" : "서버 켜기"}
              </button>
              {serverError && (
                <div style={styles.serverErrorText} data-testid="server-error">
                  {serverError}
                </div>
              )}
            </>
          ) : (
            <div style={styles.serverHint} data-testid="server-hint">
              서버 경로를 설정에서 지정하세요
            </div>
          )}
        </div>
      )}

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
  serverBanner: {
    padding: "8px 12px 0",
    display: "flex",
    flexDirection: "column" as const,
    gap: "4px",
  },
  serverBtn: {
    padding: "7px 12px",
    border: "1px solid rgba(250,204,21,0.35)",
    borderRadius: "6px",
    backgroundColor: "rgba(250,204,21,0.15)",
    color: "#facc15",
    fontSize: "12px",
    cursor: "pointer",
    fontWeight: 500,
  },
  serverBtnDisabled: {
    opacity: 0.6,
    cursor: "not-allowed",
  },
  serverErrorText: {
    fontSize: "11px",
    color: "#f87171",
  },
  serverHint: {
    fontSize: "11px",
    color: "rgba(255,255,255,0.5)",
    textAlign: "center" as const,
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
