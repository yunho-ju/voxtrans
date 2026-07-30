import React, { useState } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";

const TrafficLights: React.FC = () => {
  const [hovered, setHovered] = useState(false);

  const handleClose = async () => {
    const win = getCurrentWindow();
    await win.close();
  };

  const handleMinimize = async () => {
    const win = getCurrentWindow();
    await win.minimize();
  };

  const handleMaximize = async () => {
    const win = getCurrentWindow();
    await win.toggleMaximize();
  };

  return (
    <div
      style={styles.container}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <button
        style={{ ...styles.button, backgroundColor: "#ff5f57" }}
        onClick={handleClose}
        title="Close"
      >
        {hovered && <span style={styles.icon}>&#xd7;</span>}
      </button>
      <button
        style={{ ...styles.button, backgroundColor: "#febc2e" }}
        onClick={handleMinimize}
        title="Minimize"
      >
        {hovered && <span style={styles.icon}>&minus;</span>}
      </button>
      <button
        style={{ ...styles.button, backgroundColor: "#28c840" }}
        onClick={handleMaximize}
        title="Maximize"
      >
        {hovered && (
          <span style={{ ...styles.icon, fontSize: "8px" }}>&#x2197;</span>
        )}
      </button>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
    padding: "0 10px",
    flexShrink: 0,
  },
  button: {
    width: "12px",
    height: "12px",
    borderRadius: "50%",
    border: "none",
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    padding: 0,
    transition: "filter 0.15s ease",
  },
  icon: {
    fontSize: "10px",
    lineHeight: 1,
    color: "rgba(0,0,0,0.6)",
    fontWeight: "bold",
  },
};

export default TrafficLights;
