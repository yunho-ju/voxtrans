import React from "react";

interface Props {
  status: string;
}

const statusColors: Record<string, string> = {
  connected: "#4ade80",
  connecting: "#facc15",
  disconnected: "#f87171",
};

const StatusBar: React.FC<Props> = ({ status }) => {
  const color = statusColors[status] || statusColors.disconnected;

  return (
    <div style={styles.container}>
      <div
        style={{ ...styles.dot, backgroundColor: color }}
        className={status === "connecting" ? "status-connecting" : undefined}
      />
      <span style={styles.text}>{status}</span>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: "flex",
    alignItems: "center",
    gap: "5px",
    padding: "0 6px",
  },
  dot: {
    width: "7px",
    height: "7px",
    borderRadius: "50%",
    flexShrink: 0,
    boxShadow: "0 0 4px currentColor",
  },
  text: {
    color: "rgba(255,255,255,0.55)",
    fontSize: "10px",
    textTransform: "uppercase",
    letterSpacing: "0.5px",
    userSelect: "none",
  },
};

export default StatusBar;
