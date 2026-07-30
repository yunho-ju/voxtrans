import React, { useEffect, useRef } from "react";
import { TranscriptionEntry } from "../hooks/useTranscription";

interface Props {
  entries: TranscriptionEntry[];
}

const TranscriptionOverlay: React.FC<Props> = ({ entries }) => {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [entries]);

  return (
    <div ref={scrollRef} style={styles.container}>
      {entries.length === 0 ? (
        <div style={styles.placeholder}>Waiting for speech...</div>
      ) : (
        entries.map((entry) => (
          <div
            key={entry.id}
            className="entry-animate"
            style={{
              ...styles.entry,
              ...(entry.source === "speak" ? styles.speakEntry : {}),
              opacity: entry.isFinal ? 1 : 0.6,
            }}
          >
            <div style={styles.textRow}>
              {entry.source === "speak" && (
                <span style={styles.youBadge}>YOU</span>
              )}
              <span style={styles.text}>{entry.text}</span>
            </div>
            {entry.translation && (
              <div
                style={
                  entry.source === "speak"
                    ? styles.speakTranslation
                    : styles.translation
                }
              >
                {entry.translation}
              </div>
            )}
          </div>
        ))
      )}
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    flex: 1,
    overflowY: "auto",
    padding: "8px 12px",
    scrollBehavior: "smooth",
    willChange: "transform",
    contain: "layout paint" as const,
  },
  placeholder: {
    color: "rgba(255,255,255,0.35)",
    fontSize: "13px",
    textAlign: "center",
    paddingTop: "24px",
    fontStyle: "italic",
    textShadow: "0 1px 2px rgba(0,0,0,0.5)",
  },
  entry: {
    color: "#fff",
    fontSize: "15px",
    lineHeight: "1.5",
    marginBottom: "6px",
    padding: "2px 0",
    textShadow: "-1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000, 0 0 6px rgba(0,0,0,0.8)",
  },
  speakEntry: {
    borderLeft: "2px solid rgba(74,222,128,0.5)",
    paddingLeft: "8px",
  },
  textRow: {
    display: "flex",
    alignItems: "center",
    gap: "6px",
  },
  text: {
    wordBreak: "break-word" as const,
  },
  youBadge: {
    backgroundColor: "rgba(74,222,128,0.15)",
    color: "#4ade80",
    fontSize: "9px",
    fontWeight: "bold",
    padding: "1px 5px",
    borderRadius: "3px",
    flexShrink: 0,
    letterSpacing: "0.5px",
    textShadow: "none",
  },
  translation: {
    color: "#8cb4ff",
    fontSize: "13px",
    lineHeight: "1.4",
    marginTop: "2px",
    textShadow: "-1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000, 0 0 4px rgba(0,0,0,0.7)",
  },
  speakTranslation: {
    color: "#6ee7a0",
    fontSize: "13px",
    lineHeight: "1.4",
    marginTop: "2px",
    textShadow: "-1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000, 0 0 4px rgba(0,0,0,0.7)",
  },
};

export default TranscriptionOverlay;
