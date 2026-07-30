import { useEffect, useState, useCallback, useRef } from "react";
import { listen } from "@tauri-apps/api/event";

export interface TranscriptionEntry {
  id: number;
  text: string;
  isFinal: boolean;
  timestamp: number;
  segmentId?: number;
  translation?: string;
  source: "listen" | "speak";
}

interface TranscriptionMessage {
  type: string;
  text: string;
  is_final: boolean;
  segment_id?: number;
}

interface TranslationMessage {
  type: string;
  text: string;
  segment_id: number;
}

export function useTranscription(maxEntries = 50) {
  const [entries, setEntries] = useState<TranscriptionEntry[]>([]);
  const [connectionStatus, setConnectionStatus] = useState("disconnected");
  const [isTtsSpeaking, setIsTtsSpeaking] = useState(false);
  const nextId = useRef(0);

  useEffect(() => {
    const unlistenTranscription = listen<string>("transcription", (event) => {
      try {
        const msg: TranscriptionMessage = JSON.parse(event.payload);
        if (msg.text) {
          const entry: TranscriptionEntry = {
            id: nextId.current++,
            text: msg.text,
            isFinal: msg.is_final,
            timestamp: Date.now(),
            segmentId: msg.segment_id,
            source: "listen",
          };
          setEntries((prev) => [...prev.slice(-(maxEntries - 1)), entry]);
        }
      } catch {
        // ignore parse errors
      }
    });

    const unlistenTranslation = listen<string>("translation", (event) => {
      try {
        const msg: TranslationMessage = JSON.parse(event.payload);
        if (msg.text && msg.segment_id != null) {
          setEntries((prev) =>
            prev.map((entry) =>
              entry.segmentId === msg.segment_id
                ? { ...entry, translation: msg.text }
                : entry
            )
          );
        }
      } catch {
        // ignore parse errors
      }
    });

    const unlistenSpeakTranscription = listen<string>(
      "speak-transcription",
      (event) => {
        try {
          const msg: TranscriptionMessage = JSON.parse(event.payload);
          if (msg.text) {
            const entry: TranscriptionEntry = {
              id: nextId.current++,
              text: msg.text,
              isFinal: true,
              timestamp: Date.now(),
              segmentId: msg.segment_id,
              source: "speak",
            };
            setEntries((prev) => [...prev.slice(-(maxEntries - 1)), entry]);
            setIsTtsSpeaking(true);
          }
        } catch {
          // ignore parse errors
        }
      }
    );

    const unlistenSpeakTranslation = listen<string>(
      "speak-translation",
      (event) => {
        try {
          const msg: TranslationMessage = JSON.parse(event.payload);
          if (msg.text && msg.segment_id != null) {
            setEntries((prev) =>
              prev.map((entry) =>
                entry.segmentId === msg.segment_id
                  ? { ...entry, translation: msg.text }
                  : entry
              )
            );
          }
        } catch {
          // ignore parse errors
        }
      }
    );

    const unlistenTtsEnd = listen<string>("tts-end", () => {
      setIsTtsSpeaking(false);
    });

    const unlistenStatus = listen<string>("connection-status", (event) => {
      setConnectionStatus(event.payload);
    });

    return () => {
      unlistenTranscription.then((fn) => fn());
      unlistenTranslation.then((fn) => fn());
      unlistenSpeakTranscription.then((fn) => fn());
      unlistenSpeakTranslation.then((fn) => fn());
      unlistenTtsEnd.then((fn) => fn());
      unlistenStatus.then((fn) => fn());
    };
  }, [maxEntries]);

  const clearEntries = useCallback(() => {
    setEntries([]);
  }, []);

  return { entries, connectionStatus, isTtsSpeaking, clearEntries };
}
