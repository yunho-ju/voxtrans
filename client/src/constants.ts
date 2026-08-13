/** Mirrors `TRANSLATION_PROMPT_DEFAULT` in `server/src/config.py`.
 *
 * The settings field is pre-filled with this rather than showing it as
 * placeholder text, so the prompt can be edited instead of retyped from
 * nothing. That makes this a real value the client sends, not a hint — keep
 * the two copies in step. They are identical strings today, so a drift changes
 * behaviour only for users who never touch the field.
 *
 * Clearing the field sends an empty prompt, which the server reads as "use
 * your own default".
 */
export const DEFAULT_TRANSLATION_PROMPT =
  "Translate to {lang} in a natural, conversational tone. " +
  "Do not include emojis or special symbols. " +
  "Reply with ONLY the translation.\n\n{text}";

/** How eagerly listen mode cuts speech into segments. A subtitle cannot appear
 * until its segment ends, so cutting sooner shows text sooner -- at the cost of
 * transcribing fragments, and of more ASR/translation passes through a
 * pipeline that runs strictly one at a time.
 *
 * Mirrors `SUBTITLE_PACE_PRESETS` in `server/src/config.py`. The server clamps
 * whatever it receives, so these are starting points rather than limits. */
export interface SubtitlePace {
  /** Seconds of speech before a mid-utterance split is allowed at all. */
  speechS: number;
  /** Milliseconds of silence that then counts as a place to split. */
  silenceMs: number;
  /** Hard cap: split here even with no pause at all. */
  maxS: number;
}

export const SUBTITLE_PACE_PRESETS: { label: string; value: SubtitlePace }[] = [
  { label: "빠름", value: { speechS: 3, silenceMs: 300, maxS: 12 } },
  { label: "균형", value: { speechS: 8, silenceMs: 400, maxS: 30 } },
  { label: "느림", value: { speechS: 15, silenceMs: 600, maxS: 45 } },
];

export const DEFAULT_SUBTITLE_PACE: SubtitlePace = SUBTITLE_PACE_PRESETS[0].value;

/** Mirrors the clamps in `server/src/config.py`. Applied here too so the field
 * shows the value that is actually in effect: the server silently clamps what
 * it receives and does not echo the result back, so an unclamped UI would
 * display a setting the pipeline is not using. */
export const SUBTITLE_PACE_BOUNDS = {
  speechS: [1, 60],
  silenceMs: [32, 3000],
  maxS: [3, 120],
} as const;

export function clampSubtitlePace(pace: SubtitlePace): SubtitlePace {
  const clamp = (n: number, [lo, hi]: readonly [number, number]) =>
    Math.max(lo, Math.min(hi, Math.round(n)));
  const speechS = clamp(pace.speechS, SUBTITLE_PACE_BOUNDS.speechS);
  return {
    speechS,
    silenceMs: clamp(pace.silenceMs, SUBTITLE_PACE_BOUNDS.silenceMs),
    // A cap below the split threshold would make the threshold dead config:
    // the forced split fires first, every time.
    maxS: Math.max(clamp(pace.maxS, SUBTITLE_PACE_BOUNDS.maxS), speechS),
  };
}
