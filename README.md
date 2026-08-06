# VoxTrans

Real-time, fully local voice translation for meetings on Apple Silicon Macs.

VoxTrans sits next to your meeting app (Google Meet, Zoom, …) and translates in
both directions — everything runs on-device with [MLX](https://github.com/ml-explore/mlx):
no cloud APIs, no audio leaves your machine.

- **Listen mode** — captures the meeting audio, transcribes it, and shows
  translated subtitles in a transparent overlay.
- **Speak mode** — captures your microphone, translates what you say, and
  speaks the translation into the meeting through a virtual audio device.

Currently tuned for Korean ↔ English.

## How it works

```
┌─────────────────────────── macOS (Apple Silicon) ───────────────────────────┐
│                                                                             │
│  Meeting app ──(system audio tap)──┐                                        │
│                                    ▼                                        │
│  ┌──────────── Tauri client ────────────┐    WebSocket    ┌── Python ─────┐ │
│  │ React UI · subtitle overlay          │ ◄─────────────► │ FastAPI       │ │
│  │ Rust core · Swift audio capture      │  localhost:8765 │ Silero VAD    │ │
│  │ (ScreenCaptureKit / CoreAudio)       │                 │ ASR (MLX)     │ │
│  └──────────────────────────────────────┘                 │ Translate(MLX)│ │
│                                    ▲                      │ TTS (MLX)     │ │
│  Microphone ──(speak mode)─────────┘                      └───────────────┘ │
│  BlackHole 2ch ◄──(translated TTS back into the meeting)                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

The server chains VAD → ASR → translation → (speak mode only) TTS, one
inference at a time on the GPU. Default models:

| Stage | Model |
|---|---|
| VAD | Silero VAD |
| ASR | `mlx-community/Qwen3-ASR-1.7B-8bit` |
| Translation | `mlx-community/Qwen3.5-4B-MLX-4bit` |
| TTS | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit` |

Every model choice in this repo was made by measurement — the benchmark
harness and the side-by-side comparisons that drove each decision live in
`specs/` and can be re-run (see [Benchmarks](#benchmarks)).

## Requirements

- **Apple Silicon Mac** (MLX requirement; developed and measured on an
  M1 Pro / 32 GB — expect ~8.5 GB resident memory while a session is active)
- **macOS 15+** (ScreenCaptureKit audio tap)
- **Python 3.13**
- **Node.js 20+** and **Rust (stable)** — the client is Tauri 2
- **Xcode Command Line Tools** (the audio capture layer is Swift)
- ~17 GB disk for model weights (runtime models ≈ 8.7 GB; the rest are
  benchmark comparison candidates)
- For speak mode only: [BlackHole 2ch](https://github.com/ExistentialAudio/BlackHole)
  virtual audio device

## Setup

### 0. Prerequisites (one-time)

With [Homebrew](https://brew.sh):

```bash
xcode-select --install                 # Xcode Command Line Tools (Swift build)
brew install python@3.13 node          # Python 3.13, Node.js
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh   # Rust (stable)
brew install --cask blackhole-2ch      # speak mode only — virtual audio device
```

Verify: `python3.13 --version`, `node --version` (≥20), `cargo --version`.

### 1. Server

```bash
cd server
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
```

### 2. Model weights (one-time, ~17 GB)

```bash
cd server
.venv/bin/python -m bench.fetch_models          # downloads and verifies all snapshots
.venv/bin/python -m bench.fetch_models --verify-only   # re-check later without downloading
```

Weights land in the standard Hugging Face cache (`~/.cache/huggingface`).
The server **fails fast at startup** if a model is missing rather than
downloading mid-session.

### 3. Run the server

```bash
cd server
.venv/bin/python -m src.main
```

Health check: `curl http://localhost:8765/health` → `{"status":"ok"}`.
The WebSocket endpoint the client uses is `ws://localhost:8765/ws`.

### 4. Client

```bash
cd client
npm install
npm run tauri dev
```

### 5. macOS permissions

| Permission | Needed for | Where |
|---|---|---|
| Screen Recording | Listen mode (system-audio tap) | System Settings → Privacy & Security → Screen & System Audio Recording |
| Microphone | Speak mode | System Settings → Privacy & Security → Microphone |

Grant them to the app that actually runs the process — in dev that is your
terminal app; for a bundled build it is `VoxTrans.app`. **A rebuilt bundle may
need the permission granted again.**

### 6. Speak-mode audio routing (optional)

1. Install BlackHole 2ch.
2. In your meeting app, set the **microphone** to `BlackHole 2ch`.
3. In VoxTrans settings, set the TTS output device to `BlackHole 2ch` and pick
   your real microphone as the capture input.
4. Leave the BlackHole device volume at 100%. Lowering it silently attenuates
   the translated voice on its way into the meeting (a −52 dB setting once made
   the app inaudible and took a while to diagnose).

Do **not** set BlackHole as the *system default* input or output — the app
picks devices explicitly, and a BlackHole default routes the app's own TTS
back into its microphone path.

## Configuration

Environment variables (unset → built-in default):

| Variable | Effect |
|---|---|
| `VOXTRANS_TRANSLATION_MODEL` | Override the translation model (any MLX-LM repo id) |
| `VOXTRANS_ASR_MODEL` | Override the ASR model (Qwen3-ASR family or `mlx-community/whisper-*`) |

Latency/segmentation knobs (VAD hold, max segment length, intermediate-split
thresholds) are constants in `server/src/config.py`, each documented in place.

## Development

```bash
# Server unit tests (fast, no models needed)
cd server && .venv/bin/python -m pytest tests/unit -q

# Server integration tests (real audio devices; skip when hardware is absent)
cd server && .venv/bin/python -m pytest tests/integration

# Client
cd client && npx vitest run          # React/TS
cd client/src-tauri && cargo test    # Rust

# Test fixtures are generated deterministically with macOS `say`
# (voices: Yuna ko_KR, Samantha en_US) — regenerate with:
server/tests/fixtures/build_fixtures.sh
```

### Benchmarks

The `server/bench` harness measures each pipeline stage against explicit
latency budgets and refuses to overwrite evidence or recorded decisions:

```bash
cd server
.venv/bin/python -m bench.run --suite all --label local
.venv/bin/python -m bench.run --suite translation --model <repo-or-override> --label try
.venv/bin/python -m bench.run --suite asr --asr-model <repo> --label try
```

Suites: `e2e`, `asr`, `translation`, `tts_chunks`, `subtitle`, `memory`.
The CLI contract is documented in
`specs/001-realtime-latency/contracts/benchmark-cli.md`.

## Project layout

```
server/          Python: FastAPI WS server, VAD/ASR/translate/TTS pipeline
  src/           pipeline code (config.py holds every tunable)
  bench/         benchmark harness, model fetcher, model-comparison tooling
  tests/         unit + integration (integration needs real audio devices)
client/          Tauri 2 app
  src/           React UI (subtitle overlay, settings, controls)
  src-tauri/     Rust core (WS client, device selection, playback lifecycle)
  src-tauri/swift/  ScreenCaptureKit/CoreAudio capture layer
specs/           feature specs, task lists, measured evidence, decision records
```

`specs/` is unusual for an app repo and deliberate: it contains the measured
evidence (latency numbers, model side-by-sides, budget revisions) behind every
non-obvious decision in the code. If you wonder *why* a constant or model is
what it is, grep `specs/` first.

## Status

Personal project under active development. Listen mode is used daily in real
meetings; speak mode's audio routing was recently reworked and is still being
validated in live calls. macOS-only by design (MLX + ScreenCaptureKit).

## License

[MIT](LICENSE). Note that the model weights are **not** part of this
repository and carry their own licenses (see each model's Hugging Face card).
