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

Every model choice here was made by measurement rather than reputation: each
candidate was run against the same latency budgets and the same set of
translation samples, and the losers lost on numbers. The harness that produced
those numbers ships with the repo and can be re-run against any other model —
see [Benchmarks](#benchmarks).

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

For development, run it from source — the app rebuilds and reloads as you edit:

```bash
cd client
npm install
npm run tauri dev
```

To produce `VoxTrans.app` and a `.dmg`:

```bash
cd client
APPLE_SIGNING_IDENTITY="Apple Development: YOUR NAME (TEAMID)" npm run tauri build
```

Bundles land in `client/src-tauri/target/release/bundle/`. Install by copying
`bundle/macos/VoxTrans.app` to `/Applications`.

**Set `APPLE_SIGNING_IDENTITY`.** Without it the build still succeeds, but the
app is ad-hoc signed, and macOS then identifies it by the hash of its
executable. Every rebuild is a different app as far as the system is concerned,
so **the permissions below have to be granted again after every single build**.
Signing with a stable certificate makes the identity `com.voxtrans.client` +
your certificate instead, and the grants persist. List the certificates you
already have with:

```bash
security find-identity -v -p codesigning
```

Any `Apple Development: …` certificate works for running the app on your own
machine (a free Apple ID gives you one through Xcode). Distributing the `.dmg`
to *other* people additionally needs a Developer ID certificate and
notarization, which requires the paid Apple Developer Program.

### 5. macOS permissions

| Permission | Needed for | Where |
|---|---|---|
| Screen & System Audio Recording | Listen mode (system-audio tap) | System Settings → Privacy & Security → Screen & System Audio Recording |
| Microphone | Speak mode | System Settings → Privacy & Security → Microphone |

The app asks for both on launch, and shows a banner with a shortcut to the
right settings pane if either is denied. macOS only prompts once per app
identity — after a denial, only System Settings can change the answer, and
**the app must be restarted for a newly granted permission to take effect.**

Grant them to whatever actually runs the process: in `tauri dev` that is your
terminal app, for a bundle it is `VoxTrans.app`.

> **If a permission looks enabled but nothing works**, its entry is probably
> bound to an older signature of the app (see the signing note above). The
> entry stays visible and checked while being dead. Remove the app from the
> list with `−`, or reset it outright, then relaunch:
>
> ```bash
> tccutil reset ScreenCapture com.voxtrans.client
> tccutil reset Microphone com.voxtrans.client
> ```

### 6. Speak-mode audio routing

Skip this if you only want subtitles. It is required for the meeting to hear
your translated voice.

The idea: VoxTrans speaks into a virtual audio device, and the meeting app
listens to that device instead of your real microphone.

```
your voice → mic → VoxTrans → translated speech → BlackHole 2ch → Meet/Zoom
```

**1. Install BlackHole 2ch**

```bash
brew install --cask blackhole-2ch
```

It has no interface — it appears as an audio device once installed. Confirm
with System Settings → Sound → Output; `BlackHole 2ch` should be listed. A
restart is sometimes needed before it shows up.

**2. Check its volume is 100%**

Open **Audio MIDI Setup** (`/Applications/Utilities/`), select `BlackHole 2ch`,
and make sure the output volume sliders are at maximum. This device is silent
to you, so a low setting is invisible — it just quietly attenuates your
translated voice on the way into the meeting. A −52 dB setting once made the
app inaudible to everyone and took a long time to find.

**3. Set the devices in VoxTrans** (settings → gear icon)

| Setting | Value |
|---|---|
| Microphone | your real mic, e.g. `MacBook Pro 마이크` |
| Output | `BlackHole 2ch` |

**Select both explicitly — do not leave them on "default".** If the system
default input happens to be BlackHole, the app listens to a silent loopback and
nothing reaches the server, with no error to explain it.

**4. Set the devices in the meeting app**

| Meet / Zoom setting | Value |
|---|---|
| Microphone | `BlackHole 2ch` |
| Speaker | your real speakers, e.g. `MacBook Pro 스피커` |

**5. Leave the macOS default input and output alone**

Never make BlackHole the *system* default. The app picks its devices
explicitly, and a BlackHole default routes the app's own speech back into its
own microphone path.

#### Hearing your own translated voice

You will not hear it — BlackHole has no speakers, which is the point. To
monitor it, open Audio MIDI Setup → `+` → **Create Multi-Output Device**, tick
both `BlackHole 2ch` and your speakers, and select that as the Output in
VoxTrans. Your translated voice then goes to the meeting *and* to you.

Note that your speech will then also come out of your speakers, where the
system-audio tap can pick it up. The app excludes its own output from the tap,
but this particular combination has not been verified in a live call — try it
before a meeting, not during one.

## Using it in a meeting

Start the server, then the app, then join the call. In the app:

- **Subtitles** appear on their own — listen mode runs for the whole session.
- **To speak**, hold **Space** and talk (default mic mode). The hold is one
  utterance: pauses inside it are fine, and the translation is spoken when you
  release. Finish your sentence before letting go, and keep each utterance
  under ~10 seconds.
- Space only reaches the app when **its window has focus**. Holding it while
  looking at your browser does nothing.
- Switching mic mode to **Auto (VAD)** removes the key entirely — convenient
  when testing alone, risky in a meeting, where every cough gets translated.

## Troubleshooting

Almost every failure here is silent — the pipeline keeps running and simply
produces nothing. Start with the logs: run the server in a terminal you can
see, and if the app misbehaves, launch it from a terminal too
(`/Applications/VoxTrans.app/Contents/MacOS/voxtrans-client`) so its stderr is
visible. Double-clicking the app sends that output nowhere.

| Symptom | Likely cause | Fix |
|---|---|---|
| Start immediately flips back to Stop | Capture failed to start, or the WebSocket was reset | Check the app's stderr. `Failed to start audio capture` = permissions; `WebSocket … Connection reset` = server side |
| No subtitles, no server activity at all | Screen Recording permission is missing or bound to an old signature | See §5, including the `tccutil` reset |
| Nothing happens when you speak | Push-to-talk: Space not held, or the app window is not focused | Focus the window and hold Space; the server logs `[speak]` only when mic audio arrives |
| The meeting cannot hear your translation | Output is not `BlackHole 2ch`, its volume is down, or the meeting app's mic is not BlackHole | Re-check every row in §6 |
| You cannot hear your own translation | Expected — BlackHole has no speakers | See "Hearing your own translated voice" in §6 |
| Changing a device in settings seems to do nothing | Fixed — device changes now apply live. On older builds this needed Stop/Start | Rebuild, or Stop/Start |
| Permission prompts return after every build | The build was ad-hoc signed | Set `APPLE_SIGNING_IDENTITY` (§4) |
| Server says `{"status":"ok"}` but the app cannot hold a session | A leftover server process from a crashed app still owns the port | Fixed — the server now reclaims port 8765 on startup. Otherwise kill it by hand |
| Translations are strange after editing the prompt | The template fell back to the default | Look for `Translation prompt … using default` in the server log — usually a missing `{text}` |

## Configuration

Environment variables (unset → built-in default):

| Variable | Effect |
|---|---|
| `VOXTRANS_TRANSLATION_MODEL` | Override the translation model (any MLX-LM repo id) |
| `VOXTRANS_ASR_MODEL` | Override the ASR model (Qwen3-ASR family or `mlx-community/whisper-*`) |
| `APPLE_SIGNING_IDENTITY` | Code-signing identity for `tauri build` (see [§4](#4-client)) |

The **translation prompt** is editable in the app's settings rather than here.
`{lang}` and `{text}` are substituted per segment; leaving the field empty uses
the built-in default. A template that drops `{text}` (or has an unbalanced
brace, or an unknown placeholder) falls back to the default rather than sending
the model an instruction with nothing to translate.

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
Run `python -m bench.run --help` for the full CLI.

The harness refuses to overwrite a previous run's results, which matters more
than it sounds: a benchmark that silently replaces its own evidence lets you
"improve" a number without noticing you changed what was being measured.

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
```

If you wonder *why* a constant or a model is what it is, the reasoning is in
the comment next to it. Numbers in this codebase came from measurements, and
the ones that were expensive to learn — why mic audio is discarded rather than
queued while TTS plays, why a 384 ms floor is not the 400 ms it looks like —
are written down where you will trip over them, not in a separate document.

## Status

Personal project under active development. Listen mode is used daily in real
meetings; speak mode's audio routing was recently reworked and is still being
validated in live calls. macOS-only by design (MLX + ScreenCaptureKit).

## License

[MIT](LICENSE). Note that the model weights are **not** part of this
repository and carry their own licenses (see each model's Hugging Face card).
