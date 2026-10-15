# VoxTrans

Real-time voice translation for meetings, running entirely on your own Mac.

VoxTrans sits next to your meeting app (Google Meet, Zoom, …) and translates in
both directions:

- **Subtitles** — it captures the meeting's audio, transcribes it, and shows
  the translation in a transparent overlay above the call.
- **Speaking** — hold a key and talk; it translates what you said and speaks
  the translation into the meeting through a virtual microphone, so the other
  participants hear it in their language.

Everything happens on-device with [MLX](https://github.com/ml-explore/mlx).
No cloud APIs, no account, no audio leaving the machine — which is the point:
if the meeting is under an NDA, a cloud translator is not an option at all.

## What to expect

Honest numbers, measured on an M1 Pro / 32 GB, not projected:

| | |
|---|---|
| Delay from speech to translation | **~2.5 s** end to end |
| Memory while a session is active | **~8.5 GB** resident |
| Disk for model weights | ~9 GB (plus ~8 GB if you re-run the model comparisons) |
| Languages | 5 in the UI (Korean, English, Japanese, Chinese, Spanish), 10 accepted by the server; tuned and measured on **Korean ↔ English** |
| Platform | **Apple Silicon Mac, macOS 15+** — no Intel, no Windows, no Linux |

Two and a half seconds is not simultaneous interpretation. It is fast enough
to follow a conversation and to answer in it; it is not fast enough to talk
over someone. The delay is dominated by waiting for you to stop speaking, so
short sentences feel considerably better than long ones.

**Maturity**: a personal project, used daily in real meetings by its author.
Subtitles are solid. Speaking works but has fewer hours on it, and its audio
routing needs a one-time setup ([§6](#6-speak-mode-audio-routing)) that is
easy to get subtly wrong. Expect to spend twenty minutes on first setup,
mostly downloading models.

## How it works

```
┌─────────────────────────── macOS (Apple Silicon) ───────────────────────────┐
│                                                                             │
│  Meeting app ──(system audio tap)──┐                                        │
│                                    ▼                                        │
│  ┌──────────── Tauri client ────────────┐    WebSocket    ┌── Python ─────┐ │
│  │ React UI · subtitle overlay          │ ◄─────────────► │ FastAPI       │ │
│  │ Rust core · Swift audio capture      │ localhost:18765 │ Silero VAD    │ │
│  │ (ScreenCaptureKit / CoreAudio)       │                 │ ASR (MLX)     │ │
│  └──────────────────────────────────────┘                 │ Translate(MLX)│ │
│                                    ▲                      │ TTS (MLX)     │ │
│  Microphone ──(speak mode)─────────┘                      └───────────────┘ │
│  BlackHole 2ch ◄──(translated TTS back into the meeting)                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

A Tauri app captures audio and draws the overlay; a local Python server holds
the models and does the work. They talk over a WebSocket on localhost, which
is the only network traffic involved.

The server chains VAD → ASR → translation → (when speaking) TTS, one inference
at a time on the GPU — they contend for the same hardware, so serialising them
is what keeps the tail latency predictable. Default models:

| Stage | Model |
|---|---|
| VAD | Silero VAD |
| ASR | `mlx-community/Qwen3-ASR-1.7B-8bit` |
| Translation | `mlx-community/Qwen3.5-4B-MLX-4bit` |
| TTS | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit` |

The TTS stage has a second backend you can select at startup — see
[Swapping the TTS backend](#swapping-the-tts-backend).

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

Health check: `curl http://localhost:18765/health` → `{"status":"ok"}`.
The WebSocket endpoint the client uses is `ws://localhost:18765/ws`.

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
| System Audio Recording Only | Subtitles (the system-audio tap) | System Settings → Privacy & Security → **Screen & System Audio Recording** |
| Microphone | Speaking | System Settings → Privacy & Security → **Microphone** |

Note the first row: the app captures system audio with a Core Audio process
tap and **never records your screen**, but macOS files that permission under
the Screen Recording pane, where it appears as "System Audio Recording Only".
Grant that one; full Screen Recording is not required.

The app asks for the microphone on launch. It cannot ask for the system-audio
permission in advance — Core Audio taps have no such API — so macOS prompts for
that the first time you press Start. **A newly granted permission takes effect
only after restarting the app.**

Grant them to whatever actually runs the process: in `tauri dev` that is your
terminal app, for a bundle it is `VoxTrans.app`.

> **If a permission looks enabled but nothing works**, its entry may be bound
> to an older signature of the app (see the signing note in [§4](#4-client)).
> The entry stays visible and checked while being dead. Remove the app from the
> list with `−`, or reset it outright, then relaunch:
>
> ```bash
> tccutil reset ScreenCapture com.voxtrans.client   # covers system audio
> tccutil reset Microphone com.voxtrans.client
> ```
>
> Before reaching for this, check the app's stderr. "Start bounces back to
> Stop" looks like a permissions problem and usually is not — see
> [Troubleshooting](#troubleshooting).

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
| No subtitles, no server activity at all | The system-audio permission is missing or bound to an old signature | See §5, including the `tccutil` reset |
| Start bounces to Stop, but the log says `Audio capture started` | Not a permissions problem — the capture is fine and the WebSocket is failing. A stale server process is the usual cause | Kill any `python -m src.main` still running and start a fresh one |
| Nothing happens when you speak | Push-to-talk: Space not held, or the app window is not focused | Focus the window and hold Space; the server logs `[speak]` only when mic audio arrives |
| The meeting cannot hear your translation | Output is not `BlackHole 2ch`, its volume is down, or the meeting app's mic is not BlackHole | Re-check every row in §6 |
| You cannot hear your own translation | Expected — BlackHole has no speakers | See "Hearing your own translated voice" in §6 |
| Changing a device in settings seems to do nothing | Fixed — device changes now apply live. On older builds this needed Stop/Start | Rebuild, or Stop/Start |
| Permission prompts return after every build | The build was ad-hoc signed | Set `APPLE_SIGNING_IDENTITY` (§4) |
| Server says `{"status":"ok"}` but the app cannot hold a session | A leftover server process from a crashed app still owns the port | Fixed — the server now reclaims its port on startup. Otherwise kill it by hand |
| "서버 켜기" waits three minutes, then says the server is still loading | Another program may own the configured port, so the health probe reaches it instead of VoxTrans. `lsof -nP -iTCP:18765 -sTCP:LISTEN` checks the default port | Stop that program, or choose another port (see "Changing the server port") |
| Translations are strange after editing the prompt | The template fell back to the default | Look for `Translation prompt … using default` in the server log — usually a missing `{text}` |

## Configuration

Environment variables (unset → built-in default):

| Variable | Effect |
|---|---|
| `VOXTRANS_TRANSLATION_MODEL` | Override the translation model (any MLX-LM repo id) |
| `VOXTRANS_ASR_MODEL` | Override the ASR model (Qwen3-ASR family or `mlx-community/whisper-*`) |
| `VOXTRANS_PORT` | Override the default server port, 18765 (see "Changing the server port" below). A value that is not a port in 1–65535 is refused at startup rather than falling back |
| `APPLE_SIGNING_IDENTITY` | Code-signing identity for `tauri build` (see [§4](#4-client)) |

### Changing the server port

The default is 18765. Earlier builds used 8765; on launch, the app moves a
saved default `ws://localhost:8765/ws` or `ws://127.0.0.1:8765/ws` to the new
port. A custom server URL stays as saved. If another process already uses the
configured port, startup may fail and the app can wait out its three-minute
health-check window.

To move: set the server URL in the app's settings to the new port (e.g.
`ws://localhost:18766/ws`). The app passes that port to the server it spawns, so
both sides move together. A server you start by hand needs the variable
directly:

```bash
VOXTRANS_PORT=18766 python -m src.main
```

The server never evicts a program it did not start: on startup it reclaims the
port from a previous VoxTrans server, but an unrelated process holding it is
logged and left alone.

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

## Using it from your own code

The server is a plain FastAPI process on `localhost:18765`, so the pipeline is
usable without the app:

| | |
|---|---|
| `GET /health` | `{"status":"ok"}` once the models are loaded |
| `POST /translate` | `{"text": "...", "target_language": "korean"}` → `{"translated": "..."}` |
| `WS /ws` | the live pipeline: send audio frames, receive transcriptions and translations |

`/translate` is a single request, no audio involved — the cheapest way to reuse
the local translation model for something else entirely.

The WebSocket protocol is what the app speaks, and it is not documented yet;
`server/src/protocol.py` is the source of truth.

## Other models and backends

Translation and ASR are swappable by model, within MLX; TTS is swappable by
backend, and its seam (`server/src/tts_backends.py`) is the shape the other
two would take.

```bash
VOXTRANS_TRANSLATION_MODEL=mlx-community/<repo> \
VOXTRANS_ASR_MODEL=mlx-community/<repo> \
  .venv/bin/python -m src.main
```

Anything that is not an MLX model needs a small amount of glue that does not
exist yet on those two stages: `server/src/translation.py` and `asr.py` each
load and call a model directly rather than sitting behind an interface.

### Swapping the TTS backend

The app's settings have a **TTS Engine** picker, and changing it reloads the
server's TTS worker on the other backend — a few seconds during which nothing
speaks, and no restart. `VOXTRANS_TTS_BACKEND` sets which one the server boots
with (default `qwen3`); the picker adopts whatever is loaded and takes over
from there. Nothing else in the pipeline changes either way: the same process,
the same chunk protocol, the same suppression deadlines.

| | `qwen3` (default) | `kokoro` |
|---|---|---|
| Model | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit` | `prince-canuma/Kokoro-82M` |
| First audio | 394–401 ms, then streams | 249–267 ms, whole utterance at once |
| Full utterance | ~2.2 s for 3.2 s of speech | ~0.25 s for 3.9 s of speech |
| Voice clone | yes | no — falls back to a preset voice |
| Spoken languages | EN, ZH presets | EN out of the box; ZH, JA, ES, FR, IT, HI, PT each need their own G2P installed — **no Korean at all** |

(Measured on one machine, same phrase, warm — `Ryan`/English, second and third
synthesis after a discarded warm-up. Reproduce with `bench/run.py tts_chunks`.)

Kokoro is non-autoregressive, so it does not stream: it returns the whole
utterance in one chunk, which here still arrives sooner than the incumbent's
*first* chunk. What it gives up is the voice clone — voice is a stored style
vector, not a reference clip, so a clone request logs a warning and speaks in
the mapped preset instead.

The voice picker is filled from the server (`GET /voices`), so it shows
whatever the selected backend actually offers: the four presets under `qwen3`,
and under `kokoro` those four plus every voicepack in the downloaded snapshot
(54 of them). Switching engines resets the voice when the
new backend has no such voice, because `af_heart` means nothing to Qwen3-TTS
and would be spoken by an unconditioned voice that sounds wrong without
saying why.

Kokoro's voicepacks are per language, so the app's four voice names are slots
filled per target language rather than fixed voices: Ryan and Aiden are the
male slots (`am_michael`/`am_adam` in English, `jm_kumo` in Japanese,
`zm_yunxi`/`zm_yunjian` in Mandarin), Vivian and Serena the female ones. A
language that has no voice for a slot — Japanese ships one male voice, French
ships one voice total — substitutes within the same language and says so in
the log, because the wrong language is the worse mismatch. Any raw voicepack
name typed into the setting is passed through untouched.

The first utterance in a given language is slower (~900 ms here, against
~250 ms warm): Kokoro builds that language's front-end on first use.

Kokoro has no text front-end of its own, so it needs an optional G2P stack that
the default install does not carry. From `server/`:

```bash
.venv/bin/pip install -r requirements-kokoro.txt
VOXTRANS_TTS_BACKEND=kokoro .venv/bin/python -m src.main
```

That installs the Mandarin and Japanese extras as well as English. The other
supported languages (ES, FR, IT, HI, PT) use the espeak front-end. Kokoro has
no Korean voice. mlx-audio does not chunk long text, so split on newlines.

A missing English front-end fails at model load, with the install line. A
missing per-language one fails on the first request in that language, because
that is when Kokoro builds the pipeline for it. If a runtime switch fails, the
server reports the error and restores the prior engine. Switching while TTS is
speaking is rejected; try again after the utterance. `VOXTRANS_TTS_MODEL`
overrides the repo for either backend.

**Support for other LLM backends is very welcome** — an OpenAI-compatible
endpoint, Anthropic, Ollama, llama.cpp, LM Studio, or anything else you already
run. The natural shape is a translation backend selected by environment
variable, with the MLX path as one implementation among several; the same
applies to ASR and TTS.

Worth saying plainly: routing translation through a cloud API gives up the
property this project exists for — meeting audio staying on the machine. That
is a legitimate trade for someone who wants a bigger model on hardware that
cannot run one, and it should be an explicit choice rather than a default, so
the local path stays the one you get without asking. A local backend that is
not MLX (Ollama on the same Mac, say) gives up nothing at all.

If you want to add one, open an issue with the backend you have in mind and
the interface will be shaped around a real case rather than guessed at.

## Status

Under active development, and shaped by being used rather than planned: most
of what is in here was written after a meeting went wrong. macOS-only by
design — MLX and ScreenCaptureKit have no equivalent elsewhere, and the
audio-tap layer is Swift.

Korean ↔ English is where the measurements were taken, not a limit. Source and
target language are dropdowns (Korean, English, Japanese, Chinese, Spanish),
the server accepts ten languages, and the translation prompt itself is
editable — so a pair the UI does not list is a one-line change, not a fork.
What is genuinely English/Chinese-only is the **spoken** output: the TTS
voices are Ryan and Aiden (EN) and Vivian and Serena (ZH). Subtitles have no
such constraint.

Issues and pull requests are welcome. If something failed silently for you,
that report is more valuable than a feature request — silent failure is this
project's characteristic bug.

## License

[MIT](LICENSE). Note that the model weights are **not** part of this
repository and carry their own licenses (see each model's Hugging Face card).
