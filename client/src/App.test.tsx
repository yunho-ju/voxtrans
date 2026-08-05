// Critic round 1, findings 1 and 3 (specs/002-audio-routing): nothing
// previously proved the SettingsDrawer picker's selected microphone UID
// actually reaches `invoke("start_capture", ...)`, and nothing proved that
// a device-selection failure reported by the backend (`server-error`)
// reaches the user. Both hops are pure frontend wiring, invisible to
// `tsc`/`vite build` (a passing type-check proves the *shapes* line up,
// not that a value flows through them) and to the server-side/Rust suites,
// which cannot see React state or Tauri's IPC boundary at all. This is the
// client's first test infrastructure -- see `evidence/US3.md`'s "Critic
// round 1 fixes" section for why Vitest + Testing Library, not a manual
// harness, was the right amount of infrastructure for exactly two frontend
// wiring facts.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import App from "./App";

// Deliberately not named "BlackHole 2ch" (unlike the input device below):
// giving the output and input pickers a colliding option name makes
// `findByRole("option", { name: ... })` ambiguous across both `<select>`s.
const OUTPUT_DEVICES: string[][] = [["USB Audio Output", "USBAudioOutput_UID"]];
const INPUT_DEVICES: string[][] = [
  ["BlackHole 2ch", "BlackHole2ch_UID"],
  ["MacBook Pro 마이크", "BuiltInMicrophoneDevice"],
];

// `listen` callbacks registered by the app, keyed by event name, so a test
// can simulate the backend emitting an event (e.g. `server-error`) without
// a real Tauri runtime -- the only way to observe the receiving half of
// finding 3 without launching the GUI.
type ListenCallback = (event: { payload: string }) => void;
let listenCallbacks: Record<string, ListenCallback[]>;

const invokeMock = vi.fn(async (cmd: string, _args?: Record<string, unknown>) => {
  if (cmd === "list_output_devices") return OUTPUT_DEVICES;
  if (cmd === "list_devices") return INPUT_DEVICES;
  return undefined;
});

vi.mock("@tauri-apps/api/core", () => ({
  invoke: (cmd: string, args?: Record<string, unknown>) => invokeMock(cmd, args),
}));

vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn(async (event: string, callback: ListenCallback) => {
    (listenCallbacks[event] ??= []).push(callback);
    return () => {
      listenCallbacks[event] = (listenCallbacks[event] ?? []).filter((cb) => cb !== callback);
    };
  }),
}));

function emit(event: string, payload: string) {
  for (const cb of listenCallbacks[event] ?? []) cb({ payload });
}

beforeEach(() => {
  invokeMock.mockClear();
  listenCallbacks = {};
  window.localStorage.clear();
});

afterEach(() => {
  cleanup();
});

/** Opens Settings, selects `uid` in the Microphone picker, then clicks Start. */
async function selectMicrophoneAndStartCapture(uid: string) {
  render(<App />);

  fireEvent.click(screen.getByTitle("Settings"));

  const micSelect = await screen.findByLabelText("Microphone");
  // The option only exists once list_devices() has resolved and populated
  // inputDevices -- waiting for it is what proves the picker is wired to
  // real loaded devices, not just accepting arbitrary values.
  await screen.findByRole("option", { name: "BlackHole 2ch" });
  fireEvent.change(micSelect, { target: { value: uid } });

  fireEvent.click(screen.getByText("▶ Start"));

  // handleToggle is async (awaits set_server_url then start_capture);
  // flush microtasks so the start_capture call has actually happened.
  await Promise.resolve();
  await Promise.resolve();
}

describe("the microphone picker's selection reaches start_capture", () => {
  it("sends the selected UID as inputDeviceUid", async () => {
    await selectMicrophoneAndStartCapture("BlackHole2ch_UID");

    expect(invokeMock).toHaveBeenCalledWith(
      "start_capture",
      expect.objectContaining({ inputDeviceUid: "BlackHole2ch_UID" })
    );
  });

  it("sends null (the explicit 'system default' choice) when nothing is picked", async () => {
    // Not selecting anything leaves the picker on its default ("") option
    // -- start_capture must still receive an explicit null, matching
    // contracts/audio-devices.md's "empty UID means system default", not
    // an accidentally-omitted argument.
    render(<App />);
    fireEvent.click(screen.getByText("▶ Start"));
    await Promise.resolve();
    await Promise.resolve();

    expect(invokeMock).toHaveBeenCalledWith(
      "start_capture",
      expect.objectContaining({ inputDeviceUid: null })
    );
  });
});

// specs/002-audio-routing/briefs/US1-round2-T017-T017b.md (T017): the
// output device's *name* has to reach `start_capture` too, alongside its
// UID -- once the device has actually disappeared, `commands.rs` can no
// longer re-query a name for it (a fresh `list_output_devices()` no longer
// lists it), so "last known name" must be captured at selection time and
// threaded through, the same way `inputDeviceUid` already is.
async function selectOutputDeviceAndStartCapture(uid: string) {
  render(<App />);

  fireEvent.click(screen.getByTitle("Settings"));

  const outputSelect = await screen.findByLabelText("Output");
  // As with the microphone picker's helper above, waiting for the real
  // option proves the picker is wired to loaded devices, not accepting an
  // arbitrary value no `list_output_devices()` response ever produced.
  await screen.findByRole("option", { name: "USB Audio Output" });
  fireEvent.change(outputSelect, { target: { value: uid } });

  fireEvent.click(screen.getByText("▶ Start"));
  await Promise.resolve();
  await Promise.resolve();
}

describe("the output device picker's selection reaches start_capture with its name", () => {
  it("sends the selected UID and its last-known name", async () => {
    await selectOutputDeviceAndStartCapture("USBAudioOutput_UID");

    expect(invokeMock).toHaveBeenCalledWith(
      "start_capture",
      expect.objectContaining({
        outputDeviceUid: "USBAudioOutput_UID",
        outputDeviceName: "USB Audio Output",
      })
    );
  });

  it("sends null uid and null name (the explicit 'system default' choice) when nothing is picked", async () => {
    render(<App />);
    fireEvent.click(screen.getByText("▶ Start"));
    await Promise.resolve();
    await Promise.resolve();

    expect(invokeMock).toHaveBeenCalledWith(
      "start_capture",
      expect.objectContaining({ outputDeviceUid: null, outputDeviceName: null })
    );
  });
});

describe("a device-selection failure reported by the backend reaches the user", () => {
  it("shows the failed UID when the backend emits server-error", async () => {
    render(<App />);
    // The `listen("server-error", ...)` registration happens inside a
    // `useEffect`, which React may not have flushed synchronously by the
    // time `render()` returns; a microtask tick is enough for it to run.
    await Promise.resolve();

    // Simulates what commands.rs actually does: mic_capture::start_mic_capture
    // returns Err(String) naming the UID, which start_capture's spawned
    // thread emits as the Tauri event `server-error`. Wrapped in `act`
    // because React 18 batches the resulting `setLastError` update into a
    // microtask when triggered from outside a DOM event.
    act(() => {
      emit(
        "server-error",
        "Failed to start mic capture on device 'com.voxtrans.fabricated-uid-999'. The device may be missing, removed, or microphone permission may not be granted."
      );
    });

    // getByText throws (failing the test) if no matching node exists, so
    // reaching this line at all is already the proof; the extra assertion
    // just makes the intent explicit.
    expect(screen.getByText(/com\.voxtrans\.fabricated-uid-999/)).toBeTruthy();
  });

  it("shows nothing before any server-error has been emitted", () => {
    render(<App />);

    expect(screen.queryByRole("alert")).toBeNull();
  });

  // Brief instruction (specs/002-audio-routing/briefs/US1-round2-T017-T017b.md,
  // T017): "출력 장치 문구로 같은 패턴 테스트를 추가하라" -- the mic-flavored
  // case above only proves the generic delivery mechanism (any string
  // `commands.rs` emits reaches the banner); it says nothing about the
  // *output*-device wording `audio_output.rs`'s
  // `output_start_error_message` actually produces (UID + last-known name,
  // both branches). This simulates that exact wording reaching the UI.
  it("shows the failed output device's UID and last-known name when the backend emits server-error", async () => {
    render(<App />);
    await Promise.resolve();

    act(() => {
      emit(
        "server-error",
        "Failed to start audio output on device 'com.voxtrans.fabricated-output-uid-42' " +
          "('Fabricated Speakers'). The device may be missing, removed, or disconnected."
      );
    });

    expect(screen.getByText(/com\.voxtrans\.fabricated-output-uid-42/)).toBeTruthy();
    expect(screen.getByText(/Fabricated Speakers/)).toBeTruthy();
  });
});
