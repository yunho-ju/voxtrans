// Ticket: server-off -> "서버 켜기" button (spawn + /health poll + auto
// connect). Before this, a down server left Connect failing with a raw,
// swallowed error (Controls.tsx:74 `console.error` only) -- nothing in the
// UI told the user the server itself needed starting, let alone offered to
// start it. These tests prove the health-check-driven button, its three
// failure/edge paths, and the "already healthy" race are all wired through
// `invoke`, without a real server (mocked `check_server_health` /
// `start_server` / `start_capture` responses only).
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import Controls, { portFromServerUrl } from "./Controls";

const START_CAPTURE_DEFAULT_ARGS = {
  outputDeviceUid: null,
  outputDeviceName: null,
  inputDeviceUid: null,
};

const invokeMock = vi.fn(
  async (_cmd: string, _args?: Record<string, unknown>): Promise<unknown> => undefined
);

vi.mock("@tauri-apps/api/core", () => ({
  invoke: (cmd: string, args?: Record<string, unknown>) => invokeMock(cmd, args),
}));

const baseProps = {
  status: "disconnected",
  onClear: () => {},
  serverUrl: "ws://localhost:18765/ws",
  outputDevice: "",
  outputDeviceName: "",
  inputDevice: "",
  showTypeInput: true,
  micMode: "auto",
  micActive: false,
  textInput: "",
  onTextInputChange: () => {},
  onSendText: () => {},
  isTtsSpeaking: false,
  serverDir: "/repo/server",
};

/** Flushes the microtask chain the mount-time health check runs through
 * (`useEffect` -> `invoke` -> `setState`) without touching any timers. */
async function flushHealthCheck() {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
}

function countCallsTo(cmd: string): number {
  return invokeMock.mock.calls.filter(([c]) => c === cmd).length;
}

beforeEach(() => {
  invokeMock.mockReset();
  invokeMock.mockImplementation(async () => undefined);
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

// portFromServerUrl: the pure function behind every check_server_health /
// start_server call's port argument -- Critic round 1 minor ("무테스트").
describe("portFromServerUrl", () => {
  it("returns the server's documented default when the URL has no explicit port", () => {
    expect(portFromServerUrl("ws://localhost/ws")).toBe(18765);
  });

  it("parses an explicit port out of the URL", () => {
    expect(portFromServerUrl("ws://localhost:9999/ws")).toBe(9999);
  });

  it("falls back to the default for a URL with no parseable port at all", () => {
    expect(portFromServerUrl("not-a-url")).toBe(18765);
  });
});

// C1
describe("when the server is already healthy", () => {
  it("probes check_server_health with the URL's port, and shows no button", async () => {
    invokeMock.mockImplementation(async (cmd: string) => {
      if (cmd === "check_server_health") return true;
      return undefined;
    });

    render(<Controls {...baseProps} />);
    await flushHealthCheck();

    // Critic round 1 ("준-공허"): the button being absent is also true if
    // the health check silently never ran at all -- pin down that the
    // probe itself happened, with the port actually parsed from serverUrl.
    expect(invokeMock).toHaveBeenCalledWith("check_server_health", { port: 18765 });
    expect(screen.queryByTestId("start-server-btn")).toBeNull();
    expect(screen.queryByTestId("server-hint")).toBeNull();
  });

  it("probes the port parsed from a non-default serverUrl", async () => {
    invokeMock.mockImplementation(async (cmd: string) => {
      if (cmd === "check_server_health") return true;
      return undefined;
    });

    render(<Controls {...baseProps} serverUrl="ws://localhost:9999/ws" />);
    await flushHealthCheck();

    expect(invokeMock).toHaveBeenCalledWith("check_server_health", { port: 9999 });
  });
});

// C2
describe("when the server is down and a server directory is known", () => {
  it("shows the button; clicking spawns, polls, then auto-connects", async () => {
    let healthCalls = 0;
    invokeMock.mockImplementation(async (cmd: string) => {
      if (cmd === "check_server_health") {
        healthCalls += 1;
        // False on mount and on the click's own pre-check (calls 1-3),
        // true starting the second poll after start_server (call 4).
        return healthCalls > 3;
      }
      return undefined;
    });

    render(<Controls {...baseProps} />);
    await flushHealthCheck();

    // Fetched (and fake timers enabled) only after the mount-time health
    // check has already resolved with real timers -- findByTestId's own
    // internal polling would otherwise hang forever against a fake clock
    // nothing is advancing yet.
    const btn = screen.getByTestId("start-server-btn");
    expect(btn.textContent).toBe("서버 켜기");
    vi.useFakeTimers();

    await act(async () => {
      fireEvent.click(btn);
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(invokeMock).toHaveBeenCalledWith("start_server", { serverDir: "/repo/server" });

    await act(async () => {
      await vi.advanceTimersByTimeAsync(3000);
    });

    // Critic round 1 ("auto-connect 인자 단언 공백"): proves auto-connect
    // reuses the exact same connect flow the manual Connect/Start button
    // uses, not just that *some* start_capture call happened.
    expect(invokeMock).toHaveBeenCalledWith("set_server_url", {
      url: baseProps.serverUrl,
    });
    expect(invokeMock).toHaveBeenCalledWith(
      "start_capture",
      START_CAPTURE_DEFAULT_ARGS
    );
  });
});

// C3
describe("when start_server fails", () => {
  it("shows the error message and never polls", async () => {
    invokeMock.mockImplementation(async (cmd: string) => {
      if (cmd === "check_server_health") return false;
      if (cmd === "start_server") {
        throw "Python virtualenv not found at /repo/server/.venv/bin/python";
      }
      return undefined;
    });

    render(<Controls {...baseProps} />);
    await flushHealthCheck();

    const btn = screen.getByTestId("start-server-btn");
    await act(async () => {
      fireEvent.click(btn);
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(screen.getByTestId("server-error").textContent).toContain(
      "/repo/server/.venv/bin/python"
    );
    // Only the mount check and the click's own pre-check ran -- no poll.
    expect(countCallsTo("check_server_health")).toBe(2);
    expect((screen.getByTestId("start-server-btn") as HTMLButtonElement).disabled).toBe(
      false
    );
  });
});

// C4
describe("when the server never becomes healthy", () => {
  it("times out after polling, shows an error, and re-enables the button", async () => {
    invokeMock.mockImplementation(async (cmd: string) => {
      if (cmd === "check_server_health") return false;
      return undefined;
    });

    render(<Controls {...baseProps} />);
    await flushHealthCheck();

    const btn = screen.getByTestId("start-server-btn");
    vi.useFakeTimers();

    await act(async () => {
      fireEvent.click(btn);
      await Promise.resolve();
      await Promise.resolve();
    });

    // Critic round 1 [major]: the server loads several GB of models inside
    // FastAPI's lifespan before uvicorn ever binds a socket, so 30s was
    // nowhere near enough -- polling now runs up to 180 x 1s. Give it a
    // comfortable margin past that.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(181_000);
    });

    // Critic round 1 [major]: the message now has to actually say a retry
    // resumes waiting rather than restarting -- the old generic "timed
    // out" text would have been literally true again after a duplicate
    // spawn, which is exactly the bug this round fixes.
    expect(screen.getByTestId("server-error").textContent).toMatch(/수 분|이어서/);
    expect((screen.getByTestId("start-server-btn") as HTMLButtonElement).disabled).toBe(
      false
    );
    expect(invokeMock).not.toHaveBeenCalledWith(
      "start_capture",
      expect.anything()
    );
  });
});

// C5
describe("when no server directory can be resolved", () => {
  it("shows a hint instead of a button", async () => {
    invokeMock.mockImplementation(async (cmd: string) => {
      if (cmd === "check_server_health") return false;
      return undefined;
    });

    render(<Controls {...baseProps} serverDir={null} />);
    await flushHealthCheck();

    expect(screen.queryByTestId("start-server-btn")).toBeNull();
    expect(screen.getByTestId("server-hint")).toBeTruthy();
  });
});

// Design decision 2 ("스폰 전 이미 healthy면 no-op 성공"): covers the race
// where the server became healthy between the mount check and the click.
describe("when the server becomes healthy before start_server would be called", () => {
  it("skips start_server and connects directly", async () => {
    let healthCalls = 0;
    invokeMock.mockImplementation(async (cmd: string) => {
      if (cmd === "check_server_health") {
        healthCalls += 1;
        return healthCalls > 1; // unhealthy on mount, healthy from the click on
      }
      return undefined;
    });

    render(<Controls {...baseProps} />);
    await flushHealthCheck();

    const btn = await screen.findByTestId("start-server-btn");
    await act(async () => {
      fireEvent.click(btn);
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(invokeMock).not.toHaveBeenCalledWith(
      "start_server",
      expect.anything()
    );
    expect(invokeMock).toHaveBeenCalledWith("set_server_url", {
      url: baseProps.serverUrl,
    });
    expect(invokeMock).toHaveBeenCalledWith(
      "start_capture",
      START_CAPTURE_DEFAULT_ARGS
    );
  });
});

// Critic round 1 minor ("타임아웃 이후 배너 stale"): if the server finishes
// coming up on its own -- no further clicks -- the banner has to notice and
// clear itself, not sit there forever pointing at a server that is
// actually fine now.
describe("when the server becomes healthy on its own while the banner is showing", () => {
  it("clears the banner via a low-frequency background re-probe", async () => {
    let healthy = false;
    invokeMock.mockImplementation(async (cmd: string) => {
      if (cmd === "check_server_health") return healthy;
      return undefined;
    });

    // Fake timers must be active *before* the mount effect registers the
    // 5s re-probe interval -- switching afterwards would leave that
    // interval running on the real clock, unaffected by advancing the
    // fake one (the same ordering pitfall as the click-driven polling
    // tests above).
    vi.useFakeTimers();
    render(<Controls {...baseProps} />);
    await flushHealthCheck();

    expect(screen.getByTestId("start-server-btn")).toBeTruthy();

    healthy = true;

    await act(async () => {
      await vi.advanceTimersByTimeAsync(5000);
    });

    expect(screen.queryByTestId("start-server-btn")).toBeNull();
  });
});

describe("the typing row is a preference, not a mode", () => {
  it("renders the type input when showTypeInput is true", async () => {
    render(<Controls {...baseProps} showTypeInput={true} />);
    await flushHealthCheck();
    expect(screen.getByPlaceholderText("번역할 텍스트 입력...")).toBeTruthy();
  });

  it("hides the type input when showTypeInput is false", async () => {
    render(<Controls {...baseProps} showTypeInput={false} />);
    await flushHealthCheck();
    expect(screen.queryByPlaceholderText("번역할 텍스트 입력...")).toBeNull();
  });
});
