"""The server exits when the app that spawned it disappears.

The app kills the server on its own clean exit, but that path does not run
when the app is force-quit or crashes. An orphaned server keeps port 8765 and
answers `/health` with 200 while resetting every WebSocket the next app opens
-- which presents as a broken app, not a stale server, and is diagnosable only
from the client's stderr.

`_watch_parent` runs only when `VOXTRANS_PARENT_PID` is set, so a server
started by hand from a terminal is unaffected.
"""

import os
import signal
import subprocess
import sys
import threading
import time

import pytest

from src.main import _watch_parent


def test_it_signals_this_process_once_the_parent_is_gone(monkeypatch):
    # A real short-lived process, so "gone" means what the OS means by it
    # rather than what a mock decides.
    victim = subprocess.Popen([sys.executable, "-c", "pass"])
    victim.wait()

    signalled: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: (
        signalled.append((pid, sig))
        if sig == signal.SIGTERM
        else _raise_if_dead(pid)
    ))

    def _raise_if_dead(pid: int) -> None:
        if pid == victim.pid:
            raise ProcessLookupError

    watcher = threading.Thread(target=_watch_parent, args=(victim.pid,), daemon=True)
    watcher.start()
    watcher.join(timeout=10)

    assert not watcher.is_alive(), "the watchdog should return after signalling"
    assert signalled == [(os.getpid(), signal.SIGTERM)], (
        "it must SIGTERM its own process -- a hard exit would leak the TTS "
        "worker instead of running uvicorn's shutdown"
    )


def test_it_keeps_waiting_while_the_parent_lives():
    # This process is its own parent for the purposes of the test: alive, so
    # the watchdog must not fire.
    fired = threading.Event()
    original_kill = os.kill

    def _kill(pid, sig):
        if sig == signal.SIGTERM:
            fired.set()
            return
        original_kill(pid, sig)

    watcher = threading.Thread(
        target=lambda: _patched_watch(os.getpid(), _kill), daemon=True
    )
    watcher.start()
    time.sleep(5)

    assert not fired.is_set(), "a live parent must not trigger a shutdown"


def _patched_watch(pid: int, kill_fn) -> None:
    """`_watch_parent` with `os.kill` swapped, without touching global state
    that the other test (or the real server) would then inherit."""
    import src.main as main_module

    original = main_module.os.kill
    main_module.os.kill = kill_fn
    try:
        _watch_parent(pid)
    finally:
        main_module.os.kill = original


def test_the_watchdog_is_opt_in(monkeypatch):
    """No `VOXTRANS_PARENT_PID` means no watchdog: a server run from a
    terminal must not die when that terminal does.
    """
    import src.main as main_module

    monkeypatch.delenv("VOXTRANS_PARENT_PID", raising=False)
    started: list[object] = []
    monkeypatch.setattr(
        main_module.threading, "Thread",
        lambda *a, **k: started.append((a, k)) or _NoopThread(),
    )
    monkeypatch.setattr(main_module, "_reclaim_port", lambda port: None)
    monkeypatch.setattr(main_module.uvicorn, "run", lambda *a, **k: None)

    main_module.main()

    assert started == [], "the watchdog must not start without the env var"


class _NoopThread:
    def start(self) -> None:
        pass


@pytest.mark.parametrize("value", ["", "not-a-pid", "12x"])
def test_a_malformed_parent_pid_is_ignored(monkeypatch, value):
    """A junk value must not crash startup or spawn a watcher for pid 0."""
    import src.main as main_module

    monkeypatch.setenv("VOXTRANS_PARENT_PID", value)
    started: list[object] = []
    monkeypatch.setattr(
        main_module.threading, "Thread",
        lambda *a, **k: started.append((a, k)) or _NoopThread(),
    )
    monkeypatch.setattr(main_module, "_reclaim_port", lambda port: None)
    monkeypatch.setattr(main_module.uvicorn, "run", lambda *a, **k: None)

    main_module.main()

    assert started == []
