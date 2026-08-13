import asyncio
import logging
import os
import signal
import subprocess
import threading
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel

from .asr import asr_engine
from .config import HOST, PORT
from .translation import translation_engine
from .tts import tts_engine
from .vad import load_vad_model
from .ws_handler import handle_websocket

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Loading models...")
    # Load TTS first (mlx) before torch-based models to avoid segfault
    tts_engine.load()
    load_vad_model()
    asr_engine.load()
    translation_engine.load()
    logger.info("All models loaded. Server ready.")
    yield
    # Shutdown
    logger.info("Shutting down: cleaning up resources...")
    tts_engine.shutdown()
    logger.info("Shutdown complete.")


app = FastAPI(title="VoxTrans Server", lifespan=lifespan)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await handle_websocket(websocket)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


class TranslateRequest(BaseModel):
    text: str
    target_language: str = "english"


class TranslateResponse(BaseModel):
    translated: str


@app.post("/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest) -> TranslateResponse:
    result = await asyncio.to_thread(
        translation_engine.translate, req.text, req.target_language
    )
    return TranslateResponse(translated=result)


def _reclaim_port(port: int) -> None:
    """Kill a previous VoxTrans server still holding `port`, then wait for it.

    The app spawns the server as a child; when the app dies unexpectedly the
    server is reparented to init and keeps the port. The next start then either
    fails to bind or -- worse, as seen live -- leaves a stale process answering
    /health while resetting every WebSocket, which reads as "the app is broken"
    with no error anywhere.

    Only processes whose command line contains this server's module are killed:
    an unrelated program on 8765 is reported and left alone, since taking it
    down would be well beyond what starting VoxTrans should do.
    """
    try:
        out = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return  # no lsof, or it misbehaved: let uvicorn's own bind error speak

    for pid_text in out.split():
        pid = int(pid_text)
        if pid == os.getpid():
            continue
        try:
            cmdline = subprocess.run(
                ["ps", "-o", "command=", "-p", pid_text],
                capture_output=True, text=True, timeout=5,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        if "src.main" not in cmdline:
            logger.warning(
                "Port %d is held by PID %d, which is not a VoxTrans server "
                "-- leaving it alone: %s", port, pid, cmdline.strip()[:120],
            )
            continue
        logger.info("Port %d held by a previous server (PID %d); stopping it", port, pid)
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except PermissionError:
            logger.warning("Not permitted to stop PID %d; start will likely fail", pid)
            continue
        # TTS worker teardown means SIGTERM is not instant. Escalate rather
        # than race uvicorn's bind.
        for _ in range(50):  # up to 5s
            time.sleep(0.1)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
        else:
            logger.warning("PID %d ignored SIGTERM after 5s; sending SIGKILL", pid)
            try:
                os.kill(pid, signal.SIGKILL)
                time.sleep(0.5)
            except ProcessLookupError:
                pass


def _watch_parent(parent_pid: int) -> None:
    """Shut down once `parent_pid` is gone.

    The app sets `VOXTRANS_PARENT_PID` when it spawns the server, and only
    then does this run: a server started by hand from a terminal has no
    business exiting when that terminal does.

    The app also kills the server on its own clean exit, but that path does not
    run when the app is force-quit or crashes. An orphaned server keeps port
    8765 and answers `/health` with 200 while resetting every WebSocket the
    next app opens, which looks like a broken app rather than a stale server.
    Watching the parent covers every way it can die.
    """
    while True:
        time.sleep(2)
        try:
            os.kill(parent_pid, 0)
        except ProcessLookupError:
            logger.info("Parent process %d is gone; shutting down", parent_pid)
            # SIGTERM rather than a hard exit so uvicorn runs its shutdown and
            # the TTS worker process is cleaned up instead of being leaked.
            os.kill(os.getpid(), signal.SIGTERM)
            return
        except PermissionError:
            # Still alive, just not ours to signal.
            continue


def main() -> None:
    _reclaim_port(PORT)
    parent_pid = os.environ.get("VOXTRANS_PARENT_PID")
    if parent_pid and parent_pid.isdigit():
        threading.Thread(
            target=_watch_parent, args=(int(parent_pid),), daemon=True
        ).start()
    uvicorn.run(
        "src.main:app",
        host=HOST,
        port=PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
