import asyncio
import logging
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


def main() -> None:
    uvicorn.run(
        "src.main:app",
        host=HOST,
        port=PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
