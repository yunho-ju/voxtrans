import logging

import numpy as np

from .config import CHUNK_BYTES, CHUNK_SAMPLES

logger = logging.getLogger(__name__)

# Maximum buffer size: 20 chunks worth of data (~2s at 100ms chunks)
MAX_BUFFER_BYTES = CHUNK_BYTES * 20


class AudioBuffer:
    """Accumulates raw bytes into complete audio chunks."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    def add(self, data: bytes) -> list[np.ndarray]:
        """Add raw bytes and return list of complete float32 chunks."""
        self._buffer.extend(data)

        # Prevent unbounded buffer growth (e.g. from a misbehaving client)
        if len(self._buffer) > MAX_BUFFER_BYTES:
            logger.warning("AudioBuffer exceeded max size (%d bytes), clearing", len(self._buffer))
            self._buffer.clear()
            return []

        chunks = []
        while len(self._buffer) >= CHUNK_BYTES:
            raw = bytes(self._buffer[:CHUNK_BYTES])
            del self._buffer[:CHUNK_BYTES]
            chunk = np.frombuffer(raw, dtype=np.float32)
            if chunk.shape[0] != CHUNK_SAMPLES:
                raise ValueError(
                    f"Audio chunk has {chunk.shape[0]} samples, expected {CHUNK_SAMPLES}"
                )
            chunks.append(chunk)
        return chunks

    def clear(self) -> None:
        self._buffer.clear()
