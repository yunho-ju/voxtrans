#!/usr/bin/env python3
"""Test client: sends a WAV file as 100ms chunks over WebSocket."""

import argparse
import asyncio
import json
import sys
import time

import numpy as np
import soundfile as sf
import websockets


SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1600  # 100ms @ 16kHz
CHUNK_BYTES = CHUNK_SAMPLES * 4  # float32


async def send_wav(uri: str, wav_path: str, language: str = "auto") -> None:
    print(f"Connecting to {uri}...")
    async with websockets.connect(uri) as ws:
        # Read initial status
        status = await ws.recv()
        print(f"Server: {status}")

        # Configure language
        if language != "auto":
            await ws.send(json.dumps({"type": "configure", "language": language}))
            resp = await ws.recv()
            print(f"Server: {resp}")

        # Load and resample audio
        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # mono
        if sr != SAMPLE_RATE:
            # Simple resampling via interpolation
            duration = len(audio) / sr
            n_samples = int(duration * SAMPLE_RATE)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, n_samples),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)

        print(f"Audio: {len(audio)/SAMPLE_RATE:.1f}s, {len(audio)} samples")

        # Send in 100ms chunks with real-time pacing
        total_chunks = len(audio) // CHUNK_SAMPLES
        t0 = time.time()

        # Start a task to receive results concurrently
        results = []
        recv_done = asyncio.Event()

        async def receive_results():
            try:
                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    data = json.loads(msg)
                    if data.get("type") in ("transcription_final", "transcription_partial"):
                        print(f"\n  >> [{data['type']}] {data['text']}")
                        results.append(data)
                    else:
                        print(f"\n  Server: {msg}")
            except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                pass
            recv_done.set()

        recv_task = asyncio.create_task(receive_results())

        for i in range(total_chunks):
            start = i * CHUNK_SAMPLES
            chunk = audio[start : start + CHUNK_SAMPLES]
            await ws.send(chunk.tobytes())

            # Real-time pacing
            elapsed = time.time() - t0
            expected = (i + 1) * 0.1  # 100ms per chunk
            if expected > elapsed:
                await asyncio.sleep(expected - elapsed)

            # Progress
            pct = (i + 1) / total_chunks * 100
            sys.stdout.write(f"\rSending: {pct:.0f}% ({i+1}/{total_chunks} chunks)")
            sys.stdout.flush()

        print("\nDone sending. Waiting for final results...")

        # Wait for remaining results
        await asyncio.sleep(5)
        recv_task.cancel()
        try:
            await recv_task
        except asyncio.CancelledError:
            pass

        print(f"\nTotal transcriptions received: {len(results)}")
        if results:
            full_text = " ".join(r["text"] for r in results)
            print(f"Full text: {full_text}")


def main():
    parser = argparse.ArgumentParser(description="VoxTrans test client")
    parser.add_argument("wav_file", help="Path to WAV file")
    parser.add_argument("--uri", default="ws://localhost:18765/ws", help="WebSocket URI")
    parser.add_argument("--language", default="auto", help="Language (auto, English, Korean, ...)")
    args = parser.parse_args()

    asyncio.run(send_wav(args.uri, args.wav_file, args.language))


if __name__ == "__main__":
    main()
