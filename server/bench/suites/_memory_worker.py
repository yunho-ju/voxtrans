"""Child process for the `memory` suite (`bench.suites.memory`, Critic
T020 round-1 major 1). Run only via `python -m bench.suites._memory_worker`
-- never imported directly by anything else.

Loads every model fresh in a brand-new process and prints its own and the
TTS worker's `phys_footprint` (harness.read_phys_footprint) as one JSON line
on stdout. Existing as a *separate* process is the whole point: `--suite
all` runs every suite in one long-lived process, and if `memory` measured
itself in that same process, whichever suite ran before it (e.g.
`translation`, which loads its own `mlx_lm` model instance and never
releases it) would still be resident, so the measurement would double-count
weights a clean startup never holds. A fresh child process starts from
nothing every time, independent of suite order or which suites ran first.
"""

from __future__ import annotations

import json
import os
import sys

from .. import harness


def main() -> int:
    from src.asr import ASREngine
    from src.translation import TranslationEngine
    from src.tts import TTSEngine

    asr_engine = ASREngine()
    asr_engine.load()
    translation_engine = TranslationEngine()
    translation_engine.load()
    tts_engine = TTSEngine()
    tts_engine.load()

    try:
        main_footprint = harness.read_phys_footprint(os.getpid())
        tts_footprint = harness.read_phys_footprint(tts_engine._process.pid)
    finally:
        tts_engine.shutdown()

    print(
        json.dumps(
            {
                "main_phys_footprint_bytes": main_footprint,
                "tts_phys_footprint_bytes": tts_footprint,
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
