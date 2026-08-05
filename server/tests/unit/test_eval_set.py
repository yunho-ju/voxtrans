"""Deterministic validation of `specs/001-realtime-latency/eval-set.jsonl`
(data-model.md §5, tasks.md T029/T030).

The eval set is a fixed asset, not a behavior -- these tests are the
"deterministic validation" tasks.md calls for on asset-only work, not a
performance Red. They still guard the invariants FR-011's side-by-side
comparison depends on: exactly 20 entries, unique ids, the 10/10 language
split, and every required field actually populated.
"""

from __future__ import annotations

import json
from pathlib import Path

_EVAL_SET_PATH = (
    Path(__file__).resolve().parents[3] / "specs" / "001-realtime-latency" / "eval-set.jsonl"
)

_REQUIRED_FIELDS = ("id", "source_lang", "target_lang", "text", "note")


def _load_entries() -> list[dict]:
    entries = []
    with _EVAL_SET_PATH.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def test_eval_set_file_exists():
    assert _EVAL_SET_PATH.exists(), f"missing {_EVAL_SET_PATH}"


def test_eval_set_has_exactly_twenty_entries():
    assert len(_load_entries()) == 20


def test_eval_set_ids_are_unique():
    entries = _load_entries()
    ids = [entry["id"] for entry in entries]
    assert len(ids) == len(set(ids)), f"duplicate id(s) found: {ids}"


def test_eval_set_is_split_ten_ko_en_and_ten_en_ko():
    entries = _load_entries()
    ko_to_en = [e for e in entries if e["source_lang"] == "korean" and e["target_lang"] == "english"]
    en_to_ko = [e for e in entries if e["source_lang"] == "english" and e["target_lang"] == "korean"]
    assert len(ko_to_en) == 10, f"expected 10 ko->en pairs, found {len(ko_to_en)}"
    assert len(en_to_ko) == 10, f"expected 10 en->ko pairs, found {len(en_to_ko)}"
    assert len(ko_to_en) + len(en_to_ko) == len(entries), (
        "found a pair whose source_lang/target_lang is neither ko->en nor en->ko"
    )


def test_every_entry_has_every_required_field_non_empty():
    entries = _load_entries()
    for entry in entries:
        for field in _REQUIRED_FIELDS:
            assert field in entry, f"{entry.get('id', '?')} missing field {field!r}"
            value = entry[field]
            assert isinstance(value, str) and value.strip(), (
                f"{entry.get('id', '?')} field {field!r} is empty: {value!r}"
            )


def test_eval_set_covers_exactly_one_technical_term_and_one_self_correction():
    """data-model.md §5 / tasks.md T029: "one technical term, one
    mid-sentence self-correction" -- singular, unlike the other categories
    (conversational turns, numbers/dates, honorifics), which may repeat."""
    entries = _load_entries()
    technical = [e for e in entries if "technical term" in e["note"]]
    self_correction = [e for e in entries if "self-correction" in e["note"]]
    assert len(technical) == 1, f"expected exactly 1 technical-term pair, found {len(technical)}"
    assert len(self_correction) == 1, (
        f"expected exactly 1 self-correction pair, found {len(self_correction)}"
    )
