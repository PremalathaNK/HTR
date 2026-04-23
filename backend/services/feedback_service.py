"""
Feedback & Correction Service — The fine-tuning data flywheel.

When users correct uncertain OCR words, those corrections are saved as JSONL.
This JSONL later feeds directly into the TrOCR fine-tuning pipeline.

Key behaviour:
- Each image gets a stable SHA-256 hash (image_hash) so the same image can be
  looked up across sessions, even if the user uploads it again.
- lookup_correction_for_image(image_hash) returns the last corrected_text if a
  human has previously corrected this exact image, so OCR is NOT re-run.
- Duplicate feedback for the same image_hash is deduplicated (new corrections
  update the existing record instead of appending identical rows).

JSONL format per line:
{
  "image_id": "uuid-string",
  "image_hash": "sha256-hex",          <- stable ID across re-uploads
  "original_ocr": "Thc quick brwn fox",
  "corrected_text": "The quick brown fox",
  "word_corrections": [
    {"index": 0, "original": "Thc", "corrected": "The"},
    {"index": 2, "original": "brwn", "corrected": "brown"}
  ],
  "model_used": "trocr-handwritten",
  "sequence_confidence": 0.61,
  "timestamp": "2024-04-20T10:30:00Z"
}
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Feedback storage location
FEEDBACK_DIR = Path(__file__).parent.parent / "data" / "feedback"
FEEDBACK_FILE = FEEDBACK_DIR / "corrections.jsonl"
STATS_FILE = FEEDBACK_DIR / "stats.json"
# In-memory index: image_hash -> corrected_text  (rebuilt on first load)
_HASH_INDEX: dict[str, str] = {}


def _ensure_dirs():
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)


# ── Image hashing ─────────────────────────────────────────────────────────────

def compute_image_hash(image_bytes: bytes) -> str:
    """Compute a SHA-256 hex digest of raw image bytes.
    Stable across re-uploads of the exact same file."""
    return hashlib.sha256(image_bytes).hexdigest()


# ── Hash index (for same-image lookup) ────────────────────────────────────────

def _build_hash_index() -> None:
    """Rebuild the in-memory hash->corrected_text index from the JSONL file."""
    global _HASH_INDEX
    _HASH_INDEX = {}
    if not FEEDBACK_FILE.exists():
        return
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                h = rec.get("image_hash")
                if h:
                    _HASH_INDEX[h] = rec.get("corrected_text", "")
            except json.JSONDecodeError:
                pass


def lookup_correction_for_image(image_hash: str) -> Optional[str]:
    """
    Return the previously corrected text for this image hash, or None if
    no correction has been saved yet.

    Call this BEFORE running OCR so that re-uploading the same image returns
    the human-corrected version instead of re-running (possibly wrong) OCR.
    """
    if not _HASH_INDEX:
        _build_hash_index()
    return _HASH_INDEX.get(image_hash)


def save_correction(
    image_id: str,
    original_ocr: str,
    corrected_text: str,
    word_corrections: list[dict],
    model_used: str = "unknown",
    sequence_confidence: float = 0.0,
    image_hash: str = "",
) -> dict:
    """
    Save a user correction to the JSONL feedback file.

    If image_hash is provided and this image was already corrected, the
    existing record is UPDATED (not duplicated) so the latest correction
    wins and the hash index stays clean.

    Args:
        image_id: Unique ID of the source image (returned by /api/extract-text)
        original_ocr: Raw OCR output before correction
        corrected_text: Full corrected text from user
        word_corrections: List of {index, original, corrected} word diffs
        model_used: Which OCR model was used
        sequence_confidence: Model's confidence for this result
        image_hash: SHA-256 of the raw image bytes (from compute_image_hash)

    Returns:
        dict confirming save with total correction count
    """
    _ensure_dirs()

    record = {
        "image_id": image_id,
        "image_hash": image_hash,
        "original_ocr": original_ocr,
        "corrected_text": corrected_text,
        "word_corrections": word_corrections,
        "model_used": model_used,
        "sequence_confidence": sequence_confidence,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # ── Deduplication: if hash already exists, rewrite file with updated record ──
    is_update = False
    if image_hash and FEEDBACK_FILE.exists():
        existing = load_all_corrections()
        for i, rec in enumerate(existing):
            if rec.get("image_hash") == image_hash:
                existing[i] = record  # overwrite with latest correction
                is_update = True
                break
        if is_update:
            with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
                for rec in existing:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if not is_update:
        # Append new record
        with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ── Update in-memory hash index immediately ──
    if image_hash:
        _HASH_INDEX[image_hash] = corrected_text

    # Update stats
    stats = _load_stats()
    if not is_update:
        stats["total_corrections"] += 1
        stats["total_words_corrected"] += len(word_corrections)
    else:
        # Update word count delta only
        stats["total_words_corrected"] += len(word_corrections)
    stats["last_updated"] = record["timestamp"]
    _save_stats(stats)

    return {
        "status": "updated" if is_update else "saved",
        "image_id": image_id,
        "image_hash": image_hash,
        "words_corrected": len(word_corrections),
        "total_corrections_in_dataset": stats["total_corrections"],
    }


def get_stats() -> dict:
    """Return aggregate stats about the feedback dataset."""
    _ensure_dirs()
    stats = _load_stats()
    stats["feedback_file"] = str(FEEDBACK_FILE)
    stats["file_exists"] = FEEDBACK_FILE.exists()
    if FEEDBACK_FILE.exists():
        stats["file_size_kb"] = round(FEEDBACK_FILE.stat().st_size / 1024, 2)
    return stats


def load_all_corrections() -> list[dict]:
    """Load all saved corrections (for fine-tuning pipeline use)."""
    if not FEEDBACK_FILE.exists():
        return []
    records = []
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _load_stats() -> dict:
    if STATS_FILE.exists():
        try:
            with open(STATS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"total_corrections": 0, "total_words_corrected": 0, "last_updated": None}


def _save_stats(stats: dict):
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
