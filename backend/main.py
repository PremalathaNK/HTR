"""
ScriptBridge — Main FastAPI Backend v3.0
Multilingual Handwritten Note Digitization, Translation & Speech Accessibility

Endpoints:
  GET  /                          Health check
  POST /api/extract-text          OCR with pre-processing + confidence scores
  POST /api/translate-and-tts     Translate + TTS with word timestamps
  POST /api/feedback/correct      Save user word corrections (feeds fine-tuning)
  GET  /api/feedback/stats        Dataset stats for the correction store
  GET  /api/feedback/export       Download all corrections as JSON (for fine-tuning)
  GET  /api/languages             Supported languages list
  GET  /api/history               Recent OCR sessions history
  GET  /api/dashboard             Combined dashboard stats
  DELETE /api/history/{image_id}  Remove a history entry
"""

import io
import uuid
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

# ─── Service Imports ──────────────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(__file__))

from services.preprocessor import bytes_to_pil, preprocess_image
from services.ocr_service import run_ocr, ModelChoice
from services.translate_service import translate_text, get_supported_languages
from services.tts_service import synthesize_speech
from services.feedback_service import (
    save_correction, get_stats, load_all_corrections,
    compute_image_hash, lookup_correction_for_image,
)

# ─── Session History Storage ──────────────────────────────────────────────────
HISTORY_DIR = Path(__file__).parent / "data" / "history"
HISTORY_FILE = HISTORY_DIR / "sessions.jsonl"
MAX_HISTORY = 50   # Keep last 50 sessions


def _ensure_history_dir():
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def _save_session(session: dict):
    _ensure_history_dir()
    sessions = _load_sessions()
    # Dedup by image_id
    sessions = [s for s in sessions if s.get("image_id") != session.get("image_id")]
    sessions.insert(0, session)
    sessions = sessions[:MAX_HISTORY]
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        for s in sessions:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def _load_sessions() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    sessions = []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    sessions.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return sessions


def _delete_session(image_id: str) -> bool:
    sessions = _load_sessions()
    new_sessions = [s for s in sessions if s.get("image_id") != image_id]
    if len(new_sessions) == len(sessions):
        return False
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        for s in new_sessions:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    return True


# ─── App Init ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ScriptBridge API",
    description="AI Multilingual Handwritten Note Digitization, Translation & Speech",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Request / Response Models ────────────────────────────────────────────────

class WordCorrection(BaseModel):
    index: int
    original: str
    corrected: str


class CorrectionRequest(BaseModel):
    image_id: str
    image_hash: str = ""
    original_ocr: str
    corrected_text: str
    word_corrections: list[WordCorrection]
    model_used: str = "unknown"
    sequence_confidence: float = 0.0


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    return {
        "status": "running",
        "app": "ScriptBridge API v3.0",
        "message": "AI Multilingual HTR + Translation + TTS System",
        "version": "3.0.0",
    }


@app.get("/api/languages")
def get_languages():
    """Return list of supported languages for the frontend selector."""
    return get_supported_languages()


@app.post("/api/extract-text")
async def extract_text(
    file: UploadFile = File(...),
    model_choice: str = Form("easyocr"),
    preprocess: bool = Form(True),
):
    """
    Upload an image -> pre-process -> run OCR -> return text + confidence scores.
    Also stores session in history.
    """
    try:
        contents = await file.read()
        image_id = str(uuid.uuid4())

        image_hash = compute_image_hash(contents)

        # If this exact image was already corrected by the user, return that
        prior_correction = lookup_correction_for_image(image_hash)
        if prior_correction is not None:
            corrected_words = [
                {"word": w, "confidence": 1.0, "flagged": False}
                for w in prior_correction.split()
            ]
            session = {
                "image_id": image_id,
                "image_hash": image_hash,
                "extracted_text": prior_correction,
                "engine": "correction_cache",
                "sequence_confidence": 1.0,
                "word_count": len(corrected_words),
                "from_correction": True,
                "filename": file.filename or "image",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "preprocessing": {},
            }
            _save_session(session)
            return {**session, "words": corrected_words}

        # Load image
        pil_image = bytes_to_pil(contents)

        valid_choices = ["easyocr", "trocr-handwritten", "auto"]
        if model_choice not in valid_choices:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model_choice. Must be one of: {valid_choices}"
            )

        async def run_pipeline(m_choice: str):
            img = pil_image
            p_meta = {}
            if preprocess:
                pre_mode = "trocr" if m_choice == "trocr-handwritten" else "easyocr"
                pre_result = preprocess_image(img, mode=pre_mode)
                img = pre_result["processed_image"]
                p_meta = {
                    "quality_score": pre_result["quality_score"],
                    "is_blurry": pre_result["is_blurry"],
                    "skew_corrected": pre_result["skew_corrected"],
                    "mode": pre_result["mode"],
                }
            result = await run_in_threadpool(run_ocr, img, model_choice=m_choice)
            return result, p_meta

        if model_choice == "auto":
            res_easy, meta_easy = await run_pipeline("easyocr")
            res_trocr, meta_trocr = await run_pipeline("trocr-handwritten")
            if res_trocr["sequence_confidence"] > res_easy["sequence_confidence"]:
                ocr_result = res_trocr
                preprocessing_meta = meta_trocr
            else:
                ocr_result = res_easy
                preprocessing_meta = meta_easy
        else:
            ocr_result, preprocessing_meta = await run_pipeline(model_choice)

        session = {
            "image_id": image_id,
            "image_hash": image_hash,
            "extracted_text": ocr_result["full_text"],
            "engine": ocr_result["engine"],
            "sequence_confidence": ocr_result["sequence_confidence"],
            "word_count": len(ocr_result["words"]),
            "from_correction": False,
            "filename": file.filename or "image",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "preprocessing": preprocessing_meta,
        }
        _save_session(session)

        return {
            **session,
            "words": ocr_result["words"],
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] extract_text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/translate-and-tts")
async def translate_and_tts(
    text: str = Form(...),
    target_language: str = Form("en"),
    source_language: str = Form("auto"),
):
    """
    Translate text to target language and generate TTS audio with word timestamps.
    """
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Empty text provided.")

    try:
        trans_result = await run_in_threadpool(
            translate_text,
            text=text,
            target_lang=target_language,
            source_lang=source_language,
        )
        translated = trans_result["translated_text"]

        tts_result = await run_in_threadpool(
            synthesize_speech, text=translated, lang=target_language
        )

        return {
            "translated_text": translated,
            "source_lang": source_language,
            "target_lang": target_language,
            "target_lang_name": trans_result["target_lang_name"],
            "audio_base64": tts_result["audio_base64"],
            "word_timestamps": tts_result["word_timestamps"],
            "duration_estimate_ms": tts_result["duration_estimate_ms"],
            "char_count": len(translated),
            "word_count": len(translated.split()),
        }

    except Exception as e:
        print(f"[ERROR] translate_and_tts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/feedback/correct")
async def submit_correction(req: CorrectionRequest):
    """
    Save user word corrections. This data feeds the TrOCR fine-tuning pipeline.
    """
    try:
        result = save_correction(
            image_id=req.image_id,
            original_ocr=req.original_ocr,
            corrected_text=req.corrected_text,
            word_corrections=[wc.model_dump() for wc in req.word_corrections],
            model_used=req.model_used,
            sequence_confidence=req.sequence_confidence,
            image_hash=req.image_hash,
        )
        return result
    except Exception as e:
        print(f"[ERROR] submit_correction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/feedback/stats")
def feedback_stats():
    """Return stats about the correction dataset size."""
    return get_stats()


@app.get("/api/feedback/export")
def export_feedback():
    """Export all corrections as a JSON array for fine-tuning."""
    try:
        records = load_all_corrections()
        return {
            "total": len(records),
            "records": records,
            "usage": (
                "Use these records with: "
                "python fine_tuning_and_evaluation.py --mode finetune --source feedback"
            ),
        }
    except Exception as e:
        print(f"[ERROR] export_feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history")
def get_history(limit: int = 20):
    """Return recent OCR session history (last N sessions)."""
    try:
        sessions = _load_sessions()
        return {
            "total": len(sessions),
            "sessions": sessions[:limit],
        }
    except Exception as e:
        print(f"[ERROR] get_history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/history/{image_id}")
def delete_history_entry(image_id: str):
    """Remove a specific session from history."""
    deleted = _delete_session(image_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "image_id": image_id}


@app.get("/api/dashboard")
def get_dashboard():
    """
    Combined dashboard stats: feedback stats + history overview + engine usage breakdown.
    """
    try:
        fb_stats = get_stats()
        sessions = _load_sessions()

        # Engine usage breakdown
        engine_usage: dict[str, int] = {}
        total_words = 0
        avg_confidences = []
        from_cache = 0
        for s in sessions:
            eng = s.get("engine", "unknown")
            engine_usage[eng] = engine_usage.get(eng, 0) + 1
            total_words += s.get("word_count", 0)
            conf = s.get("sequence_confidence", 0)
            if conf > 0:
                avg_confidences.append(conf)
            if s.get("from_correction"):
                from_cache += 1

        avg_conf = round(sum(avg_confidences) / len(avg_confidences), 4) if avg_confidences else 0.0

        return {
            "feedback": fb_stats,
            "sessions": {
                "total": len(sessions),
                "total_words_extracted": total_words,
                "from_cache": from_cache,
                "avg_confidence": avg_conf,
                "engine_breakdown": engine_usage,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        print(f"[ERROR] get_dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
