"""
OCR Service — Smart routing between TrOCR (handwritten) and EasyOCR.

Available models:
  trocr-handwritten — Microsoft TrOCR fine-tuned on handwritten text (IAM)
  easyocr           — EasyOCR for clear/printed/mixed documents

Key features:
- Returns per-word confidence scores
- Flags low-confidence words (< threshold) for user correction
- Lazy-loads models to save startup memory
"""

import easyocr
import torch
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from typing import Literal

# ─── Device Setup ────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Model Registry (lazy-loaded singletons) ─────────────────────────────────
_models: dict = {}

CONFIDENCE_THRESHOLD = 0.75  # Words below this are flagged for correction


def _load_trocr(model_path: str = "microsoft/trocr-base-handwritten"):
    """
    Load TrOCR from a HuggingFace model ID or a local fine-tuned checkpoint.
    Pass a local path (e.g. './models/trocr-finetuned/final') after fine-tuning.
    """
    key = f"trocr-{model_path}"
    if key not in _models:
        print(f"[OCR] Loading TrOCR from: {model_path}...")
        processor = TrOCRProcessor.from_pretrained(model_path)
        model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)
        model.eval()
        _models[key] = {"processor": processor, "model": model}
        print(f"[OCR] TrOCR loaded on {device}.")
    return _models[key]


# Default model path — swap this to your fine-tuned checkpoint after training:
# TROCR_HANDWRITTEN_PATH = "./models/trocr-finetuned/final"
TROCR_HANDWRITTEN_PATH = "microsoft/trocr-base-handwritten"


def _load_easyocr(languages: list[str] = None):
    if languages is None:
        languages = ["en"]
    key = "easyocr-" + "-".join(sorted(languages))
    if key not in _models:
        print(f"[OCR] Loading EasyOCR for langs: {languages}...")
        _models[key] = easyocr.Reader(languages, gpu=torch.cuda.is_available())
        print("[OCR] EasyOCR loaded.")
    return _models[key]


# ─── TrOCR Inference ─────────────────────────────────────────────────────────

def _run_trocr(image: Image.Image, model_path: str = None) -> dict:
    """
    Run TrOCR on a single image.
    Returns the full text string. TrOCR does not provide per-word scores,
    so we assign a uniform confidence derived from beam token probabilities.

    Args:
        image: PIL Image (RGB)
        model_path: HuggingFace model ID or local fine-tuned checkpoint path.
                    Defaults to TROCR_HANDWRITTEN_PATH.
    """
    if model_path is None:
        model_path = TROCR_HANDWRITTEN_PATH
    bundle = _load_trocr(model_path)
    processor = bundle["processor"]
    model = bundle["model"]

    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            output_scores=True,
            return_dict_in_generate=True,
            max_new_tokens=128,
        )

    generated_ids = outputs.sequences
    # Compute sequence-level confidence from token scores
    if outputs.scores:
        token_probs = [
            torch.softmax(score, dim=-1).max(dim=-1).values
            for score in outputs.scores
        ]
        seq_conf = float(torch.stack(token_probs).mean())
    else:
        seq_conf = 1.0

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    # Build per-word entries (confidence distributed equally from sequence score)
    words = text.split()
    word_entries = [
        {
            "word": w,
            "confidence": round(seq_conf, 4),
            "flagged": seq_conf < CONFIDENCE_THRESHOLD,
        }
        for w in words
    ]

    return {
        "full_text": text,
        "words": word_entries,
        "engine": "trocr-handwritten",
        "sequence_confidence": round(seq_conf, 4),
    }


# ─── EasyOCR Inference ───────────────────────────────────────────────────────

def _run_easyocr(image: Image.Image, languages: list[str] = None) -> dict:
    """
    Run EasyOCR. Returns per-detection confidence scores.
    Each detection = one line / text block.
    """
    if languages is None:
        languages = ["en"]
    reader = _load_easyocr(languages)

    img_array = np.array(image)
    # Returns: list of (bbox, text, confidence)
    results = reader.readtext(img_array, detail=1)

    all_words = []
    full_text_parts = []
    for (_bbox, text, conf) in results:
        full_text_parts.append(text)
        for word in text.split():
            all_words.append({
                "word": word,
                "confidence": round(float(conf), 4),
                "flagged": float(conf) < CONFIDENCE_THRESHOLD,
            })

    full_text = " ".join(full_text_parts)
    avg_conf = (
        float(np.mean([e["confidence"] for e in all_words]))
        if all_words else 0.0
    )

    return {
        "full_text": full_text,
        "words": all_words,
        "engine": "easyocr",
        "sequence_confidence": round(avg_conf, 4),
    }


# ─── Public API ──────────────────────────────────────────────────────────────

ModelChoice = Literal["trocr-handwritten", "easyocr"]


def run_ocr(image: Image.Image, model_choice: ModelChoice = "easyocr",
            languages: list[str] = None) -> dict:
    """
    Run OCR using the selected engine.

    Args:
        image: PIL Image (RGB)
        model_choice: "trocr-handwritten" (best for handwriting) or "easyocr" (best for clear/printed)
        languages: Language codes for EasyOCR (ignored for TrOCR)

    Returns:
        dict with keys:
            full_text (str): Extracted text
            words (list): Per-word dicts {word, confidence, flagged}
            engine (str): Which engine ran
            sequence_confidence (float): Overall confidence [0-1]
    """
    if languages is None:
        languages = ["en"]

    if model_choice == "trocr-handwritten":
        return _run_trocr(image)
    else:
        return _run_easyocr(image, languages)


def get_flagged_words(ocr_result: dict) -> list[dict]:
    """Return only the words flagged as uncertain."""
    return [w for w in ocr_result.get("words", []) if w.get("flagged")]
