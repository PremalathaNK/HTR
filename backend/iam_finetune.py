"""
IAM Handwriting Database — Fine-Tuning Pipeline for ScriptBridge
=================================================================

Supports:
  1. TrOCR (handwritten)  — full fine-tuning on IAM sentences/lines
  2. EasyOCR              — evaluation benchmark only (EasyOCR cannot be
                            fine-tuned directly; accuracy is improved via
                            better pre-processing, which this script validates)

IAM Dataset Structure (after download from fki.unibe.ch):
  iam_dataset/
    lines/                  ← line-level PNG images
      a01/
        a01-000u/
          a01-000u-00.png
          ...
    sentences/              ← sentence-level images (alternative)
    ascii/
      lines.txt             ← ground-truth transcripts for lines
      sentences.txt         ← ground-truth transcripts for sentences

HOW TO GET THE IAM DATASET:
  1. Register (free) at: https://fki.unibe.ch/databases/iam-handwriting-database
  2. Download: lines.tgz  +  ascii.tgz
  3. Extract both into:  backend/iam_dataset/
  4. Run this script:
       python iam_finetune.py --mode finetune --epochs 10
  5. After training, update ocr_service.py:
       TROCR_HANDWRITTEN_PATH = "./models/iam-trocr/final"

QUICK START:
  # Fine-tune TrOCR on IAM lines (recommended):
  python iam_finetune.py --mode finetune --epochs 10 --batch-size 4

  # Evaluate both models on IAM test split:
  python iam_finetune.py --mode evaluate

  # Fine-tune from a previous checkpoint:
  python iam_finetune.py --mode finetune --base-model ./models/iam-trocr/final

Requirements:
  pip install transformers>=4.41 torch torchvision datasets evaluate jiwer pillow tqdm accelerate
"""

import os
import re
import sys
import json
import random
import argparse
from pathlib import Path
from datetime import datetime

import torch
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm
import jiwer

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
IAM_DIR      = BASE_DIR / "iam_dataset"
LINES_DIR    = IAM_DIR / "lines"
ASCII_FILE   = IAM_DIR / "ascii" / "lines.txt"
OUTPUT_DIR   = BASE_DIR / "models" / "iam-trocr"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {device}")


# ══════════════════════════════════════════════════════════════════════════════
# 1 ─ IAM PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_iam_lines_txt(ascii_path: Path) -> list[dict]:
    """
    Parse IAM ascii/lines.txt into a list of {image_id, text} dicts.

    IAM lines.txt format (lines starting with # are comments):
      a01-000u-00 ok 154 19 408 746 1661 89 A|MOVE|to|stop|Mr.|Gaitskell
      <id> <seg> <gray> <threshold> <components> <x> <y> <w> <h> <transcript>
    Transcript words are joined with '|' — we replace with spaces.
    Only lines with seg='ok' are included (others are poorly segmented).
    """
    if not ascii_path.exists():
        raise FileNotFoundError(
            f"IAM ascii file not found at {ascii_path}\n"
            "Download lines.tgz + ascii.tgz from https://fki.unibe.ch/databases/iam-handwriting-database"
        )

    samples = []
    with open(ascii_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(" ")
            if len(parts) < 9:
                continue
            line_id = parts[0]          # e.g. a01-000u-00
            seg_ok  = parts[1] == "ok"  # skip badly segmented lines
            if not seg_ok:
                continue
            transcript = " ".join(parts[8:]).replace("|", " ").strip()
            if not transcript:
                continue
            samples.append({"image_id": line_id, "corrected_text": transcript})

    print(f"[IAM] Parsed {len(samples)} valid lines from {ascii_path.name}")
    return samples


def resolve_iam_image(image_id: str, lines_dir: Path) -> Path | None:
    """
    Resolve IAM image_id (e.g. 'a01-000u-00') to its PNG path.
    IAM lines are stored as: lines/<writer>/<form>/<line>.png
      a01-000u-00  →  lines/a01/a01-000u/a01-000u-00.png
    """
    parts = image_id.split("-")          # ['a01', '000u', '00']
    if len(parts) < 3:
        return None
    writer = parts[0]                    # a01
    form   = f"{parts[0]}-{parts[1]}"   # a01-000u
    fname  = f"{image_id}.png"
    p = lines_dir / writer / form / fname
    return p if p.exists() else None


# ══════════════════════════════════════════════════════════════════════════════
# 2 ─ IAM SPLIT (official train / val1 / val2 / test)
# ══════════════════════════════════════════════════════════════════════════════

# Official IAM writer-based split (writer IDs)
_TRAIN_WRITERS = {
    "a01","a02","a03","a04","a05","a06","a07","a08","a09","a10",
    "a11","a12","a13","a14","a15","a16","a17","a18","a19","a20",
    "b01","b02","b03","b04","b05","b06","c01","c02","c03","c04",
    "c05","c06","d01","d02","d03","d04","d05","d06","d07","d08",
    "e01","e02","e03","e04","f01","f02","f03","f04","g01","g02",
    "g03","g04","g05","g06","g07","h01","h02","h03","h04","h05",
    "h06","h07","h08","h09","h10","h11","h12","h13","h14","h15",
    "h16","h17","h18","h19","h20","h21","h22","h23","h24","h25",
}
_TEST_WRITERS = {"d06","d07","d08","e01","e02","e03","e04","f01","f02","f03","f04"}
_VAL_WRITERS  = {"a28","a29","a30","a31","a32","a33","a34","b13","b14","b15","b16"}


def split_iam_samples(samples: list[dict]) -> tuple[list, list, list]:
    """
    Split IAM samples into train / val / test sets using official writer splits.
    Falls back to 80/10/10 random split if writer IDs not recognized.
    """
    train, val, test = [], [], []
    unmatched = []

    for s in samples:
        writer = s["image_id"].split("-")[0]
        if writer in _TEST_WRITERS:
            test.append(s)
        elif writer in _VAL_WRITERS:
            val.append(s)
        elif writer in _TRAIN_WRITERS:
            train.append(s)
        else:
            unmatched.append(s)

    # For unmatched writers, do a random 80/10/10 split
    if unmatched:
        random.shuffle(unmatched)
        n = len(unmatched)
        t80, t10 = int(n * 0.8), int(n * 0.1)
        train += unmatched[:t80]
        val   += unmatched[t80:t80 + t10]
        test  += unmatched[t80 + t10:]

    print(f"[SPLIT] Train={len(train)} | Val={len(val)} | Test={len(test)}")
    return train, val, test


# ══════════════════════════════════════════════════════════════════════════════
# 3 ─ DATASET CLASS
# ══════════════════════════════════════════════════════════════════════════════

def _augment(img: Image.Image) -> Image.Image:
    """Light augmentation for training: rotation, brightness, mild blur."""
    img = img.rotate(random.uniform(-2.0, 2.0), expand=False, fillcolor=(255, 255, 255))
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.85, 1.15))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.9, 1.1))
    if random.random() < 0.4:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 0.6)))
    return img


class IAMDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for IAM fine-tuning.
    Loads line images + transcripts. Applies augmentation on training set.
    """
    def __init__(self, samples: list[dict], processor, lines_dir: Path,
                 max_length: int = 128, augment: bool = False):
        # Filter out samples whose images don't exist
        valid = []
        for s in samples:
            p = resolve_iam_image(s["image_id"], lines_dir)
            if p is not None:
                valid.append({**s, "_img_path": str(p)})
        print(f"[DATASET] {len(valid)}/{len(samples)} samples have valid images")
        self.samples   = valid
        self.processor = processor
        self.max_len   = max_length
        self.augment   = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        img = Image.open(s["_img_path"]).convert("RGB")
        if self.augment:
            img = _augment(img)

        pixel_values = self.processor(images=img, return_tensors="pt").pixel_values.squeeze(0)

        labels = self.processor.tokenizer(
            s["corrected_text"],
            padding="max_length",
            max_length=self.max_len,
            truncation=True,
        ).input_ids
        labels = [l if l != self.processor.tokenizer.pad_token_id else -100 for l in labels]

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 4 ─ CER METRIC
# ══════════════════════════════════════════════════════════════════════════════

def make_compute_cer(processor):
    from evaluate import load as load_metric
    cer_metric = load_metric("cer")

    def compute_cer(pred):
        pred_ids  = pred.predictions
        label_ids = pred.label_ids
        pred_str  = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        return {"cer": cer_metric.compute(predictions=pred_str, references=label_str)}

    return compute_cer


# ══════════════════════════════════════════════════════════════════════════════
# 5 ─ FINE-TUNE TrOCR ON IAM
# ══════════════════════════════════════════════════════════════════════════════

def fine_tune_trocr_iam(
    train_samples: list[dict],
    val_samples: list[dict],
    lines_dir: Path,
    base_model: str = "microsoft/trocr-base-handwritten",
    num_epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    output_dir: str = None,
    weight_decay: float = 0.01,
    label_smoothing: float = 0.1,
    early_stopping_patience: int = 3,
    grad_accum_steps: int = 2,
):
    from transformers import (
        TrOCRProcessor, VisionEncoderDecoderModel,
        Seq2SeqTrainer, Seq2SeqTrainingArguments,
        default_data_collator, EarlyStoppingCallback,
    )

    if not train_samples:
        print("[ERROR] No training samples. Aborting.")
        return

    if output_dir is None:
        output_dir = str(OUTPUT_DIR / datetime.now().strftime("%Y%m%d_%H%M%S"))

    print(f"\n{'='*60}")
    print(f"  TrOCR IAM Fine-Tuning")
    print(f"{'='*60}")
    print(f"  Base model  : {base_model}")
    print(f"  Train set   : {len(train_samples)} samples")
    print(f"  Val set     : {len(val_samples)} samples")
    print(f"  Epochs      : {num_epochs}")
    print(f"  LR          : {learning_rate}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Label smooth: {label_smoothing}")
    print(f"  Output      : {output_dir}")
    print(f"  Device      : {device}")
    print(f"{'='*60}\n")

    processor = TrOCRProcessor.from_pretrained(base_model)
    model = VisionEncoderDecoderModel.from_pretrained(base_model).to(device)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id           = processor.tokenizer.pad_token_id
    model.config.vocab_size             = model.config.decoder.vocab_size
    model.config.eos_token_id           = processor.tokenizer.sep_token_id
    model.config.max_length             = 128
    model.config.early_stopping         = True
    model.config.no_repeat_ngram_size   = 3
    model.config.length_penalty         = 2.0
    model.config.num_beams              = 4

    train_ds = IAMDataset(train_samples, processor, lines_dir, augment=True)
    val_ds   = IAMDataset(val_samples,   processor, lines_dir, augment=False)

    if len(train_ds) == 0:
        print("[ERROR] No valid training images found. Check your IAM dataset path.")
        return

    steps_per_epoch = max(1, len(train_ds) // (batch_size * grad_accum_steps))
    warmup_steps    = max(50, int(steps_per_epoch * num_epochs * 0.06))

    args = Seq2SeqTrainingArguments(
        predict_with_generate      = True,
        eval_strategy              = "epoch",
        save_strategy              = "epoch",
        load_best_model_at_end     = True,
        metric_for_best_model      = "cer",
        greater_is_better          = False,
        per_device_train_batch_size= batch_size,
        per_device_eval_batch_size = batch_size,
        gradient_accumulation_steps= grad_accum_steps,
        num_train_epochs            = num_epochs,
        learning_rate               = learning_rate,
        warmup_steps                = warmup_steps,
        weight_decay                = weight_decay,
        label_smoothing_factor      = label_smoothing,
        fp16                        = torch.cuda.is_available(),
        logging_steps               = 20,
        output_dir                  = output_dir,
        report_to                   = "none",
        dataloader_num_workers      = 0,
        save_total_limit            = 3,
    )

    trainer = Seq2SeqTrainer(
        model            = model,
        processing_class = processor,
        args             = args,
        compute_metrics  = make_compute_cer(processor),
        train_dataset    = train_ds,
        eval_dataset     = val_ds,
        data_collator    = default_data_collator,
        callbacks        = [EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    print("[FINETUNE] Starting training...")
    trainer.train()

    final_path = os.path.join(output_dir, "final")
    trainer.save_model(final_path)
    processor.save_pretrained(final_path)

    print(f"\n[✓] Fine-tuned model saved to: {final_path}")
    print(f"\n[NEXT STEP] Update ocr_service.py:")
    print(f"   Change:  TROCR_HANDWRITTEN_PATH = \"microsoft/trocr-base-handwritten\"")
    print(f"   To:      TROCR_HANDWRITTEN_PATH = \"{final_path}\"")

    return final_path


# ══════════════════════════════════════════════════════════════════════════════
# 6 ─ EVALUATE BOTH MODELS ON IAM TEST SET
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_on_iam(test_samples: list[dict], lines_dir: Path,
                    trocr_model: str = "microsoft/trocr-base-handwritten",
                    max_samples: int = 500):
    """
    Evaluate TrOCR and EasyOCR on the IAM test set.
    Prints CER and WER for both models.
    max_samples: limit for speed (full IAM test set can take hours on CPU)
    """
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import easyocr
    import numpy as np

    # Resolve and filter valid image paths
    valid = []
    for s in test_samples:
        p = resolve_iam_image(s["image_id"], lines_dir)
        if p is not None:
            valid.append({**s, "_img_path": str(p)})

    if not valid:
        print("[ERROR] No valid test images found.")
        return

    if len(valid) > max_samples:
        random.shuffle(valid)
        valid = valid[:max_samples]
        print(f"[EVAL] Capped at {max_samples} samples for speed")

    print(f"[EVAL] Evaluating on {len(valid)} IAM test samples...")

    # Load TrOCR
    print(f"[EVAL] Loading TrOCR from: {trocr_model}")
    processor = TrOCRProcessor.from_pretrained(trocr_model)
    model = VisionEncoderDecoderModel.from_pretrained(trocr_model).to(device)
    model.eval()

    # Load EasyOCR
    print("[EVAL] Loading EasyOCR...")
    reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

    trocr_preds, easyocr_preds, ground_truths = [], [], []

    for s in tqdm(valid, desc="Running inference"):
        img_path = s["_img_path"]
        gt       = s["corrected_text"]
        ground_truths.append(gt)

        img = Image.open(img_path).convert("RGB")

        # TrOCR inference
        pv = processor(images=img, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            ids = model.generate(pv, max_new_tokens=128)
        trocr_preds.append(processor.batch_decode(ids, skip_special_tokens=True)[0].strip())

        # EasyOCR inference
        results = reader.readtext(img_path, detail=0)
        easyocr_preds.append(" ".join(results).strip())

    # Metrics
    cer_t = jiwer.cer(ground_truths, trocr_preds)
    wer_t = jiwer.wer(ground_truths, trocr_preds)
    cer_e = jiwer.cer(ground_truths, easyocr_preds)
    wer_e = jiwer.wer(ground_truths, easyocr_preds)

    print("\n" + "=" * 55)
    print("   IAM EVALUATION RESULTS")
    print("=" * 55)
    print(f"{'Model':<28} {'CER':>8} {'WER':>8}")
    print("-" * 55)
    print(f"{'TrOCR (handwritten)':<28} {cer_t:>7.2%} {wer_t:>7.2%}")
    print(f"{'EasyOCR':<28} {cer_e:>7.2%} {wer_e:>7.2%}")
    print("=" * 55)

    # Save results
    results_path = BASE_DIR / "iam_eval_results.json"
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "trocr_model": trocr_model,
        "num_samples": len(valid),
        "trocr":   {"cer": round(cer_t, 4), "wer": round(wer_t, 4)},
        "easyocr": {"cer": round(cer_e, 4), "wer": round(wer_e, 4)},
    }
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\n[✓] Results saved to: {results_path}")

    return results_data


# ══════════════════════════════════════════════════════════════════════════════
# 7 ─ CLI ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ScriptBridge — IAM Fine-Tuning & Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Fine-tune TrOCR on IAM (10 epochs):
    python iam_finetune.py --mode finetune --epochs 10

  Evaluate both models on IAM test set:
    python iam_finetune.py --mode evaluate

  Fine-tune from a previous checkpoint:
    python iam_finetune.py --mode finetune --base-model ./models/iam-trocr/final
        """
    )
    parser.add_argument("--mode", choices=["finetune", "evaluate"], required=True)
    parser.add_argument("--iam-dir",   type=str, default=str(IAM_DIR),
                        help="Path to IAM dataset root (default: ./iam_dataset)")
    parser.add_argument("--base-model", type=str,
                        default="microsoft/trocr-base-handwritten",
                        help="HuggingFace model ID or local checkpoint path")
    parser.add_argument("--epochs",      type=int,   default=10)
    parser.add_argument("--batch-size",  type=int,   default=4)
    parser.add_argument("--lr",          type=float, default=5e-5)
    parser.add_argument("--output-dir",  type=str,   default=None)
    parser.add_argument("--weight-decay",type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--grad-accum", type=int,   default=2)
    parser.add_argument("--max-eval-samples", type=int, default=500,
                        help="Max test samples for evaluation (default 500 for speed)")

    args = parser.parse_args()

    iam_path   = Path(args.iam_dir)
    lines_dir  = iam_path / "lines"
    ascii_file = iam_path / "ascii" / "lines.txt"

    # Parse IAM transcripts
    samples = parse_iam_lines_txt(ascii_file)
    if not samples:
        print("[ERROR] No samples parsed. Check your IAM dataset path.")
        sys.exit(1)

    # Split into train/val/test
    train_samples, val_samples, test_samples = split_iam_samples(samples)

    if args.mode == "finetune":
        fine_tune_trocr_iam(
            train_samples      = train_samples,
            val_samples        = val_samples,
            lines_dir          = lines_dir,
            base_model         = args.base_model,
            num_epochs         = args.epochs,
            batch_size         = args.batch_size,
            learning_rate      = args.lr,
            output_dir         = args.output_dir,
            weight_decay       = args.weight_decay,
            label_smoothing    = args.label_smoothing,
            early_stopping_patience = args.early_stop_patience,
            grad_accum_steps   = args.grad_accum,
        )

    elif args.mode == "evaluate":
        evaluate_on_iam(
            test_samples  = test_samples,
            lines_dir     = lines_dir,
            trocr_model   = args.base_model,
            max_samples   = args.max_eval_samples,
        )


if __name__ == "__main__":
    main()
