"""
TrOCR Fine-Tuning & Evaluation Pipeline — ScriptBridge

This script has two modes:
1. EVALUATE — Compare TrOCR vs EasyOCR on a test set using CER + WER
2. FINE-TUNE — Fine-tune TrOCR on your corrected feedback dataset

The feedback data is loaded directly from backend/data/feedback/corrections.jsonl
(saved by the /api/feedback/correct endpoint when users correct OCR errors).

## Quick Start

### Evaluate models:
    python fine_tuning_and_evaluation.py --mode evaluate --test-dir ./test_images

### Fine-tune from feedback data:
    python fine_tuning_and_evaluation.py --mode finetune --epochs 5

### Fine-tune on IAM or custom dataset:
    python fine_tuning_and_evaluation.py --mode finetune --dataset-dir ./my_dataset --epochs 10

## Requirements (run on GPU — Google Colab recommended for free GPU):
    pip install transformers>=4.41.0 torch datasets jiwer pillow tqdm evaluate accelerate

## Anti-Overfitting Strategy (important!):
    - weight_decay: L2 regularisation on all non-bias parameters (default 0.01)
    - label_smoothing_factor: softens hard targets so the model doesn't memorise
    - warmup_ratio: gradual LR warm-up avoids instability at start
    - early_stopping_patience: stops training once val CER stops improving
    - Data augmentation: random rotation + brightness jitter during training
    - load_best_model_at_end=True: always saves the best checkpoint, not the last

## Changelog:
    v2.2 — Added anti-overfitting controls:
        - weight_decay, label_smoothing_factor, warmup_ratio, augmentation,
          EarlyStoppingCallback, gradient_accumulation_steps
    v2.1 — Fixed deprecated transformers APIs:
        - Removed as_target_tokenizer() (removed in transformers>=4.40)
        - Changed evaluation_strategy -> eval_strategy (transformers>=4.41)
        - Changed tokenizer= -> processing_class= in Seq2SeqTrainer
"""

import os
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
BASE_DIR = Path(__file__).parent
FEEDBACK_FILE = BASE_DIR / "data" / "feedback" / "corrections.jsonl"
MODEL_OUTPUT_DIR = BASE_DIR / "models" / "trocr-finetuned"

# ─── Device ───────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_models(test_image_paths: list[str], ground_truths: list[str]):
    """
    Compare TrOCR (handwritten) vs EasyOCR on a test set.
    Prints CER and WER for each model.

    Args:
        test_image_paths: List of paths to test images
        ground_truths: Corresponding ground-truth text strings
    """
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import easyocr

    assert len(test_image_paths) == len(ground_truths), \
        "Mismatch: number of images must equal number of ground truths"

    print(f"\n[EVAL] Evaluating on {len(test_image_paths)} samples...")

    # Load models
    print("[EVAL] Loading TrOCR handwritten...")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-base-handwritten"
    ).to(device)
    model.eval()

    print("[EVAL] Loading EasyOCR...")
    reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

    trocr_preds, easyocr_preds = [], []

    for img_path in tqdm(test_image_paths, desc="Running inference"):
        image = Image.open(img_path).convert("RGB")

        # TrOCR
        pv = processor(images=image, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            ids = model.generate(pv, max_new_tokens=128)
        trocr_text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        trocr_preds.append(trocr_text)

        # EasyOCR
        results = reader.readtext(img_path, detail=0)
        easyocr_text = " ".join(results).strip()
        easyocr_preds.append(easyocr_text)

    # Compute metrics
    cer_trocr = jiwer.cer(ground_truths, trocr_preds)
    wer_trocr = jiwer.wer(ground_truths, trocr_preds)
    cer_easy = jiwer.cer(ground_truths, easyocr_preds)
    wer_easy = jiwer.wer(ground_truths, easyocr_preds)

    print("\n" + "="*50)
    print("       MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"{'Model':<25} {'CER':>8} {'WER':>8}")
    print("-"*50)
    print(f"{'TrOCR (handwritten)':<25} {cer_trocr:>7.2%} {wer_trocr:>7.2%}")
    print(f"{'EasyOCR':<25} {cer_easy:>7.2%} {wer_easy:>7.2%}")
    print("="*50)

    return {
        "trocr": {"cer": cer_trocr, "wer": wer_trocr, "predictions": trocr_preds},
        "easyocr": {"cer": cer_easy, "wer": wer_easy, "predictions": easyocr_preds},
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATASET PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

class HandwritingDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset for TrOCR fine-tuning.
    Each item = (image, ground_truth_text).

    Supports two source formats:
    1. corrections.jsonl (from user feedback loop) — image_id lookup via image_dir
    2. Plain directory with paired image + .txt files:
       image_001.jpg + image_001.txt

    Augmentation (training only):
    - Random slight rotation (±2°) to simulate real handwriting tilt variation
    - Random brightness jitter (±15%) to simulate different scan lighting
    - These prevent the model from memorising exact pixel patterns.
    """

    def __init__(self, samples: list[dict], processor, image_dir: str = None,
                 max_length: int = 128, augment: bool = False):
        self.samples = samples
        self.processor = processor
        self.image_dir = Path(image_dir) if image_dir else None
        self.max_length = max_length
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample["corrected_text"]

        # Try to load image by image_id from image_dir
        image = None
        if self.image_dir:
            for ext in [".jpg", ".png", ".jpeg"]:
                img_path = self.image_dir / (sample["image_id"] + ext)
                if img_path.exists():
                    image = Image.open(img_path).convert("RGB")
                    break

            # Also try loading by explicit image_path key (from directory loader)
            if image is None and "image_path" in sample:
                try:
                    image = Image.open(sample["image_path"]).convert("RGB")
                except Exception:
                    pass

        if image is None:
            # Fallback: create a dummy white image (for testing without images)
            image = Image.new("RGB", (384, 64), color=(255, 255, 255))

        # ── Augmentation (training only) ──
        if self.augment:
            image = _augment_image(image)

        # Encode image
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

        # Encode text labels
        # NOTE: as_target_tokenizer() was removed in transformers>=4.40 — use tokenizer directly
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        ).input_ids

        # Replace padding token ID with -100 so it's ignored in loss
        labels = [
            label if label != self.processor.tokenizer.pad_token_id else -100
            for label in labels
        ]

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def load_feedback_samples() -> list[dict]:
    """Load correction samples from the feedback JSONL file."""
    if not FEEDBACK_FILE.exists():
        print(f"[WARN] No feedback file found at {FEEDBACK_FILE}")
        return []
    samples = []
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    print(f"[DATA] Loaded {len(samples)} samples from feedback store.")
    return samples


def load_directory_samples(dataset_dir: str) -> list[dict]:
    """
    Load samples from a directory of paired image+txt files.
    Expects: image_001.jpg + image_001.txt, image_002.jpg + image_002.txt, etc.
    """
    dataset_path = Path(dataset_dir)
    samples = []
    image_files = sorted(dataset_path.glob("*.jpg")) + sorted(dataset_path.glob("*.png"))
    for img_path in image_files:
        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists():
            text = txt_path.read_text(encoding="utf-8").strip()
            samples.append({
                "image_id": img_path.stem,
                "corrected_text": text,
                "image_path": str(img_path),
            })
    print(f"[DATA] Loaded {len(samples)} samples from {dataset_dir}.")
    return samples


# ── Data Augmentation ───────────────────────────────────────────────────────────────

def _augment_image(image: Image.Image) -> Image.Image:
    """
    Light augmentation to prevent overfitting to training pixel patterns.
    Applied ONLY on training samples, never on validation.

    Techniques:
    - Random slight rotation (±2°) — simulate real handwriting tilt
    - Random brightness jitter (±15%) — simulate different scan exposure
    - Random mild blur (50% probability) — simulate different scan quality
    """
    # Rotation
    angle = random.uniform(-2.0, 2.0)
    image = image.rotate(angle, expand=False, fillcolor=(255, 255, 255))

    # Brightness jitter
    factor = random.uniform(0.85, 1.15)
    image = ImageEnhance.Brightness(image).enhance(factor)

    # Mild random blur
    if random.random() < 0.5:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 0.7)))

    return image


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FINE-TUNING
# ══════════════════════════════════════════════════════════════════════════════

def compute_cer_metric(pred):
    """Compute CER for use with Hugging Face Seq2SeqTrainer."""
    from evaluate import load as load_metric
    cer_metric = load_metric("cer")

    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # TrOCRProcessor needed here — load globally in fine_tune_trocr
    _proc = getattr(compute_cer_metric, "_processor", None)
    if _proc is None:
        return {"cer": 1.0}

    pred_str = _proc.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = _proc.tokenizer.pad_token_id
    label_str = _proc.batch_decode(label_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}


def fine_tune_trocr(
    samples: list[dict],
    image_dir: str = None,
    base_model: str = "microsoft/trocr-base-handwritten",
    num_epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    val_split: float = 0.15,       # slightly larger val split for better monitoring
    output_dir: str = None,
    weight_decay: float = 0.01,    # L2 regularisation — KEY anti-overfitting knob
    label_smoothing: float = 0.1,  # softens targets — prevents memorisation
    early_stopping_patience: int = 3,  # stop if val CER doesn't improve for N epochs
    grad_accum_steps: int = 2,     # accumulate gradients — effective batch = batch_size*2
):
    """
    Fine-tune TrOCR on corrected samples.

    Anti-overfitting controls:
        weight_decay:             L2 penalty on model weights (default 0.01)
        label_smoothing:          Smooths target distribution (default 0.1)
        early_stopping_patience:  Stops training if val CER stagnates
        grad_accum_steps:         Larger effective batch = smoother gradients
        augmentation:             Random rotation + brightness applied to training images
        load_best_model_at_end:   Always loads the best val-CER checkpoint

    Anti-underfitting controls:
        warmup_ratio:             LR warm-up prevents premature convergence
        learning_rate:            Use 5e-5 for small datasets, 2e-5 for large
        num_epochs:               10+ for large datasets, 5 for small feedback sets

    Args:
        samples: List of {image_id, corrected_text} dicts
        image_dir: Directory where images are stored (matched by image_id)
        base_model: HuggingFace model ID or path to a previously fine-tuned checkpoint
        num_epochs: Number of training epochs
        batch_size: Per-device batch size (reduce to 2 if OOM on GPU)
        learning_rate: Optimizer learning rate
        val_split: Fraction of samples to use for validation
        output_dir: Where to save the fine-tuned model
        weight_decay: L2 regularisation strength
        label_smoothing: Label smoothing factor [0, 0.3]
        early_stopping_patience: Epochs without val-CER improvement before stopping
        grad_accum_steps: Gradient accumulation steps
    """
    from transformers import (
        TrOCRProcessor,
        VisionEncoderDecoderModel,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        default_data_collator,
        EarlyStoppingCallback,
    )

    if not samples:
        print("[ERROR] No samples provided. Aborting fine-tuning.")
        return

    if output_dir is None:
        output_dir = str(MODEL_OUTPUT_DIR / datetime.now().strftime("%Y%m%d_%H%M%S"))

    print(f"\n[FINETUNE] Starting TrOCR fine-tuning")
    print(f"  Base model        : {base_model}")
    print(f"  Samples           : {len(samples)}")
    print(f"  Epochs            : {num_epochs}")
    print(f"  LR                : {learning_rate}")
    print(f"  Weight decay      : {weight_decay}")
    print(f"  Label smoothing   : {label_smoothing}")
    print(f"  Early stop after  : {early_stopping_patience} non-improving epochs")
    print(f"  Grad accum steps  : {grad_accum_steps}")
    print(f"  Output dir        : {output_dir}")
    print(f"  Device            : {device}\n")

    # ── Load model & processor ──
    processor = TrOCRProcessor.from_pretrained(base_model)
    model = VisionEncoderDecoderModel.from_pretrained(base_model).to(device)

    # Required config for Seq2Seq generation
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 128
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    # Attach processor to metric fn for decoding inside trainer
    compute_cer_metric._processor = processor

    # ── Train/Val split — shuffle first for fairness ──
    random.shuffle(samples)
    val_count = max(1, int(len(samples) * val_split))
    train_samples = samples[val_count:]
    val_samples = samples[:val_count]

    # Training set gets augmentation; val set does NOT (clean evaluation)
    train_dataset = HandwritingDataset(
        train_samples, processor, image_dir=image_dir, augment=True
    )
    val_dataset = HandwritingDataset(
        val_samples, processor, image_dir=image_dir, augment=False
    )

    print(f"[FINETUNE] Train: {len(train_dataset)} | Val: {len(val_dataset)} samples")

    # ── Compute warmup steps from ratio ──
    steps_per_epoch = max(1, len(train_dataset) // (batch_size * grad_accum_steps))
    warmup_steps = max(50, int(steps_per_epoch * num_epochs * 0.06))  # ~6% of total steps

    # ── Training arguments with anti-overfitting controls ──
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,         # saves best val-CER model
        metric_for_best_model="cer",
        greater_is_better=False,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,  # effective batch *= accum
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,           # gradual LR ramp-up
        weight_decay=weight_decay,           # L2 regularisation
        label_smoothing_factor=label_smoothing,  # prevent memorisation
        fp16=torch.cuda.is_available(),      # mixed precision only on GPU
        logging_steps=10,
        output_dir=output_dir,
        report_to="none",
        dataloader_num_workers=0,            # Keep 0 on Windows
        save_total_limit=3,                  # keep only last 3 checkpoints
    )

    trainer = Seq2SeqTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        compute_metrics=compute_cer_metric,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
        ],
    )

    # ── Train ──
    print("[FINETUNE] Beginning training...")
    trainer.train()

    # ── Save ──
    final_path = os.path.join(output_dir, "final")
    trainer.save_model(final_path)
    processor.save_pretrained(final_path)
    print(f"\n[OK] Fine-tuned model saved to: {final_path}")
    print(f"   To use: set model path to '{final_path}' in ocr_service.py")

    return final_path


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CLI ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ScriptBridge — TrOCR Fine-Tuning & Evaluation")
    parser.add_argument("--mode", choices=["evaluate", "finetune"], required=True,
                        help="'evaluate' to benchmark models, 'finetune' to train")

    # Shared
    parser.add_argument("--image-dir", type=str, default=None,
                        help="Directory with image files (for paired dataset or feedback images)")

    # Evaluate options
    parser.add_argument("--test-dir", type=str, default=None,
                        help="[evaluate] Directory with test images + .txt ground truths")

    # Fine-tune options
    parser.add_argument("--source", choices=["feedback", "directory"], default="feedback",
                        help="[finetune] Data source: 'feedback' (JSONL) or 'directory' (image+txt pairs)")
    parser.add_argument("--dataset-dir", type=str, default=None,
                        help="[finetune] Dataset directory (required if --source directory)")
    parser.add_argument("--base-model", type=str, default="microsoft/trocr-base-handwritten",
                        help="[finetune] Base model ID or path to checkpoint")
    parser.add_argument("--epochs", type=int, default=5,
                        help="[finetune] Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="[finetune] Per-device batch size")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="[finetune] Learning rate")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="[finetune] Output directory for saved model")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="[finetune] L2 weight decay (anti-overfitting, default 0.01)")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="[finetune] Label smoothing factor 0-0.3 (anti-overfitting, default 0.1)")
    parser.add_argument("--early-stopping-patience", type=int, default=3,
                        help="[finetune] Stop if val CER doesn't improve for N epochs (default 3)")
    parser.add_argument("--grad-accum", type=int, default=2,
                        help="[finetune] Gradient accumulation steps (default 2)")

    args = parser.parse_args()

    if args.mode == "evaluate":
        if not args.test_dir:
            print("[ERROR] --test-dir is required for evaluate mode.")
            sys.exit(1)
        test_dir = Path(args.test_dir)
        image_paths, ground_truths = [], []
        image_files = sorted(test_dir.glob("*.jpg")) + sorted(test_dir.glob("*.png"))
        for img_path in image_files:
            txt_path = img_path.with_suffix(".txt")
            if txt_path.exists():
                image_paths.append(str(img_path))
                ground_truths.append(txt_path.read_text(encoding="utf-8").strip())
        if not image_paths:
            print(f"[ERROR] No paired image+txt files found in {args.test_dir}")
            sys.exit(1)
        evaluate_models(image_paths, ground_truths)

    elif args.mode == "finetune":
        if args.source == "feedback":
            samples = load_feedback_samples()
            if len(samples) < 5:
                print(f"[WARN] Only {len(samples)} feedback samples. "
                      f"Fine-tuning needs at least 10-20 for meaningful results.")
        else:
            if not args.dataset_dir:
                print("[ERROR] --dataset-dir is required when --source directory")
                sys.exit(1)
            samples = load_directory_samples(args.dataset_dir)

        fine_tune_trocr(
            samples=samples,
            image_dir=args.image_dir or args.dataset_dir,
            base_model=args.base_model,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            output_dir=args.output_dir,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            early_stopping_patience=args.early_stopping_patience,
            grad_accum_steps=args.grad_accum,
        )


if __name__ == "__main__":
    main()
