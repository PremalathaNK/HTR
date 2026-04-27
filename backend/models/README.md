# Models Directory

This directory is structured to store machine learning models used by the ScriptBridge HTR project, particularly fine-tuned checkpoints for handwritten text recognition.

## Architecture & Storage

1. **Base Pre-trained Models:**
   - The base **Microsoft TrOCR** (`microsoft/trocr-base-handwritten`) and **EasyOCR** models are downloaded automatically by their respective libraries on first run.
   - These are typically stored in the system's global cache (e.g., `~/.cache/huggingface/` and `~/.EasyOCR/model/`).

2. **Fine-Tuned Models:**
   - When you run the fine-tuning script (`fine_tuning_and_evaluation.py`) using the human-in-the-loop feedback dataset, the newly trained model checkpoints are saved in this directory.
   - Path structure: `models/trocr-finetuned/<timestamp>/` and `models/trocr-finetuned/final/`

3. **Using Fine-Tuned Models:**
   - Once a model is fine-tuned, you can swap it into the active pipeline by changing the `TROCR_HANDWRITTEN_PATH` variable in `backend/services/ocr_service.py` to point to `./models/trocr-finetuned/final`.
