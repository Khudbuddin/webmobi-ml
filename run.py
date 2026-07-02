"""
run.py — Single entry point for the entire pipeline.
Runs inference then evaluation with one command:

    python run.py

Make sure you have installed dependencies first:
    pip install -r requirements.txt
"""
import faulthandler
faulthandler.enable()
import sys
import os

# Add src/ to path so we can import from it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from inference import run as run_inference
from evaluate import run as run_evaluate

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  WEBMOBI ML ASSIGNMENT — FULL PIPELINE")
    print("  Model  : openai/whisper-small")
    print("  Dataset: librispeech_asr (test-clean)")
    print("=" * 60)

    # Step 1 — Inference
    print("\n── STEP 1: INFERENCE ───────────────────────────────────")
    run_inference()

    # Step 2 — Evaluation
    print("\n── STEP 2: EVALUATION ──────────────────────────────────")
    run_evaluate()

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("  results/predictions.csv  — model predictions")
    print("  results/metrics.json     — WER, CER, latency")
    print("  results/report.md        — human-readable report")
    print("=" * 60)