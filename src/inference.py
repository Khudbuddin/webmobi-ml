"""
inference.py — Whisper Speech Recognition Pipeline
Part 2 of Webmobi ML Internship Assignment

What this script does:
1. Downloads openai/whisper-small from Hugging Face (auto-cached after first run)
2. Downloads LibriSpeech demo dataset from Hugging Face
3. Runs Whisper on 20 audio samples
4. Saves predictions to results/predictions.csv
"""

import os
import csv
import time
import torch
import warnings
warnings.filterwarnings("ignore")

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, Audio

# ── Config ───────────────────────────────────────────────────────────────────

MODEL_NAME   = "openai/whisper-small"
NUM_SAMPLES  = 20
OUTPUT_FILE  = "results/predictions.csv"

# ── Device setup ─────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


def load_model():
    print(f"\nLoading model: {MODEL_NAME}")
    print("(First run downloads ~250MB — subsequent runs load from cache)")

    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()

    print("Model loaded successfully")
    return processor, model


def load_data():
    print(f"\nLoading dataset: hf-internal-testing/librispeech_asr_demo")
    print("(First run downloads dataset — subsequent runs load from cache)")

    dataset = load_dataset(
        "hf-internal-testing/librispeech_asr_demo",
        "clean",
        split="validation",
        trust_remote_code=False
    )

    # Cast audio to 16kHz — this is what Whisper expects
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    print(f"Dataset loaded — {len(dataset)} total samples available")
    print(f"Using first {min(NUM_SAMPLES, len(dataset))} samples")
    return dataset


def run_inference(processor, model, dataset):
    results = []
    print(f"\nRunning inference on {NUM_SAMPLES} samples...\n")

    for i in range(min(NUM_SAMPLES, len(dataset))):
        sample        = dataset[i]
        audio_array   = sample["audio"]["array"]
        sampling_rate = sample["audio"]["sampling_rate"]
        ground_truth  = sample["text"].upper().strip()

        # Convert audio → log-Mel spectrogram
        inputs = processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        )

        input_features = inputs.input_features.to(device)

        # Run inference
        start_time = time.time()
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        latency = round(time.time() - start_time, 3)

        # Decode tokens → text
        prediction = processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0].upper().strip()

        results.append({
            "audio_id"    : f"{i+1:04d}",
            "ground_truth": ground_truth,
            "prediction"  : prediction,
            "latency_sec" : latency
        })

        print(f"  [{i+1:02d}/{NUM_SAMPLES}] ({latency}s)")
        print(f"    GT  : {ground_truth[:80]}")
        print(f"    PRED: {prediction[:80]}")
        print()

    return results


def save_predictions(results):
    os.makedirs("results", exist_ok=True)
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["audio_id", "ground_truth", "prediction", "latency_sec"]
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"Predictions saved to {OUTPUT_FILE}")


def run():
    print("=" * 60)
    print("  WHISPER INFERENCE PIPELINE")
    print(f"  Model   : {MODEL_NAME}")
    print(f"  Dataset : hf-internal-testing/librispeech_asr_demo")
    print(f"  Samples : {NUM_SAMPLES}")
    print("=" * 60)

    processor, model = load_model()
    dataset          = load_data()
    results          = run_inference(processor, model, dataset)
    save_predictions(results)

    print("\nInference complete.")
    print(f"Results saved to: {OUTPUT_FILE}")
    return results


if __name__ == "__main__":
    run()