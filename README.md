# Webmobi ML Internship Assignment
### Reproducible Inference & Evaluation Pipeline — openai/whisper-small

---

## Overview

This project builds a complete speech recognition pipeline using Whisper (openai/whisper-small) from Hugging Face. It downloads the model and dataset automatically, runs inference on 20 audio samples, evaluates accuracy (WER, CER) and speed (latency), and saves all results to the `results/` folder.

The pipeline automatically detects and uses GPU if available, and falls back to CPU if not.

---

## Setup

**Step 1 — Clone the repo**
```bash
git clone https://github.com/yourusername/webmobi-ml
cd webmobi-ml
```

**Step 2 — Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows
```

**Step 3 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 4 — (Optional) Install PyTorch with GPU support**

By default `pip install torch` installs the CPU version. If you have an NVIDIA GPU and want faster inference and better accuracy, install the CUDA version instead:

```bash
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

> Note: CUDA version is ~2.5GB. CPU version is ~300MB. The code works with both — no changes needed.

---

## Run

The entire pipeline runs with one command:

```bash
python run.py
```

This will:
1. Auto-detect GPU or CPU
2. Download `openai/whisper-small` from Hugging Face (~250MB, cached after first run)
3. Download `hf-internal-testing/librispeech_asr_demo` dataset (cached after first run)
4. Run inference on 20 audio samples
5. Compute WER, CER, and latency
6. Save all results to `results/`

---

## Expected Results

Performance varies depending on whether GPU or CPU is used:

### On CPU
| Metric | Value |
|---|---|
| Word Error Rate (WER) | ~18% |
| Character Error Rate (CER) | ~5.7% |
| Average Latency per clip | ~4.0s |
| Total time for 20 clips | ~82s |

### On GPU (NVIDIA)
| Metric | Value |
|---|---|
| Word Error Rate (WER) | ~18% |
| Character Error Rate (CER) | ~1–2% |
| Average Latency per clip | ~0.5s |
| Total time for 20 clips | ~10s |

> WER difference between CPU and GPU is expected — GPU runs the model at full float32 precision faster, which reduces decoding errors. CPU uses a lower precision path which introduces more transcription mistakes.

---

## Output Files

| File | What it contains |
|---|---|
| `results/predictions.csv` | audio_id, ground_truth, prediction, latency per sample |
| `results/metrics.json` | WER, CER, avg/min/max latency, sample count |
| `results/report.md` | Human-readable evaluation report with sample predictions table |

---

## Project Structure

```
webmobi-ml/
├── src/
│   ├── inference.py      # Downloads model + dataset, runs inference, saves predictions.csv
│   └── evaluate.py       # Computes WER/CER/latency, saves metrics.json and report.md
├── results/
│   ├── predictions.csv
│   ├── metrics.json
│   └── report.md
├── run.py                # Single entry point — runs full pipeline
├── requirements.txt
└── README.md
```

---

## Model Choice — Why Whisper?

| | Wav2Vec2 | HuBERT | Whisper (chosen) |
|---|---|---|---|
| Fine-tuning needed? | Yes | Yes | No |
| Multilingual? | No | No | Yes (99 langs) |
| Works out of the box? | No | No | Yes |
| GPU needed? | No | No | No |
| Ease of use | Medium | Medium | High |

Whisper was chosen because it works out-of-the-box without fine-tuning, has the best Hugging Face documentation, and generalises well to real-world audio without any extra setup.

---

## Notes

- First run downloads ~250MB (model) + dataset — subsequent runs use local cache
- GPU is detected and used automatically — no code changes needed
- To change number of samples edit `NUM_SAMPLES` in `src/inference.py` 
- CPU version: Supports Python **3.13** (and earlier compatible versions).
- GPU (CUDA) version: Use **Python 3.11**. Python **3.13** is not currently supported for the CUDA build and may result in installation or runtime issues.
