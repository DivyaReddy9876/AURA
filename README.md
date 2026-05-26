# AURA – Where Voice Meets AI

A Hindi Automatic Speech Recognition (ASR) system built by fine-tuning `openai/whisper-small` using LoRA (PEFT) on the Google FLEURS Hindi dataset.

The project focuses on efficient multilingual speech recognition fine-tuning using limited compute resources (free Google Colab T4 GPU) while achieving significant improvements in transcription accuracy.

---

## Results

| Metric | Base Whisper-Small | Fine-Tuned (LoRA) | Improvement |
|--------|-------------------|-------------------|-------------|
| WER (Word Error Rate) | 68.86% | 35.49% | +33.37% |
| CER (Character Error Rate) | 34.62% | 14.82% | +19.80% |

---

## Project Highlights

- Fine-tuned `openai/whisper-small`
- LoRA-based parameter-efficient training using Hugging Face PEFT
- Trained on Google FLEURS Hindi (`hi_in`)
- Achieved ~50% reduction in word-level transcription errors
- Fully reproducible on free Google Colab T4 GPU
- End-to-end research + implementation + evaluation pipeline

---

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- PEFT (LoRA)
- Datasets
- Evaluate
- Google Colab
- Whisper ASR

---

## Project Links

### Project Documentation
https://docs.google.com/document/d/13tTizTR3AXYDbJRhjD6-vZEK7AB-O3ibEVjhZfejdlw/edit?usp=sharing

### Google Colab Notebook
https://colab.research.google.com/drive/1gVqnSR7ueJUB6LAw0JtDXx4KzN_C50Jd?usp=sharing

### Dataset
https://huggingface.co/datasets/google/fleurs

### Demo Video
https://drive.google.com/file/d/17KD3VfuFM4-J6EGf-_qKE_fZhlLflI3u/view?usp=sharing

---

## Model & Training Details

### Base Model
`openai/whisper-small`

### Fine-Tuning Strategy
LoRA (Low-Rank Adaptation)

### LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Rank (`r`) | 32 |
| Alpha (`lora_alpha`) | 64 |
| Target Modules | `q_proj`, `v_proj` |
| Dropout | 0.05 |

---

## Dataset

### Google FLEURS Hindi (`hi_in`)

Why FLEURS?
- Fully public dataset
- No authentication/token required
- Professionally recorded Hindi speech
- Pre-defined train/validation/test splits
- Audio already at 16kHz

### Dataset Usage

| Split | Samples Used |
|------|---------------|
| Train | 3000 |
| Validation | 500 |
| Test | 500 |

---

## Training Configuration

| Hyperparameter | Value |
|----------------|------|
| Batch Size | 8 |
| Gradient Accumulation | 2 |
| Effective Batch Size | 16 |
| Learning Rate | 1e-4 |
| Warmup Steps | 100 |
| Total Steps | 2000 |
| Precision | FP16 |
| GPU | T4 |

Estimated training time: ~2–2.5 hours on free Colab GPU.

---

## Key Learnings

- Encoder-decoder architectures perform better for Hindi ASR compared to CTC-based approaches.
- LoRA achieves strong performance improvements while training only ~0.6% of total model parameters.
- Whisper’s multilingual pretraining provides a strong Hindi baseline even before fine-tuning.

---

## Repository Structure

```text
AURA/
│
├── README.md
├── aura.py
└── AURA.ipynb
