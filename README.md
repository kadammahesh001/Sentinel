# Fine-Tuned DistilBERT for Sentiment Analysis and Review Intelligence

A comprehensive, production-grade Natural Language Processing assignment implementing the adaptation, fine-tuning, empirical evaluation, and full-stack web deployment of a pretrained **DistilBERT** language model (`distilbert/distilbert-base-uncased`) for binary review classification on the **IMDb Movie Reviews** dataset (`stanfordnlp/imdb`).

**Candidate**: **Mahesh Kadam** | **Roll Number**: **51** | **PRN**: **202401120074**  
**Institution**: MIT Academy of Engineering (MITAOE) | **Department**: Department of Data Science  
**Academic Submission Date**: November 2026

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.14-blue.svg)](https://python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.15-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/HuggingFace-Transformers_5.24-ffd21e.svg)](https://huggingface.co/transformers/)
[![Accuracy](https://img.shields.io/badge/Held--Out%20Test%20Accuracy-84.10%25-brightgreen.svg)]()
[![F1-Score](https://img.shields.io/badge/F1--Score-0.8349-success.svg)]()
[![Hardware](https://img.shields.io/badge/Hardware%20Acceleration-Apple%20Silicon%20MPS-purple.svg)]()

---

## 🌟 Project Highlights

- **Actual Fine-Tuning Execution**: Fine-tunes all 6 transformer encoder blocks and a custom classification head on local Apple Silicon Metal Performance Shaders (MPS) hardware.
- **Strict Data Leakage Prevention**: Enforces isolated train (2,000 samples), validation (500 samples), and strictly held-out test splits (1,000 samples) with seed determinism.
- **Genuine Empirical Benchmark**: Elevates test accuracy from **40.00%** (untuned baseline) to **84.10%** (**+44.10% lift**) and F1-score to **0.8349** on the held-out test partition.
- **Interactive Full-Stack Web Application**: Flask REST backend with Bootstrap 5 modern dark UI, Chart.js polarity and confidence charts, batch CSV processing, and export tools.
- **Probability Calibration & Ethics**: Explains that softmax confidence reflects token representation patterns, not factual certainty.
- **Academic Deliverables**: Includes a 12-slide PowerPoint presentation (`.pptx`), full 19-section technical report, evaluation report, and 32 viva questions with answers.

---

## 📊 Empirical Evaluation Results (Real Held-Out Test Data, N = 1,000)

Evaluated on the exact same 1,000 held-out test reviews from `stanfordnlp/imdb`:

| Metric | Baseline Pretrained Checkpoint | Fine-Tuned DistilBERT | Absolute Performance Delta |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **40.00%** | **84.10%** | **+44.10% LIFT** |
| **Precision (Binary)** | 0.4091 | **0.8683** | **+0.4592** |
| **Recall (Binary)** | 0.4500 | **0.8040** | **+0.3540** |
| **F1-Score (Binary)** | 0.4286 | **0.8349** | **+0.4063 LIFT** |
| **Macro F1-Score** | 0.3888 | **0.8409** | **+0.4521** |
| **Specificity (TNR)** | 35.00% | **87.80%** | **+52.80%** |
| **Sensitivity (TPR)** | 45.00% | **80.40%** | **+35.40%** |

### Confusion Matrix Breakdown (Fine-Tuned Model, N = 1,000)
- **True Negatives (TN)**: 439
- **False Positives (FP)**: 61
- **False Negatives (FN)**: 98
- **True Positives (TP)**: 402

---

## 📁 Repository Structure

```
review-llm-project/
├── app.py                      # Flask REST backend and web server
├── requirements.txt            # Project dependencies
├── README.md                   # Project overview & execution manual
├── train.py                    # Model fine-tuning script (PyTorch + HF Transformers)
├── evaluate.py                 # Evaluation benchmark & plot generator
├── inference.py                # SentimentClassifier inference engine
├── dataset/                    # Dataset directory
│   ├── held_out_test.csv       # 1,000 held-out test reviews for evaluation
│   └── sample_test_reviews.csv # 50 demo reviews for batch CSV upload
├── models/
│   └── finetuned_distilbert/   # Serialized model weights, tokenizer & config
├── results/                    # Actual generated evaluation logs & plots
│   ├── metrics.json            # Machine-readable evaluation metrics
│   ├── metrics_comparison.csv  # Side-by-side benchmark comparison table
│   ├── training_history.json   # Epoch loss & accuracy logs
│   ├── confusion_matrix.png    # Confusion matrix visual heatmaps
│   ├── metrics_comparison.png  # Accuracy/Precision/Recall/F1 bar chart
│   └── training_curves.png     # Loss & accuracy convergence curves
├── notebooks/
│   └── distilbert_finetuning.ipynb # Runnable Google Colab / Jupyter notebook
├── templates/
│   └── index.html              # Modern responsive analytics dashboard
├── static/
│   ├── css/
│   │   └── style.css           # Custom dark theme styles
│   ├── js/
│   │   └── main.js             # Client interactivity & Chart.js logic
│   └── img/                    # Static plot images for the dashboard
├── reports/
│   ├── technical_report.md     # Full 19-section formal academic report
│   ├── evaluation_report.md    # Deep dive into evaluation & error analysis
│   └── viva_questions.md       # 32 comprehensive viva questions and answers
├── presentation/
│   ├── generate_presentation.py # Script generating PPTX slides
│   ├── technical_presentation.pptx # 12-slide academic presentation deck
│   └── slides_outline.md       # Presentation script and slide-by-slide guide
└── tests/
    └── test_app.py             # Automated unit and integration test suite
```

---

## 🔬 Academic Architecture Note: Pretrained Language Model (PLM) vs. Generative LLM

> [!NOTE]
> **Model Classification**: DistilBERT (`distilbert-base-uncased`, ~66.9M parameters) is an **encoder-based pretrained transformer language model (PLM)**, not a generative autoregressive decoder model (like GPT-4 or Llama-3).
> 
> - **Why DistilBERT is optimal for this task**: For discriminative sentiment classification, bidirectional encoder models evaluate the entire sentence context simultaneously through the pooled `[CLS]` token. This provides superior accuracy, deterministic calibration, microsecond inference latency, and a lightweight memory footprint (~260 MB) compared to generative decoders.
> - **Instructor Scope**: If an academic evaluation committee strictly mandates a generative decoder LLM, a feasible alternative using parameter-efficient fine-tuning (PEFT/LoRA) on a small instruction model (e.g., `Qwen2.5-0.5B-Instruct` or `SmolLM2-360M-Instruct`) is documented in `reports/project_audit.md`.

---

## 🚀 Quickstart Guide

### 1. Prerequisites & Environment Setup

- **Tested Python Version**: Python 3.10 through Python 3.14 on macOS (Apple Silicon arm64) and Linux/Windows x86_64.
- **Hardware Acceleration**: Automatically selects Apple Silicon Metal Performance Shaders (`mps`), NVIDIA CUDA (`cuda`), or CPU fallback.

#### macOS / Linux (Terminal):
```bash
cd review-llm-project
pip install -r requirements.txt
```

#### Windows (PowerShell):
```powershell
Set-Location -Path review-llm-project
pip install -r requirements.txt
```

### 2. Fine-Tune the DistilBERT Model
The fine-tuning script automatically detects Apple Silicon MPS, CUDA, or CPU:

#### macOS / Linux:
```bash
python3 train.py --train_samples 2000 --val_samples 500 --test_samples 1000 --epochs 2 --batch_size 16
```

#### Windows PowerShell:
```powershell
python train.py --train_samples 2000 --val_samples 500 --test_samples 1000 --epochs 2 --batch_size 16
```
*(Note: The model has already been fine-tuned and verified in `models/finetuned_distilbert/`)*

### 3. Run Benchmark Evaluation
Evaluate the untrained baseline vs. the fine-tuned model on 1,000 held-out reviews:

#### macOS / Linux:
```bash
python3 evaluate.py
```

#### Windows PowerShell:
```powershell
python evaluate.py
```
This updates `results/metrics.json`, `results/metrics_comparison.csv`, and generates evaluation plots in `results/`.

### 4. Launch the Web Application

#### macOS / Linux:
```bash
python3 app.py
```

#### Windows PowerShell:
```powershell
python app.py
```
Open your browser and navigate to:
```
http://localhost:5000
```

### 5. Run Automated Tests
Execute the comprehensive test suite (16 automated unit/integration tests):

#### macOS / Linux:
```bash
python3 tests/test_app.py
```

#### Windows PowerShell:
```powershell
python tests/test_app.py
```

### 6. Cloud Training via Google Colab
If local compute is insufficient or you wish to demonstrate cloud training:
1. Upload [notebooks/distilbert_finetuning.ipynb](file:///Users/rajeshree/.gemini/antigravity-ide/scratch/nexusops-ai/review-llm-project/notebooks/distilbert_finetuning.ipynb) to Google Drive.
2. Open with Google Colab and set **Runtime > Change runtime type > T4 GPU**.
3. Run all cells sequentially to reproduce the training and evaluation loop on cloud hardware.

---

## 🖥️ Web Dashboard Features

1. **Live Review Analysis**:
   - Single-review input with character and word counters.
   - Pre-configured review samples (Masterpiece Positive, Critical Disaster Negative, Balanced, Sarcastic).
   - Real-time softmax confidence gauge and class probability split.
   - Calibrated intensity descriptions (*Strong Positive*, *Moderate Negative*, etc.).
   - Sequence token count and inference latency tracker.
2. **Session Inference Ledger**:
   - Live audit table recording every query analyzed during the session with 1-click reload.
3. **Batch CSV Intelligence**:
   - Drag-and-drop CSV file uploader (supports files up to 16 MB).
   - **Download Sample CSV** button for immediate viva demonstration.
   - Dynamic KPI cards: Total Reviews, Positive %, Negative %, Average Confidence.
   - **Chart.js Visualizations**: Polarity split Donut chart and Confidence Spread Bar chart.
   - Searchable, filterable table with **Export Enriched CSV** feature.
4. **Model Benchmarks & Metrics Tab**:
   - Interactive KPI cards, comparative matrix table, and confusion matrix cell breakdown.
   - Embedded high-resolution plots for metrics comparison and training loss convergence curves.
5. **Model Architecture & Viva Reference**:
   - Architectural specs table and interactive accordion containing key viva questions.

---

## 🎓 Academic Viva Voce Highlights

- **DistilBERT**: Distilled version of BERT that retains 97% of language understanding with 40% fewer parameters and 60% faster inference.
- **Classification Head**: `Linear(768 -> 768)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.2)` $\rightarrow$ `Linear(768 -> 2)` applied to the `[CLS]` sequence representation.
- **Data Leakage Prevention**: Validation split is carved strictly from the official training split; test split is strictly held-out and never seen during training or tuning.
- **Probability vs. Truth**: The model's confidence reflects statistical pattern correlation over token representations, not factual verification.

---

## 📜 License & Citation
Developed for academic assessment and educational demonstration.
References:
- Sanh et al. (2019), *DistilBERT: smaller, faster, cheaper and lighter*, arXiv:1910.01108.
- Devlin et al. (2018), *BERT: Pre-training of Deep Bidirectional Transformers*, arXiv:1810.04805.
- Maas et al. (2011), *Learning Word Vectors for Sentiment Analysis*, ACL 2011.
