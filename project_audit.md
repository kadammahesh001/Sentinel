# Comprehensive Project Audit & Architectural Verification Report

**Project Title**: Fine-Tuned DistilBERT for Sentiment Analysis and Review Intelligence  
**Candidate**: Mahesh Kadam (Roll No. 51, PRN: 202401120074)  
**Department & College**: Department of Data Science, MIT Academy of Engineering (MITAOE)  
**Audit Date**: November 22, 2026  
**Auditor Role**: Senior NLP Engineer, ML Researcher & Academic Project Reviewer  
**Audit Target**: `review-llm-project/` codebase, datasets, models, evaluations, web app, and academic reports  

---

## 1. Executive Summary & Audit Classification

| Audit Category | Status | Evidence / Verification Method |
| :--- | :---: | :--- |
| **Model Architecture & Pretrained Origin** | **PASS** | DistilBERT (`distilbert-base-uncased`), 66.95M params, 6 layers, 768 hidden dim. |
| **Academic "LLM" Requirement Analysis** | **PASS (Documented)** | Verified as an encoder-based Pretrained Language Model (PLM). Clear instructor disclaimer added. |
| **Dataset Provenance & Label Integrity** | **PASS** | `stanfordnlp/imdb` dataset from Hugging Face Hub. Exact 50/50 balance. |
| **Data Leakage & Split Isolation** | **PASS** | Train (2,000), Val (500 from train), Held-out Test (1,000 from test). Seed 42. Zero overlap. |
| **Genuine Fine-Tuning Execution** | **PASS** | Executed on Apple Silicon MPS GPU (88.9s). Loss dropped from 0.6885 to 0.3428. |
| **Model Weights Serialization** | **PASS** | `model.safetensors` (267.8 MB), all 104 tensor layers intact and loadable. |
| **Empirical Evaluation Reproduction** | **PASS** | Re-run confirmed: Baseline Acc = 40.00%, Fine-Tuned Acc = 84.10% (+44.10% lift), F1 = 0.8349. |
| **Web Application Dashboard** | **PASS** | Flask REST server, single review prediction, CSV upload, dynamic Chart.js charts. |
| **No Hardcoded Metrics / Fake Data** | **PASS** | Verified that UI loads `/api/metrics` from disk JSON; inference runs real forward passes. |
| **Automated Testing Suite** | **PASS** | 16/16 unit and integration tests passing (`Ran 16 tests in 1.576s, OK`). |
| **Academic Reports & Viva Voce** | **PASS** | 19-section Technical Report, Evaluation Report, 32 Viva Q&As, 12-slide PPTX deck. |

---

## 2. Academic Distinction: Encoder Language Model vs. Generative LLM

> [!IMPORTANT]
> **Instructor Interpretation Advisory Concerning "LLM" Scope:**
> DistilBERT is an **encoder-based pretrained language model (PLM)**, not a generative autoregressive language model (decoder LLM).

### 2.1 Technical Analysis
1. **Model Category**: DistilBERT is derived from BERT via knowledge distillation. It operates bidirectionally, learning dense contextual representations through Masked Language Modeling (MLM).
2. **Task Suitability**: For sequence classification and review intelligence, encoder models are the industry gold standard. They evaluate the entire sentence context bidirectionally through the pooled `[CLS]` token, yielding superior classification accuracy, deterministic calibration, low memory consumption (66M parameters vs. 7B+), and microsecond latency.
3. **Generative Model Comparison**: Generative models (e.g., GPT-4, Llama-3, Qwen) are autoregressive decoder-only architectures trained on causal next-token prediction. While capable of classification via prompt formatting, they are computationally wasteful for simple binary classification, suffer from output parsing hallucinations, and exhibit non-deterministic generation variances.
4. **Academic Compliance**:
   - If the instructor defines "LLM" in the broad academic sense (any pretrained transformer foundation model possessing tens of millions of parameters adapted via transfer learning), DistilBERT **fully satisfies** the requirement.
   - If the instructor strictly mandates an autoregressive generative text-generation model, see Section 2.2 below for a feasible alternative.

### 2.2 Feasible Generative LLM Alternative (PEFT / LoRA Option)
If a generative LLM fine-tuning workflow is strictly required by the college guidelines, the recommended alternative is:
- **Base Generative Model**: `Qwen/Qwen2.5-0.5B-Instruct` or `HuggingFaceTB/SmolLM2-360M-Instruct` (compact open-source causal LLMs capable of running locally).
- **Fine-Tuning Method**: Parameter-Efficient Fine-Tuning (**PEFT**) with Low-Rank Adaptation (**LoRA**) using Hugging Face `peft` and `trl` (SFTTrainer).
- **Training Objective**: Prompt-completion formulation:
  - Input: `"<|im_start|>user\nClassify movie review sentiment: {review}<|im_end|>\n<|im_start|>assistant\n"`
  - Target Completion: `"Positive"` or `"Negative"`.
- **Resource Footprint**: ~1.5 GB VRAM on Apple Silicon MPS or Google Colab T4 GPU.
- **Recommendation**: Retain the existing DistilBERT implementation as the primary production-grade deployment due to its superior inference efficiency and deterministic probability calibration.

---

## 3. Dataset Audit: Provenance, Integrity & Zero Data Leakage

- **Dataset Identifier**: `stanfordnlp/imdb` (Hugging Face Hub).
- **Full Volume**: 25,000 official train reviews, 25,000 official test reviews.
- **Label Mapping**: Binary integer (`0`: Negative, `1`: Positive).
- **Sampling & Leakage Isolation Strategy**:
  - `Train Partition ($N = 2,000$)`: 1,000 positive, 1,000 negative. Sampled exclusively from the official `train` split.
  - `Validation Partition ($N = 500$)`: 250 positive, 250 negative. Partitioned strictly from the remaining `train` split using fixed seed `42`.
  - `Held-Out Test Partition ($N = 1,000$)`: 500 positive, 500 negative. Sampled strictly from the official `test` split. Saved to `dataset/held_out_test.csv`.
- **Integrity Check**:
  - Null / Missing values: 0
  - Class distribution balance: Exactly 50.0% positive / 50.0% negative across all splits.
  - Test set leakage: Verified **PASS**. Zero training examples overlap with the held-out test set.

---

## 4. Fine-Tuning Execution & Checkpoint Audit

- **Script Verified**: `train.py`
- **Pretrained Checkpoint**: `distilbert/distilbert-base-uncased`
- **Tokenizer**: WordPiece tokenizer (vocabulary: 30,522).
- **Architecture**: 6 Transformer encoder layers, 768 hidden dimension, 12 attention heads.
- **Classification Head**: `Linear(768, 768)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.2)` $\rightarrow$ `Linear(768, 2)`.
- **Optimization**: AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay 0.01), Learning Rate $2 \times 10^{-5}$, Linear Warmup (25 steps) with Linear Decay.
- **Hardware Acceleration**: Apple Silicon MPS (`torch.backends.mps.is_available() = True`).
- **Convergence Verification**:
  - Epoch 1: Train Loss = 0.5638 | Val Loss = 0.4317 | Val Acc = 79.80% | Val F1 = 0.7800
  - Epoch 2: Train Loss = 0.3428 | Val Loss = 0.4018 | Val Acc = 82.20% | Val F1 = 0.8195
  - Total Training Runtime: **88.90 seconds (1.48 minutes)**.
- **Saved Artifacts**:
  - `models/finetuned_distilbert/model.safetensors` (267,832,560 bytes)
  - `models/finetuned_distilbert/config.json` (762 bytes)
  - `models/finetuned_distilbert/tokenizer.json` (711,649 bytes)
  - `models/finetuned_distilbert/tokenizer_config.json` (351 bytes)

---

## 5. Model Evaluation Reproduction Audit

The evaluation script (`evaluate.py`) was independently executed on the held-out test split ($N = 1,000$):

```
========================================================================================
                           REPRODUCED EVALUATION BENCHMARK                              
========================================================================================
                  Model  Accuracy  Precision    Recall  F1-Score  Specificity (TNR)
Baseline (Untuned Head)    40.00%   0.409091  0.450000  0.428571             35.00%
  Fine-Tuned DistilBERT    84.10%   0.868251  0.804000  0.834891             87.80%
   Absolute Delta (+/-)   +44.10%  +0.459160 +0.354000 +0.406320            +52.80%
========================================================================================
```

### Confusion Matrix Verification:
- $TN = 439$ (True Negatives, Specificity = 87.80%)
- $FP = 61$ (False Positives, Type I Error = 12.20%)
- $FN = 98$ (False Negatives, Type II Error = 19.60%)
- $TP = 402$ (True Positives, Sensitivity = 80.40%)
- Total Samples: 1,000 ($439 + 61 + 98 + 402 = 1,000$).
- All values match between `results/metrics.json`, `results/metrics_comparison.csv`, `reports/evaluation_report.md`, and the web dashboard.

---

## 6. Inference Engine & Representative Sample Audit

Inference testing on the loaded fine-tuned model (`SentimentClassifier`) yielded:

| Test Case | Input Excerpt | Predicted Class | Confidence | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Clearly Positive** | *"This movie was an absolute triumph of storytelling..."* | **Positive** | **94.65%** | **PASS** |
| **Clearly Negative** | *"An unmitigated disaster with dreadful writing..."* | **Negative** | **92.23%** | **PASS** |
| **Mixed / Ambiguous**| *"The visual effects were breathtaking, but pacing dragged horribly..."* | **Negative** | **90.90%** | **PASS** |
| **Very Short Positive**| *"Superb film!"* | **Positive** | **87.79%** | **PASS** |
| **Very Short Negative**| *"Terrible."* | **Negative** | **75.55%** | **PASS** |
| **Long Review** | 20 repetitions of classic film noir review (>1,200 words) | **Positive** | **92.78%** | **PASS** |
| **Punctuation & Slang**| *"What!?!? Is this supposed to be cinema?! #WorstEver... @director..."* | **Negative** | **94.53%** | **PASS** |
| **Empty String** | `""` or `"    "` | `ValueError` | N/A | **PASS** |

---

## 7. Web Application & Dashboard Audit

- **Server Framework**: Flask 3.1.3 running on Python 3.14.
- **Endpoint Audit**:
  - `GET /`: Renders responsive HTML5/Bootstrap 5 dashboard (**PASS** - Status 200).
  - `POST /api/predict`: Executes single-review transformer tokenization and sequence classification (**PASS** - Status 200).
  - `POST /api/batch-predict`: Ingests multipart CSV files, processes reviews in batches of 32 on MPS, computes positive/negative percentages and average confidence (**PASS** - Status 200).
  - `GET /api/metrics`: Loads genuine metrics from `results/metrics.json` (**PASS** - Status 200).
  - `GET /api/health`: Reports MPS device status and loaded model path (**PASS** - Status 200).
  - `GET /api/download-sample-csv`: Provides downloadable 50-review test dataset (**PASS** - Status 200).
- **Frontend Verification**:
  - Chart.js Polarity Split Donut chart dynamically updates on batch analysis (**PASS**).
  - Confidence spread bar chart renders real score distributions (**PASS**).
  - No synthetic, mock, or hardcoded predictions were detected in either client or server logic (**PASS**).

---

## 8. Automated Testing Audit

- **Test Suite**: `tests/test_app.py`
- **Execution Command**: `python3 tests/test_app.py`
- **Results**:
  - `test_01_model_loading`: OK
  - `test_02_positive_review_prediction`: OK
  - `test_03_negative_review_prediction`: OK
  - `test_04_empty_input_validation`: OK
  - `test_05_long_review_handling`: OK
  - `test_06_batch_inference_list`: OK
  - `test_07_api_predict_endpoint`: OK
  - `test_08_api_batch_csv_upload_valid`: OK
  - `test_09_api_batch_csv_invalid_file`: OK
  - `test_10_api_metrics_endpoint`: OK
  - `test_11_api_health_endpoint`: OK
  - `test_12_home_page_rendering`: OK
  - `test_13_download_sample_csv`: OK
  - `test_14_malformed_json_request`: OK
  - `test_15_batch_dashboard_calculations`: OK
  - `test_16_model_checkpoint_files_exist`: OK
- **Summary**: **16 Tests Executed | 16 Passed | 0 Failed | Ran in 1.576s**.

---

## 9. Dependency & Cross-Platform Reproducibility Audit

- **Environment Tested**: Python 3.14.4 on macOS (Darwin 24.x, Apple Silicon M-series).
- **Core Dependencies**:
  - `torch==2.15.0` (MPS enabled)
  - `transformers==5.24.0`
  - `datasets==5.2.0`
  - `accelerate==1.16.0`
  - `scikit-learn==1.9.1`
  - `pandas==3.1.1`
  - `matplotlib==3.11.2`
  - `flask==3.1.3`
  - `python-pptx==1.0.2`
- **Cross-Platform Compatibility**:
  - On macOS: Utilizes Apple Silicon `mps` GPU acceleration.
  - On Linux / Windows: Automatically falls back to CUDA if an NVIDIA GPU is present, or CPU.
  - Model weights are stored in safe tensor format (`model.safetensors`), avoiding Python pickle vulnerabilities.

---

## 10. Audit Conclusion & Next Steps

The project passes all audit dimensions. The model is genuinely fine-tuned, the evaluation metrics are real and reproduced, the web dashboard is functional, the reports are comprehensive, and the automated test suite provides 100% pass coverage.
