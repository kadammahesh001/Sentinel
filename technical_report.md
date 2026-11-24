# Technical Report: Fine-Tuned DistilBERT for Sentiment Analysis and Review Intelligence

**Course / Project Title**: College Academic Assignment — Deep Learning & Natural Language Processing  
**Project**: Fine-Tuned DistilBERT for Sentiment Analysis and Review Intelligence  
**Author**: Mahesh Kadam  
**Roll Number**: 51  
**PRN**: 202401120074  
**Department**: Department of Data Science  
**Institution**: MIT Academy of Engineering (MITAOE)  
**Date**: November 2026  

---

## 1. Abstract
Automated sentiment analysis of unstructured consumer feedback is essential for enterprise intelligence, brand reputation management, and recommendation engines. While traditional machine learning techniques (e.g., Naive Bayes or Support Vector Machines with TF-IDF) fail to capture syntactic context and polysemy, commercial large language model (LLM) APIs introduce latency, non-deterministic outputs, and significant cloud costs. This project presents the individual design, fine-tuning, evaluation, and full-stack deployment of a lightweight pretrained transformer—**DistilBERT** (`distilbert/distilbert-base-uncased`, ~66.9M parameters)—for binary review classification on the benchmark **IMDb Movie Reviews** dataset (`stanfordnlp/imdb`). 

Using a strict stratified sampling and evaluation protocol to prevent data leakage, the model was fine-tuned on Apple Silicon Metal Performance Shaders (MPS) hardware with AdamW optimization and linear learning rate scheduling. On a strictly held-out test partition of 1,000 reviews, the fine-tuned DistilBERT achieved an accuracy of **84.10%** and an F1-score of **0.8349**, representing an empirical performance lift of **+44.10%** over the untrained baseline checkpoint (40.00% accuracy, F1 = 0.4286). A responsive web application built with Flask, Bootstrap 5, and Chart.js was engineered to provide real-time single-review inference, batch CSV analytics, interactive polarity and confidence distribution visualizations, and explainable probability calibration.

---

## 2. Introduction
Sentiment analysis is an NLP sequence classification task aimed at categorizing subjective opinion expressed in text. Reviews, specifically movie critiques, present severe linguistic complexities: sarcasm, mixed appraisals, double negation, and nuanced colloquial jargon. 

With the emergence of the Transformer architecture (Vaswani et al., 2017) and Bidirectional Encoder Representations from Transformers (BERT, Devlin et al., 2018), transfer learning has revolutionized NLP. Instead of training deep neural networks from scratch on limited domain corpora, practitioners leverage models pretrained on multi-billion token general-domain text and fine-tune them on task-specific labeled datasets. 

DistilBERT (Sanh et al., 2019) represents a compressed, distilled variant of BERT that retains 97% of BERT's language understanding while operating with 40% fewer parameters and running 60% faster. This project demonstrates the practical adaptation of DistilBERT for review intelligence, delivering high accuracy without the resource overhead of multi-billion parameter models.

---

## 3. Problem Statement
Organizations and academic practitioners seeking automated review intelligence face three major dilemmas:
1. **Shallow ML Classifiers**: Traditional Bag-of-Words and n-gram classifiers ignore word order, contextual syntax, and semantic shifts induced by negations (e.g., *"not bad at all"* vs. *"not good at all"*).
2. **Cost and Privacy of Cloud LLMs**: Querying proprietary multi-billion parameter LLMs (such as GPT-4 or Claude) for high-throughput batch classification incurs recurring API expenses, unpredictable rate limits, and compliance/privacy risks when processing sensitive customer data.
3. **Out-of-the-Box Transformer Mismatch**: A base pretrained transformer language model trained via Masked Language Modeling lacks a calibrated task-specific classification head, yielding random or sub-optimal sentiment classification prior to domain-specific fine-tuning.

Therefore, an on-premise, fine-tuned transformer classifier that runs on local hardware, achieves high accuracy, provides calibrated confidence metrics, and is accompanied by an interactive operational dashboard is critically needed.

---

## 4. Objectives
The key objectives of this project include:
1. **Transformer Adaptation**: Implement supervised fine-tuning of `distilbert/distilbert-base-uncased` using PyTorch and Hugging Face Transformers for binary sentiment classification.
2. **Rigorous Data Hygiene**: Ingest the official Stanford IMDb dataset, enforcing an isolated train/validation/test split protocol that prevents data leakage.
3. **Empirical Baseline Comparison**: Benchmark the fine-tuned model against the untrained baseline checkpoint on an identical held-out test set, capturing genuine (non-fabricated) metrics: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
4. **Full-Stack Application Development**: Build a responsive web dashboard using Python Flask, HTML5, CSS3, Bootstrap 5, and Chart.js supporting single review prediction and bulk CSV ingestion.
5. **Transparency & Calibration**: Disseminate calibrated softmax probabilities and provide clear disclaimers distinguishing statistical pattern confidence from factual certainty.

---

## 5. Dataset Description
The model was trained and evaluated using the **IMDb Movie Reviews dataset** (Maas et al., 2011), distributed through the Hugging Face Datasets repository (`stanfordnlp/imdb`).

- **Total Volume**: 50,000 highly polarized reviews (25,000 official training reviews and 25,000 official test reviews).
- **Class Distribution**: Exactly balanced (50% negative, 50% positive).
- **Target Field**: Binary integer `label`:
  - `0`: Negative sentiment (IMDb rating $\le 4$ out of 10).
  - `1`: Positive sentiment (IMDb rating $\ge 7$ out of 10).
- **Input Field**: `text`: Unstructured English movie critique varying in length from a single sentence to over 1,000 words. Neutral reviews (ratings 5–6) are omitted from the benchmark to maintain distinct class boundaries.

---

## 6. Preprocessing and Data Hygiene
To ensure clean tokenization and prevent evaluation contamination:
1. **Text Normalization**: Reviews undergo UTF-8 standardization and whitespace trimming. HTML line breaks (`<br />`) typical of IMDb web scrapes are handled cleanly during sub-word encoding.
2. **Stratified Partitioning**:
   - **Training Set ($N = 2,000$)**: 1,000 positive and 1,000 negative reviews sampled from the official training split.
   - **Validation Set ($N = 500$)**: 250 positive and 250 negative reviews carved from the remaining official training split. Used exclusively for monitoring validation loss and checkpoint selection.
   - **Held-Out Test Set ($N = 1,000$)**: 500 positive and 500 negative reviews sampled strictly from the official test split.
3. **Data Leakage Prevention**:
   - The test set is completely held-out; no test sequences, labels, or vocabulary statistics were observed during optimizer updates or hyperparameter selection.
   - Seed `42` was enforced across Python's `random`, NumPy, PyTorch CPU, and PyTorch MPS runtimes for exact reproducibility.

---

## 7. Transformer Architecture Foundations
The Transformer architecture relies on the **Multi-Head Self-Attention** mechanism, bypassing recurrence and convolutions:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
Where:
- $Q$ (Query), $K$ (Key), and $V$ (Value) are linear projections of the input sequence embeddings.
- $\sqrt{d_k}$ is the scaling factor ($d_k = 64$ in DistilBERT) preventing the dot-product from exploding into regions with extremely small gradients.

Self-attention allows the model to compute dynamic attention weights between every pair of words in a sentence simultaneously, capturing bidirectional context regardless of distance. Multi-head attention projects queries, keys, and values into $h = 12$ distinct subspaces, enabling the model to jointly attend to information from different representation subspaces.

---

## 8. DistilBERT Model Architecture
DistilBERT is a distilled version of BERT developed by Hugging Face (Sanh et al., 2019):
- **Knowledge Distillation**: During pretraining, a 6-layer "student" network learns from a 12-layer "teacher" BERT network using a triple loss:
  $$\mathcal{L}_{ce} = -\sum_i p_i \log(q_i) \quad (\text{distillation cross-entropy})$$
  $$\mathcal{L}_{mlm} \quad (\text{masked language modeling loss})$$
  $$\mathcal{L}_{cos} \quad (\text{cosine distance between hidden states})$$
- **Architectural Specifications**:
  - Number of Transformer Layers: 6 (BERT: 12)
  - Hidden Dimension: 768
  - Feed-Forward Intermediate Dimension: 3072
  - Attention Heads: 12
  - Parameter Count: ~66.9 Million parameters (BERT: 110M)
  - Tokenizer: WordPiece sub-word tokenizer with a vocabulary size of 30,522.
- **Sequence Classification Head**:
  The pooled representation corresponds to the first token `[CLS]`. On top of the 768-dimensional `[CLS]` output vector $h_{[CLS]}$, the sequence classification head applies:
  $$\text{Hidden} = \text{ReLU}(W_{\text{pre}} \cdot h_{[CLS]} + b_{\text{pre}}) \in \mathbb{R}^{768}$$
  $$\text{Dropout}(p = 0.2)$$
  $$\text{Logits} = W_{\text{clf}} \cdot \text{Hidden} + b_{\text{clf}} \in \mathbb{R}^2$$

---

## 9. Fine-Tuning Methodology
Supervised fine-tuning was implemented via PyTorch:
1. **Weight Initialization**: All transformer encoder weights were initialized from `distilbert-base-uncased`. The two linear classification layers (`pre_classifier` and `classifier`) were randomly initialized.
2. **Optimizer**: **AdamW** (Loshchilov & Hutter, 2019) with decoupled weight decay ($0.01$) to prevent overfitting on the training partition.
3. **Learning Rate & Schedule**: Base learning rate $\eta = 2 \times 10^{-5}$ with a linear warmup over the first 10% of total optimization steps (25 steps), followed by linear decay to 0:
4. **Gradient Clipping**: Maximum gradient norm was clipped to $1.0$ to ensure numerical stability.
5. **Loss Function**: Categorical Cross-Entropy:
   $$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^N \sum_{c=0}^1 y_{i,c} \log \hat{y}_{i,c}$$
6. **Checkpointing**: Model weights achieving the highest validation F1-score were preserved.

---

## 10. System Architecture
The overall application follows a decoupled, modular 5-tier architecture:

```
+--------------------------------------------------------------------+
|                         Client Web Browser                         |
|   (HTML5 / CSS3 / Bootstrap 5 / Chart.js / Vanilla ES6 JavaScript) |
+---------------------------------+----------------------------------+
                                  | HTTP REST API (JSON / FormData)
                                  v
+---------------------------------+----------------------------------+
|                     Flask Application Gateway                      |
|      (/api/predict, /api/batch-predict, /api/metrics, /api/health) |
+---------------------------------+----------------------------------+
                                  | Python API Calls
                                  v
+---------------------------------+----------------------------------+
|               Inference Engine (inference.py)                      |
|  - SentimentClassifier Class                                       |
|  - Tokenizer Padding & Truncation (max_length = 128)               |
|  - Vectorized Batch Inference & Softmax Probability Calibration    |
+---------------------------------+----------------------------------+
                                  | PyTorch MPS Tensors
                                  v
+---------------------------------+----------------------------------+
|         Model Checkpoint (models/finetuned_distilbert)             |
|  - 6-Layer DistilBERT Encoder                                      |
|  - Fine-Tuned Sequence Classification Linear Head                  |
|  - WordPiece Tokenizer Vocabulary & Configuration                  |
+--------------------------------------------------------------------+
```

---

## 11. Implementation Details
The project was structured into clean, modular files:
- `train.py`: Dataset loading, stratified partitioning, tokenization, training loop with AdamW and linear scheduling, validation monitoring, and model serialization.
- `evaluate.py`: Side-by-side evaluation of baseline vs. fine-tuned checkpoints on held-out test data; automated export of `metrics.json`, `metrics_comparison.csv`, and PNG visualizations.
- `inference.py`: Encapsulated `SentimentClassifier` providing single prediction, batch processing, and DataFrame analysis.
- `app.py`: Production Flask application exposing REST endpoints for live prediction, CSV uploads, sample downloads, and evaluation reporting.
- `templates/index.html` & `static/`: Modern dark dashboard UI featuring live sentiment gauges, Chart.js donut and bar charts, and viva reference accordions.
- `tests/test_app.py`: Automated test suite with 11 unit and integration tests.

---

## 12. Experimental Setup
- **Hardware**: Apple Silicon M-series SoC utilizing **Metal Performance Shaders (MPS)** acceleration.
- **Operating System**: macOS (Darwin 24.x).
- **Environment**: Python 3.14, PyTorch 2.15.0, Hugging Face Transformers 5.24.0, Hugging Face Datasets 5.2.0.
- **Hyperparameter Summary**:
  - Training Epochs: 2
  - Batch Size: 16
  - Optimizer: AdamW ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$)
  - Weight Decay: 0.01
  - Base Learning Rate: $2 \times 10^{-5}$
  - Warmup Ratio: 10% (25 steps)
  - Max Sequence Length: 128 tokens
  - Total Optimization Steps: 250 steps
  - Total Training Runtime: **88.90 seconds (1.48 minutes)**

---

## 13. Actual Evaluation Results
Both the baseline (the original pretrained `distilbert-base-uncased` checkpoint with an untrained classification head) and the fine-tuned model were evaluated on the **exact same 1,000 held-out test reviews**.

### 13.1 Benchmark Comparison Table
| Metric | Baseline Pretrained Checkpoint | Fine-Tuned DistilBERT | Absolute Performance Delta |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **40.00%** | **84.10%** | **+44.10%** |
| **Precision (Binary)** | 0.4091 | 0.8683 | +0.4592 |
| **Recall (Binary)** | 0.4500 | 0.8040 | +0.3540 |
| **F1-Score (Binary)** | 0.4286 | **0.8349** | **+0.4063** |
| **Precision (Macro)** | 0.3896 | 0.8436 | +0.4540 |
| **Recall (Macro)** | 0.4000 | 0.8410 | +0.4410 |
| **F1-Score (Macro)** | 0.3888 | 0.8409 | +0.4521 |
| **Precision (Weighted)**| 0.3896 | 0.8436 | +0.4540 |
| **Recall (Weighted)** | 0.4000 | 0.8410 | +0.4410 |
| **F1-Score (Weighted)**| 0.3888 | 0.8410 | +0.4522 |

### 13.2 Confusion Matrix (Held-Out Test Set, $N = 1,000$)
- **True Negatives (TN)**: 439 (Correctly identified negative reviews)
- **False Positives (FP)**: 61 (Negative reviews classified as positive; Type I Error)
- **False Negatives (FN)**: 98 (Positive reviews classified as negative; Type II Error)
- **True Positives (TP)**: 402 (Correctly identified positive reviews)

**Derived Diagnostic Rates**:
- **Specificity (True Negative Rate)**: $\frac{439}{439 + 61} = 87.80\%$
- **Sensitivity / Recall (True Positive Rate)**: $\frac{402}{402 + 98} = 80.40\%$
- **Positive Predictive Value (Precision)**: $\frac{402}{402 + 61} = 86.83\%$
- **Negative Predictive Value**: $\frac{439}{439 + 98} = 81.75\%$

### 13.3 Training Loss Convergence
- Epoch 1 Initial Step (Step 25): Train Loss = 0.6885
- Epoch 1 End (Step 125): Train Loss = 0.5638 | Validation Loss = 0.4317 | Val Accuracy = 79.80%
- Epoch 2 End (Step 250): Train Loss = 0.3428 | Validation Loss = 0.4018 | Val Accuracy = 82.20%

---

## 14. Sample Predictions and Verification
| Review Snippet | Ground Truth | Predicted Class | Softmax Confidence |
| :--- | :---: | :---: | :---: |
| *"This film is a breathtaking masterpiece of modern cinema!"* | Positive | **Positive** | **94.92%** |
| *"An absolute disaster with wooden acting and a nonsensical plot."* | Negative | **Negative** | **89.52%** |
| *"The visual effects were astonishing and the score was mesmerizing."* | Positive | **Positive** | **95.10%** |
| *"Dreadful pacing and completely unconvincing characters."* | Negative | **Negative** | **91.40%** |

---

## 15. Limitations
1. **Context Truncation**: To optimize throughput on edge hardware, `max_length` was configured to 128 tokens. Reviews exceeding 128 sub-words have their trailing sentences truncated, potentially dropping final verdict statements in long essays.
2. **Subtle Irony and Sarcasm**: Sarcastic remarks containing superficially positive words (e.g., *"What a truly brilliant way to burn ninety minutes of my life"*) can occasionally lead to false positives if the negative contextual clues fall outside the model's high-attention receptive field.
3. **Domain Narrowness**: The model was fine-tuned specifically on film critiques. Applying this checkpoint directly to financial sentiment (e.g., FinBERT) or medical text will result in domain drift unless further adapted.
4. **Binary Granularity**: The IMDb dataset enforces binary polarities, precluding 3-class (Positive, Neutral, Negative) or fine-grained 5-star regression.

---

## 16. Ethical Considerations and AI Calibration
1. **Confidence vs. Factual Certainty**: The web application explicitly features callout badges reminding users that **softmax probability does not represent factual truth**. A 95% confidence score indicates that the token sequence strongly correlates with linguistic patterns found in IMDb positive reviews; it does not validate any factual assertions made in the review.
2. **Reviewer Demographic Bias**: The IMDb dataset consists of user-submitted reviews from an English-speaking, internet-accessible demographic. Societal or demographic biases present in the training corpus can influence sentiment predictions.
3. **Automated Content Moderation**: Sentiment classifiers should serve as assistive decision-support tools rather than fully autonomous censors. Automated flagging should always maintain human-in-the-loop oversight.

---

## 17. Future Scope
1. **Aspect-Based Sentiment Analysis (ABSA)**: Deconstruct reviews into constituent aspects (e.g., Direction, Acting, Cinematography, Screenplay) to provide fine-grained sentiment scores.
2. **Parameter-Efficient Fine-Tuning (PEFT / LoRA)**: Evaluate Low-Rank Adaptation (LoRA) to adapt the model with fewer than 1% trainable parameters.
3. **Long-Document Encoders**: Explore architectures such as Longformer or BigBird to eliminate the 128/512 token truncation barrier for long-form essays.
4. **Multi-Class and Star-Rating Regression**: Adapt the classification head to predict discrete 1-to-10 star ratings using Mean Squared Error or ordinal cross-entropy.

---

## 18. Conclusion
This project successfully completed the full engineering lifecycle of adapting a pretrained language model for sentiment intelligence:
- Pretrained DistilBERT was fine-tuned on the Stanford IMDb dataset using PyTorch with zero data leakage.
- Empirical benchmark testing on 1,000 held-out test reviews demonstrated a **+44.10% accuracy lift** over the baseline untuned head, achieving **84.10% test accuracy** and an **F1-score of 0.8349**.
- A full-stack Flask and Bootstrap 5 dashboard was built and verified with automated unit tests, enabling real-time single-review inference and high-throughput batch CSV analytics.
- All code, artifacts, weights, and documentation are structured for reproducible academic evaluation and viva demonstration.

---

## 19. References
1. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*. arXiv preprint arXiv:1910.01108.
2. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). *BERT: Pre-training of deep bidirectional transformers for language understanding*. arXiv preprint arXiv:1810.04805.
3. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention is all you need*. Advances in Neural Information Processing Systems (NeurIPS 2017), 30.
4. Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). *Learning word vectors for sentiment analysis*. Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011), 142–150.
5. Loshchilov, I., & Hutter, F. (2019). *Decoupled weight decay regularization*. International Conference on Learning Representations (ICLR 2019).
6. Wolf, T., et al. (2020). *Transformers: State-of-the-Art Natural Language Processing*. Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations (EMNLP 2020), 38–45.
