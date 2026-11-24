# Presentation Slides Outline & Script

**Presentation File**: `presentation/technical_presentation.pptx`  
**Deck Format**: 16:9 Widescreen (13.33" x 7.5"), Professional Academic Dark Navy Palette  
**Target Audience**: College Project Evaluation Committee & External Viva Examiners  

---

### Slide 1: Title & Candidate Credentials
- **Headline**: Fine-Tuned DistilBERT for Sentiment Analysis and Review Intelligence
- **Subtitle**: Pretrained Transformer Adaptation, Evaluation Benchmarking, and Full-Stack Web Deployment
- **Student Credentials**:
  - Student Name: `Mahesh Kadam`
  - Roll Number: `51`
  - PRN: `202401120074`
  - Department: `Department of Data Science`
  - Institution: `MIT Academy of Engineering (MITAOE)`
  - Academic Year: `2026-2027`

### Slide 2: Problem Statement & Motivation
- **Left Column (Limitations of Traditional Approaches)**: Bag-of-Words & TF-IDF ignore syntax; Cloud LLMs (GPT-4) introduce recurring costs, latency, and data privacy risks; base pretrained checkpoints lack task-specific calibration.
- **Right Column (DistilBERT Fine-Tuned Solution)**: 40% smaller and 60% faster than BERT; domain adaptation on IMDb movie reviews; runs on edge/local hardware (MPS/CUDA); calibrated confidence scoring.

### Slide 3: Project Objectives & Scope
- **5 Core Pillars**: Pretrained Model Adaptation, Leakage-Free Dataset Pipeline, Empirical Baseline Benchmark, Interactive Full-Stack System, Academic Deliverables.

### Slide 4: LLM & Transformer Architectural Foundations
- **Pillar 1**: Multi-Head Self-Attention ($\text{Attention}(Q,K,V) = \text{softmax}(QK^T/\sqrt{d_k})V$).
- **Pillar 2**: Pretraining (Masked LM on 3.3B words) vs. Fine-Tuning (Supervised classification).
- **Pillar 3**: WordPiece Sub-Word Tokenization (30,522 vocabulary) resolving Out-of-Vocabulary issues.

### Slide 5: Dataset Ingestion & Split Strategy
- **Dataset**: Stanford IMDb (`stanfordnlp/imdb`), 50,000 balanced binary reviews.
- **Leakage-Free Partitioning**:
  - Training: 2,000 samples (from official train split).
  - Validation: 500 samples (from train split, seed 42).
  - Held-Out Test: 1,000 samples (strictly isolated from official test split).

### Slide 6: DistilBERT Model Architecture & Classification Head
- **Backbone**: 6 Transformer encoder blocks, 768 hidden dimension, 12 attention heads, ~66.9M parameters.
- **Classification Head**: `Linear(768, 768)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(p=0.2)` $\rightarrow$ `Linear(768, 2)` applied to pooled `[CLS]` embedding.

### Slide 7: Fine-Tuning Methodology & Hyperparameters
- **Optimization**: AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay 0.01), Learning Rate $2 \times 10^{-5}$ with linear warmup (25 steps) and linear decay.
- **Batching**: Batch size 16, max sequence length 128, gradient clipping max_norm 1.0.
- **Runtime**: 2 epochs completed in 88.90s on Apple Silicon MPS.

### Slide 8: End-to-End System Architecture
- **5-Layer Architecture**:
  1. Training Layer (`train.py`)
  2. Evaluation Layer (`evaluate.py`)
  3. Inference Engine (`inference.py`)
  4. Application Server (`app.py` REST API)
  5. Responsive UI Dashboard (`templates/index.html` + Bootstrap 5 + Chart.js)

### Slide 9: Actual Empirical Results (Held-Out Test Set N=1,000)
- **Real Benchmark**:
  - Baseline Accuracy: 40.00% $\rightarrow$ Fine-Tuned Accuracy: **84.10%** (**+44.10% Lift**)
  - Baseline F1: 0.4286 $\rightarrow$ Fine-Tuned F1: **0.8349** (**+0.4063 Lift**)
  - Precision: 0.8683 | Recall: 0.8040
- **Confusion Matrix**: TN = 439 (87.80% Specificity), FP = 61, FN = 98, TP = 402 (80.40% Sensitivity).

### Slide 10: Web Application & Demonstration
- Single Review Inference with real-time confidence gauge & token count.
- Pre-set sample review buttons (Masterpiece, Disaster, Balanced, Sarcastic).
- Batch CSV upload with automatic text column detection and enriched CSV export.
- Chart.js dynamic visualizations: Polarity split Donut chart and Confidence Spread Bar chart.

### Slide 11: Limitations & Ethical Considerations
- **Technical Limitations**: Sequence truncation at 128 tokens; subtle irony/sarcasm challenges; domain specificity.
- **Ethics & Calibration**: Model confidence reflects statistical correlation, not factual certainty; mitigation of review demographic bias; human-in-the-loop decision making.

### Slide 12: Conclusion, Future Scope & References
- **Summary**: Delivered complete lifecycle from model fine-tuning to empirical benchmark and web deployment.
- **Future Scope**: Aspect-Based Sentiment Analysis (ABSA), PEFT/LoRA adaptation, multi-class star rating regression.
- **Academic References**: Sanh et al. (2019), Devlin et al. (2018), Vaswani et al. (2017), Maas et al. (2011).
