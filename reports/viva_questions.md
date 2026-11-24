# Comprehensive Viva Voce Examination Guide
## Fine-Tuned DistilBERT for Sentiment Analysis and Review Intelligence

This document contains 32 comprehensive viva questions and technically rigorous yet easy-to-explain answers, categorized into five core domains:
1. Large Language Models & Transformer Foundations
2. DistilBERT Architecture & Distillation
3. Tokenization & Preprocessing
4. Training Dynamics, Fine-Tuning & Hyperparameters
5. Evaluation Metrics, System Architecture & Practical Deployment

---

### Domain 1: Large Language Models & Transformer Foundations

#### Q1: What is a Large Language Model (LLM), and is DistilBERT considered an LLM?
**Answer:**  
In modern machine learning terminology:
- **Pretrained Language Models (PLMs)**: Encompasses all large transformer-based models (both encoder and decoder architectures) trained on massive text corpora to internalize linguistic structure and semantics. DistilBERT (~66.9M parameters) is an **encoder-based pretrained language model**.
- **Generative Large Language Models (Decoder LLMs)**: Strictly refers to multi-billion parameter autoregressive models (such as GPT-4, Llama 3, or Mistral) trained on causal language modeling (predicting the next token) to generate continuous open-ended text.
- **Academic Distinction**: DistilBERT is **not** a generative text generator. It is a bidirectional encoder foundation model specifically engineered for sequence representation, classification, and discriminative NLP tasks.

#### Q2: What is the fundamental difference between an Encoder-only model (DistilBERT/BERT) and a Decoder-only model (GPT/Llama)?
**Answer:**  
1. **Attention Direction**:
   - **Encoder Models (DistilBERT)**: Use **fully bidirectional self-attention**. Every token can attend to tokens that appear before AND after it simultaneously. This gives superior sentence-level comprehension ideal for classification.
   - **Decoder Models (GPT)**: Use **causal (masked) self-attention**. Tokens can only attend to prior tokens ($t \le i$). This enables autoregressive text generation but introduces directional bias during sequence classification.
2. **Computational Overhead**:
   - DistilBERT has 66M parameters, fits in ~260 MB of memory, runs in milliseconds on commodity CPU/MPS, and yields deterministic classification outputs.
   - Generative LLMs typically range from 0.5B to 70B+ parameters, require gigabytes of VRAM, and generate non-deterministic text requiring complex output parsing.
3. **Application Suitability**:
   - Review classification, sentiment intelligence, and named entity recognition are natively encoder tasks. Generative models are suited for conversational dialogue, summarization, and creative writing.

#### Q2: What is a Transformer, and why does it outperform RNNs and LSTMs?
**Answer:**  
A Transformer is a neural network architecture introduced by Vaswani et al. in 2017 (*"Attention Is All You Need"*) that dispenses with recurrence and convolutions entirely, relying exclusively on **self-attention mechanisms**.  
Transformers outperform Recurrent Neural Networks (RNNs) and LSTMs because:
1. **Parallelization**: RNNs process tokens sequentially step-by-step ($t_1 \rightarrow t_2 \rightarrow \dots$), preventing GPU parallelization across time steps. Transformers process all tokens in a sequence simultaneously.
2. **Elimination of Vanishing Gradients**: Self-attention provides direct, constant-path connections between any two words in a sequence, completely resolving the vanishing gradient problem over long sequences.
3. **Global Receptive Field**: Every token can attend to every other token in the sequence in a single forward pass.

#### Q3: What is the Self-Attention mechanism, and how is it calculated mathematically?
**Answer:**  
Self-attention enables each token in a sentence to dynamically calculate an attention weight with respect to all other tokens, allowing the model to focus on contextually relevant words.  
Mathematically:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
Where:
- $Q$ (Query): Represents what the current token is seeking.
- $K$ (Key): Represents what the target token contains.
- $V$ (Value): The actual information carried by the target token.
- $\sqrt{d_k}$: Scaling factor ($d_k$ is the projection dimension, 64 in DistilBERT) that prevents large dot products from pushing the softmax function into regions with near-zero gradients.

#### Q4: What is the difference between Pretraining and Fine-Tuning?
**Answer:**  
- **Pretraining**: Self-supervised learning on massive unlabelled text corpora (e.g., Wikipedia and BooksCorpus). The model learns general linguistic patterns and semantics via tasks like Masked Language Modeling (MLM). It is computationally expensive, often requiring hundreds of GPU hours.
- **Fine-Tuning**: Supervised adaptation where the pretrained model's weights are further updated on a specific, smaller, labeled task dataset (e.g., binary sentiment on IMDb) using task-specific loss functions (such as Cross-Entropy). It is fast and computationally lightweight.

#### Q5: Why do we use a pretrained model instead of training from scratch?
**Answer:**  
1. **Data Efficiency**: Training transformers from scratch requires millions of labeled examples. Pretrained models possess prior linguistic understanding and require only a fraction of task data.
2. **Computational Feasibility**: Pretraining requires supercomputing clusters; fine-tuning takes mere minutes to hours on a single GPU.
3. **Generalization Performance**: Pretrained representations act as an effective prior, mitigating overfitting on small downstream datasets.

---

### Domain 2: DistilBERT Architecture & Distillation

#### Q6: What is DistilBERT, and how was it created?
**Answer:**  
DistilBERT (*Distilled BERT*) is a lightweight, high-performance language model developed by Hugging Face (Sanh et al., 2019). It was created using **Knowledge Distillation**, where a compact 6-layer "student" network was trained to reproduce the output behaviors and hidden representations of a larger 12-layer "teacher" BERT network. It retains 97% of BERT's language understanding capabilities while being 40% smaller and 60% faster.

#### Q7: How does Knowledge Distillation work during DistilBERT's pretraining?
**Answer:**  
The student network is trained using a composite loss function comprising three terms:
1. **Distillation Loss ($\mathcal{L}_{ce}$)**: Soft cross-entropy between the student's softmax logits and the teacher's temperature-softened probabilities ($p_i \propto \exp(z_i / T)$).
2. **Masked Language Modeling Loss ($\mathcal{L}_{mlm}$)**: Standard hard cross-entropy predicting masked tokens.
3. **Cosine Distance Loss ($\mathcal{L}_{cos}$)**: Minimizes the angle between the student's and teacher's hidden state vectors.

#### Q8: What are the key architectural specifications of DistilBERT?
**Answer:**  
- **Number of Transformer Layers**: 6 encoder layers (half of BERT's 12).
- **Hidden Embedding Size ($d_{\text{model}}$)**: 768.
- **Attention Heads**: 12 heads per layer ($d_k = 768 / 12 = 64$).
- **Intermediate Feed-Forward Dimension**: 3,072.
- **Total Parameters**: ~66.9 Million (BERT-base has ~110 Million).
- **Pooling Mechanism**: Utilizes the final hidden state vector of the `[CLS]` token ($h_{[CLS]} \in \mathbb{R}^{768}$) for sequence classification.

#### Q9: What is a Classification Head, and why is it needed?
**Answer:**  
The base DistilBERT encoder produces 768-dimensional contextual vector representations for tokens, not category predictions. A **Classification Head** is a task-specific neural network appended on top of the encoder's `[CLS]` embedding:
1. A dense linear projection: $\text{Linear}(768 \rightarrow 768)$ with $\text{ReLU}$ activation.
2. A dropout layer ($p = 0.2$) to prevent co-adaptation of features.
3. An output linear layer: $\text{Linear}(768 \rightarrow 2)$ that outputs two unnormalized scalar logits ($z_0, z_1$) corresponding to Negative and Positive classes.

---

### Domain 3: Tokenization & Preprocessing

#### Q10: What is Tokenization, and what tokenizer does DistilBERT use?
**Answer:**  
Tokenization is the process of breaking raw text strings into discrete numerical IDs that a neural network can process. DistilBERT uses the **WordPiece** sub-word tokenization algorithm with a vocabulary of 30,522 tokens.

#### Q11: What is Sub-Word Tokenization, and what problem does it solve?
**Answer:**  
Sub-word tokenization decomposes words into frequent constituent fragments (e.g., *"cinematography"* $\rightarrow$ *"cinema"*, *"##to"*, *"##graphy"*).  
It solves the **Out-of-Vocabulary (OOV)** problem: instead of mapping rare or unknown words to a generic `[UNK]` token, the tokenizer breaks them into known sub-word units, preserving semantic meaning.

#### Q12: What are the special tokens used by DistilBERT?
**Answer:**  
- `[CLS]` (ID 101): Placed at the beginning of every sequence. Its final hidden state serves as the aggregate sequence representation for classification.
- `[SEP]` (ID 102): Placed at the end of a sequence or between paired sentences.
- `[PAD]` (ID 0): Appended to shorter sequences to ensure uniform batch tensor dimensions.
- `[UNK]` (ID 100): Represents genuinely unencodable characters.
- `[MASK]` (ID 103): Used during pretraining for Masked Language Modeling.

#### Q13: Why are `truncation` and `padding` necessary during tokenization?
**Answer:**  
Neural network matrix multiplications require rectangular tensor shapes within each batch. 
- **Padding** fills sequences shorter than the batch target length with `[PAD]` tokens.
- **Truncation** clips reviews exceeding `max_length` (e.g., 128 tokens) to prevent out-of-memory errors on GPU hardware.
- The **Attention Mask** (a binary vector of 1s and 0s) tells the self-attention mechanism to ignore `[PAD]` tokens so they do not distort context.

---

### Domain 4: Training Dynamics, Fine-Tuning & Hyperparameters

#### Q14: What is the role of Epochs, Batch Size, and Learning Rate?
**Answer:**  
- **Epoch**: One complete pass through the entire training dataset. In our project, 2 epochs proved optimal, preventing overfitting while achieving full convergence.
- **Batch Size**: The number of training examples processed before model weights are updated (we used 16). Smaller batches introduce beneficial stochastic noise; larger batches offer stable gradient estimates.
- **Learning Rate**: The step size taken in parameter space during gradient descent ($2 \times 10^{-5}$ in our implementation). Too high causes catastrophic forgetting; too low results in stalled convergence.

#### Q15: Why did you choose the AdamW optimizer instead of standard SGD or Adam?
**Answer:**  
Standard Adam applies $L_2$ regularization directly to the gradient, which inadvertently interacts with the adaptive learning rate scaling and leads to sub-optimal generalization in deep transformers. **AdamW** (Loshchilov & Hutter, 2019) decouples weight decay from the gradient update step:
$$\theta_{t+1} = \theta_t - \eta_t \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t\right)$$
This yields strictly bounded weights and superior generalization on NLP tasks.

#### Q16: What is a Learning Rate Scheduler with Warmup, and why is it used?
**Answer:**  
In fine-tuning, the transformer layers already contain well-tuned representations, but the classification head is randomly initialized. If a high learning rate is applied immediately, large initial gradients from the classification head can destroy the pretrained transformer weights (**catastrophic forgetting**).  
A **Linear Warmup** gradually ramps the learning rate from 0 to $2 \times 10^{-5}$ over the first 10% of steps (warmup), stabilizing the classification head before decaying the learning rate linearly to 0.

#### Q17: What is Overfitting, and how was it prevented in this project?
**Answer:**  
Overfitting occurs when a model memorizes idiosyncrasies and noise in the training set rather than learning generalizable patterns, resulting in low training loss but poor test performance.  
In this project, overfitting was prevented by:
1. Limiting fine-tuning to 2 epochs.
2. Using Dropout ($p = 0.2$) in the classification head.
3. Incorporating AdamW weight decay ($\lambda = 0.01$).
4. Monitoring validation loss at every epoch and retaining the best checkpoint.

#### Q18: What is Data Leakage, and how did your split strategy eliminate it?
**Answer:**  
Data leakage occurs when information from outside the training dataset (such as test set samples or future data) contaminates the training or hyperparameter selection process, leading to falsely optimistic evaluation metrics.  
We eliminated data leakage by:
1. Sampling the **test set ($N = 1,000$) strictly from the official held-out test split** of IMDb.
2. Carving the **validation set ($N = 500$) strictly from the official training split**.
3. Ensuring the model never computed gradients, observed tokens, or evaluated losses on the test split until training was completely finished.

---

### Domain 5: Evaluation Metrics, System Architecture & Practical Deployment

#### Q19: What are Accuracy, Precision, Recall, and F1-Score?
**Answer:**  
- **Accuracy**: Proportion of total correct predictions:
  $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
- **Precision**: Proportion of positive predictions that were actually positive (measures exactness):
  $$\text{Precision} = \frac{TP}{TP + FP}$$
- **Recall (Sensitivity)**: Proportion of actual positive reviews that were correctly identified (measures completeness):
  $$\text{Recall} = \frac{TP}{TP + FN}$$
- **F1-Score**: Harmonic mean of Precision and Recall, balancing both false positives and false negatives:
  $$\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

#### Q20: What were the actual evaluation results of your baseline versus fine-tuned model?
**Answer:**  
Evaluated on the exact same held-out test set ($N = 1,000$):
- **Baseline Checkpoint (Untrained Head)**:
  - Accuracy: **40.00%** | Precision: 0.4091 | Recall: 0.4500 | F1-Score: **0.4286**
- **Fine-Tuned DistilBERT Checkpoint**:
  - Accuracy: **84.10%** | Precision: 0.8683 | Recall: 0.8040 | F1-Score: **0.8349**
- **Empirical Lift**:
  - Accuracy: **+44.10%** absolute improvement (+110% relative lift).
  - F1-Score: **+0.4063** absolute improvement (+94.8% relative lift).

#### Q21: Explain the values in your model's Confusion Matrix.
**Answer:**  
On the 1,000 test reviews:
- **True Negatives (TN)** = 439 (Correct negative predictions; Specificity = 87.80%)
- **False Positives (FP)** = 61 (Type I error; negative reviews misclassified as positive)
- **False Negatives (FN)** = 98 (Type II error; positive reviews misclassified as negative)
- **True Positives (TP)** = 402 (Correct positive predictions; Sensitivity = 80.40%)

#### Q22: How does the web application load and run the trained model?
**Answer:**  
1. When Flask starts, `inference.py` initializes the `SentimentClassifier` singleton class.
2. It detects available hardware acceleration (`torch.backends.mps.is_available()` on Apple Silicon, CUDA on Linux/Windows, or CPU fallback).
3. It loads the tokenizer and fine-tuned weights from `models/finetuned_distilbert/` via `AutoTokenizer.from_pretrained()` and `AutoModelForSequenceClassification.from_pretrained()`.
4. It sets `model.eval()` to disable dropout.
5. Inbound reviews from `/api/predict` or `/api/batch-predict` are tokenized into PyTorch tensors and evaluated inside a `torch.no_grad()` block to eliminate gradient memory overhead.

#### Q23: Why is model confidence score not equivalent to factual truth?
**Answer:**  
The model output is a softmax probability computed over the output logits:
$$\sigma(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$
This score measures **statistical pattern similarity** between the input tokens and the features learned from training data. High confidence means the linguistic structure strongly resembles positive reviews in the training corpus; it does not verify whether the claims or opinions in the review are objectively true in the real world.

#### Q24: What are the main limitations of sentiment analysis using DistilBERT?
**Answer:**  
1. **Context Truncation**: Documents exceeding `max_length` (128 tokens) are truncated, potentially losing trailing conclusions in long reviews.
2. **Sarcasm and Figurative Irony**: Superficial praising phrases used sarcastically can mislead the model unless negative cues are strongly present.
3. **Domain Drift**: A model fine-tuned on movie reviews will experience degraded performance if applied to financial filings or medical critiques without domain re-tuning.
4. **Binary Constraint**: It ignores neutral or moderate ratings (e.g., 3-star reviews).

#### Q25: How does your application handle batch processing of CSV files?
**Answer:**  
The user uploads a CSV file through `/api/batch-predict`. The Flask backend uses Pandas to parse the file, auto-detects candidate text columns (`text`, `review`, `comment`), and feeds the reviews to `predict_batch()`. The inference engine tokenizes batches of 32 reviews simultaneously on GPU/MPS, computes predictions and probabilities in parallel, calculates aggregate metrics (total reviews, positive %, negative %, average confidence), and returns enriched records for Chart.js visualization and CSV re-export.

#### Q26: What is the purpose of Gradient Clipping?
**Answer:**  
Gradient clipping rescales gradient vectors if their $L_2$ norm exceeds a threshold (we set `max_norm = 1.0`). In deep architectures, steep loss surfaces can cause gradients to explode, leading to `NaN` parameter values. Clipping enforces numerical stability.

#### Q27: What is Catastrophic Forgetting, and how was it avoided?
**Answer:**  
Catastrophic forgetting occurs when a neural network overwrites its prior pretrained representations while adapting to a new task. It was avoided by using a conservative learning rate ($2 \times 10^{-5}$), linear warmup, and fine-tuning for only 2 epochs.

#### Q28: What is Cross-Entropy Loss, and how does it optimize classification?
**Answer:**  
Cross-entropy measures the divergence between the true probability distribution $y$ (one-hot vector) and the predicted probability distribution $\hat{y}$:
$$\mathcal{L} = -\sum_{c} y_c \log(\hat{y}_c)$$
Minimizing cross-entropy penalizes confident wrong predictions logarithmically, forcing the model to increase the logit of the ground-truth class.

#### Q29: What is the significance of the `[CLS]` token over average pooling?
**Answer:**  
In BERT-style models, the `[CLS]` token is connected via multi-head self-attention to all tokens across all layers. While average pooling averages token representations uniformly, the `[CLS]` embedding dynamically weights tokens based on self-attention, learning a task-oriented representation for the entire sequence.

#### Q30: What hardware was used, and what was the training throughput?
**Answer:**  
Training was conducted on an Apple Silicon M-series SoC using **PyTorch Metal Performance Shaders (MPS)**. It completed 250 optimization steps across 2,000 samples in **88.90 seconds (1.48 minutes)**, achieving a throughput of approximately 22.5 sequences per second.

#### Q31: What are Aspect-Based Sentiment Analysis (ABSA) and Parameter-Efficient Fine-Tuning (PEFT)?
**Answer:**  
- **ABSA**: Identifies sentiments toward specific sub-components (e.g., in a movie review: *"Acting: Positive"*, *"Screenplay: Negative"*).
- **PEFT / LoRA**: Low-Rank Adaptation freezes the base model weights and inserts small trainable low-rank decomposition matrices into the attention projections, fine-tuning fewer than 1% of parameters while achieving comparable performance.

#### Q32: What ethical considerations are relevant to automated sentiment analysis?
**Answer:**  
1. **Bias Mitigation**: Demographic and linguistic biases present in user-submitted reviews can propagate into automated models.
2. **Assistive Moderation**: Systems should assist human moderators rather than automatically silencing consumer feedback.
3. **Transparent Calibration**: Providing probability scores and explainability ensures users understand model uncertainty.
