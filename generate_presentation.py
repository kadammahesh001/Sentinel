"""
generate_presentation.py
Generates the 12-slide Technical Academic Presentation in PPTX format
using python-pptx.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE


def create_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Color Palette: Deep Academic Navy, Slate, Electric Blue Accent, White
    BG_DARK = RGBColor(11, 15, 25)
    NAVY_CARD = RGBColor(20, 27, 45)
    BLUE_ACCENT = RGBColor(59, 130, 246)
    TEXT_LIGHT = RGBColor(243, 244, 246)
    TEXT_MUTED = RGBColor(156, 163, 175)
    GREEN_ACCENT = RGBColor(16, 185, 129)

    blank_layout = prs.slide_layouts[6]

    def set_slide_background(slide):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = BG_DARK
        bg.line.fill.background()
        return bg

    def add_header(slide, title_text, category="COLLEGE ACADEMIC PROJECT PRESENTATION"):
        # Category tag
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11), Inches(0.3))
        tf_cat = cat_box.text_frame
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category.upper()
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = BLUE_ACCENT

        # Title
        t_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.65), Inches(11.5), Inches(0.8))
        tf = t_box.text_frame
        p = tf.paragraphs[0]
        p.text = title_text
        p.font.size = Pt(24)
        p.font.bold = True
        p.font.color.rgb = TEXT_LIGHT

    # ================= SLIDE 1: TITLE SLIDE =================
    slide1 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide1)

    # Accent decorative box
    acc = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.5), Inches(11.7), Inches(0.08))
    acc.fill.solid()
    acc.fill.fore_color.rgb = BLUE_ACCENT
    acc.line.fill.background()

    title_box = slide1.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(2.2))
    tf1 = title_box.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "Fine-Tuned DistilBERT for Sentiment Analysis and Review Intelligence"
    p1.font.size = Pt(36)
    p1.font.bold = True
    p1.font.color.rgb = TEXT_LIGHT

    sub_p = tf1.add_paragraph()
    sub_p.text = "Pretrained Transformer Adaptation, Evaluation Benchmarking, and Full-Stack Web Deployment"
    sub_p.font.size = Pt(18)
    sub_p.font.color.rgb = BLUE_ACCENT
    sub_p.space_before = Pt(12)

    # Student Details Card
    meta_card = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(4.3), Inches(11.7), Inches(2.2))
    meta_card.fill.solid()
    meta_card.fill.fore_color.rgb = NAVY_CARD
    meta_card.line.color.rgb = BLUE_ACCENT

    meta_box = slide1.shapes.add_textbox(Inches(1.2), Inches(4.5), Inches(11), Inches(1.8))
    tf_meta = meta_box.text_frame
    fields = [
        ("Student Name:", "Mahesh Kadam"),
        ("Roll Number / Candidate ID:", "Roll No. 51 | PRN: 202401120074"),
        ("Department & Specialization:", "Department of Data Science"),
        ("Institution / University:", "MIT Academy of Engineering (MITAOE)"),
        ("Academic Year & Guide:", "Academic Year 2026-2027 | Department of Data Science")
    ]
    for label, val in fields:
        p = tf_meta.add_paragraph()
        run1 = p.add_run()
        run1.text = f"{label:30} "
        run1.font.bold = True
        run1.font.size = Pt(13)
        run1.font.color.rgb = BLUE_ACCENT
        run2 = p.add_run()
        run2.text = val
        run2.font.size = Pt(13)
        run2.font.color.rgb = TEXT_LIGHT

    # ================= SLIDE 2: PROBLEM STATEMENT =================
    slide2 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide2)
    add_header(slide2, "1. Problem Statement & Motivation")

    card_w = Inches(5.6)
    card_h = Inches(5.3)

    # Card 1: Challenges
    c1 = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), card_w, card_h)
    c1.fill.solid()
    c1.fill.fore_color.rgb = NAVY_CARD
    c1.line.color.rgb = RGBColor(40, 50, 75)

    tb1 = slide2.shapes.add_textbox(Inches(1.1), Inches(1.8), card_w - Inches(0.6), card_h - Inches(0.4))
    tf_c1 = tb1.text_frame
    tf_c1.word_wrap = True
    p = tf_c1.paragraphs[0]
    p.text = "Limitations of Traditional Approaches"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT

    points1 = [
        "Bag-of-Words & TF-IDF ignore syntax, polysemy, and long-range semantic dependencies.",
        "External black-box LLM APIs introduce recurring token costs, non-deterministic latency, and privacy compliance risks.",
        "Zero-shot base transformers lack domain-specific calibration on nuanced movie critique jargon.",
        "Enterprises need localized, low-latency, interpretable sentiment intelligence models."
    ]
    for pt in points1:
        p = tf_c1.add_paragraph()
        p.text = f"•  {pt}"
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_LIGHT
        p.space_before = Pt(14)

    # Card 2: Proposed Solution
    c2 = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.9), Inches(1.6), card_w, card_h)
    c2.fill.solid()
    c2.fill.fore_color.rgb = NAVY_CARD
    c2.line.color.rgb = RGBColor(40, 50, 75)

    tb2 = slide2.shapes.add_textbox(Inches(7.2), Inches(1.8), card_w - Inches(0.6), card_h - Inches(0.4))
    tf_c2 = tb2.text_frame
    tf_c2.word_wrap = True
    p = tf_c2.paragraphs[0]
    p.text = "The DistilBERT Fine-Tuned Paradigm"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = GREEN_ACCENT

    points2 = [
        "Knowledge Distillation: Retains 97% of BERT's contextual understanding at 60% faster inference.",
        "Domain Adaptation: Fine-tunes all transformer layers on IMDb movie review syntax.",
        "Edge / Local Hardware Efficiency: Runs efficiently on local GPUs (Apple Silicon MPS / CUDA).",
        "Deterministic Probability Output: Softmax calibrated probabilities with token sequence analysis."
    ]
    for pt in points2:
        p = tf_c2.add_paragraph()
        p.text = f"•  {pt}"
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_LIGHT
        p.space_before = Pt(14)

    # ================= SLIDE 3: PROJECT OBJECTIVES =================
    slide3 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide3)
    add_header(slide3, "2. Project Scope & Technical Objectives")

    boxes = [
        ("1. Pretrained Model Adaptation", "Fine-tune 'distilbert-base-uncased' using PyTorch and Hugging Face Transformers for sequence classification."),
        ("2. Leakage-Free Dataset Pipeline", "Ingest stanfordnlp/imdb, creating isolated train/val/test splits to guarantee zero data leakage during evaluation."),
        ("3. Baseline Empirical Benchmark", "Evaluate untuned checkpoint vs fine-tuned model on identical held-out test split (1,000 samples) across Accuracy, Precision, Recall, and F1."),
        ("4. Interactive Full-Stack System", "Engineer a responsive Flask web platform with Chart.js analytics, batch CSV analysis, real-time confidence gauges, and export capabilities."),
        ("5. Academic Rigor & Deliverables", "Generate technical reports, presentation decks, Jupyter Colab notebook, and 30+ viva examination questions.")
    ]

    y_pos = 1.6
    for title, desc in boxes:
        card = slide3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(y_pos), Inches(11.7), Inches(0.95))
        card.fill.solid()
        card.fill.fore_color.rgb = NAVY_CARD
        card.line.color.rgb = BLUE_ACCENT

        tb = slide3.shapes.add_textbox(Inches(1.0), Inches(y_pos + 0.08), Inches(11.3), Inches(0.8))
        tf = tb.text_frame
        p = tf.paragraphs[0]
        run1 = p.add_run()
        run1.text = title + "  —  "
        run1.font.bold = True
        run1.font.size = Pt(14)
        run1.font.color.rgb = BLUE_ACCENT
        run2 = p.add_run()
        run2.text = desc
        run2.font.size = Pt(13)
        run2.font.color.rgb = TEXT_LIGHT

        y_pos += 1.05

    # ================= SLIDE 4: LLM & TRANSFORMER FUNDAMENTALS =================
    slide4 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide4)
    add_header(slide4, "3. LLM & Transformer Architectural Foundations")

    col_w = Inches(3.64)
    col_h = Inches(5.3)

    pillars = [
        ("Self-Attention Mechanism", "Computes pairwise token relationships using Queries, Keys, and Values:\n\nAttention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V\n\nAllows every word to dynamically attend to every other word regardless of positional distance.", BLUE_ACCENT),
        ("Encoder PLM vs. Generative LLM", "DistilBERT is a Bidirectional Encoder Foundation Model (~66.9M params). Captures full-sentence context via pooled [CLS] token, optimal for deterministic classification.\n\nGenerative LLMs (GPT/Llama) are autoregressive decoders; heavy for simple classification.", GREEN_ACCENT),
        ("Sub-Word Tokenization", "DistilBERT uses WordPiece tokenization (vocabulary: 30,522 tokens).\n\nDeconstructs unseen vocabulary into frequent sub-word tokens (e.g., 'breathtaking' -> 'breath', '##taking'), eliminating Out-of-Vocabulary (OOV) errors.", RGBColor(168, 85, 247))
    ]

    x_pos = 0.8
    for heading, text, col_color in pillars:
        c = slide4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x_pos), Inches(1.6), col_w, col_h)
        c.fill.solid()
        c.fill.fore_color.rgb = NAVY_CARD
        c.line.color.rgb = col_color

        tb = slide4.shapes.add_textbox(Inches(x_pos + 0.2), Inches(1.8), col_w - Inches(0.4), col_h - Inches(0.4))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = heading
        p.font.size = Pt(16)
        p.font.bold = True
        p.font.color.rgb = col_color

        p_desc = tf.add_paragraph()
        p_desc.text = text
        p_desc.font.size = Pt(13)
        p_desc.font.color.rgb = TEXT_LIGHT
        p_desc.space_before = Pt(14)

        x_pos += 4.03

    # ================= SLIDE 5: DATASET & SPLIT STRATEGY =================
    slide5 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide5)
    add_header(slide5, "4. Dataset Ingestion & Split Strategy")

    c_left = slide5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.3))
    c_left.fill.solid()
    c_left.fill.fore_color.rgb = NAVY_CARD
    c_left.line.color.rgb = RGBColor(40, 50, 75)

    tb_l = slide5.shapes.add_textbox(Inches(1.1), Inches(1.8), Inches(5.0), Inches(4.8))
    tf_l = tb_l.text_frame
    tf_l.word_wrap = True
    p = tf_l.paragraphs[0]
    p.text = "Stanford IMDb Movie Reviews Dataset"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT

    dataset_bullets = [
        "Source: Hugging Face Datasets ('stanfordnlp/imdb').",
        "Total Volume: 50,000 highly polarized movie reviews.",
        "Labels: Binary classification (0 = Negative, 1 = Positive).",
        "Class Balance: Exactly 50% positive and 50% negative.",
        "Linguistic Nuance: Rich in sarcasm, double negatives, genre jargon, and complex narrative critique."
    ]
    for b in dataset_bullets:
        p = tf_l.add_paragraph()
        p.text = f"•  {b}"
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_LIGHT
        p.space_before = Pt(12)

    # Right Card: Split Strategy
    c_right = slide5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.9), Inches(1.6), Inches(5.6), Inches(5.3))
    c_right.fill.solid()
    c_right.fill.fore_color.rgb = NAVY_CARD
    c_right.line.color.rgb = RGBColor(40, 50, 75)

    tb_r = slide5.shapes.add_textbox(Inches(7.2), Inches(1.8), Inches(5.0), Inches(4.8))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True
    p = tf_r.paragraphs[0]
    p.text = "Zero Data Leakage Split Protocol"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = GREEN_ACCENT

    splits = [
        ("Training Set (2,000 samples)", "Stratified 50/50 balanced subset sampled strictly from the official training split."),
        ("Validation Set (500 samples)", "Stratified from remaining training partition using fixed random seed (42) to tune early stopping."),
        ("Held-Out Test Set (1,000 samples)", "Strictly isolated from the official test split. Never exposed to optimizer, tokenizer vocabulary, or loss calculation."),
        ("Seed Determinism", "Seed 42 applied to Python, NumPy, PyTorch, and MPS/CUDA runtime.")
    ]
    for s_name, s_desc in splits:
        p = tf_r.add_paragraph()
        run1 = p.add_run()
        run1.text = f"•  {s_name}: "
        run1.font.bold = True
        run1.font.size = Pt(13)
        run1.font.color.rgb = GREEN_ACCENT
        run2 = p.add_run()
        run2.text = s_desc
        run2.font.size = Pt(12)
        run2.font.color.rgb = TEXT_LIGHT
        p.space_before = Pt(10)

    # ================= SLIDE 6: DISTILBERT MODEL ARCHITECTURE =================
    slide6 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide6)
    add_header(slide6, "5. DistilBERT Architecture & Sequence Classification Head")

    card_arch = slide6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.3))
    card_arch.fill.solid()
    card_arch.fill.fore_color.rgb = NAVY_CARD
    card_arch.line.color.rgb = BLUE_ACCENT

    tb_arch = slide6.shapes.add_textbox(Inches(1.2), Inches(1.8), Inches(11.0), Inches(4.8))
    tf_a = tb_arch.text_frame
    tf_a.word_wrap = True
    p = tf_a.paragraphs[0]
    p.text = "Model Specifications & Sequence Classification Head"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT

    specs = [
        ("Base Backbone", "distilbert/distilbert-base-uncased (Knowledge-distilled from BERT-base)."),
        ("Layer Configuration", "6 Transformer Encoder blocks (BERT has 12), reducing latency by ~60%."),
        ("Hidden Dimension", "768-dimensional token embeddings across 12 Multi-Head Attention heads."),
        ("Total Parameters", "66,955,010 total parameters (vs. 110M in BERT-base)."),
        ("Sequence Pooling", "Uses the [CLS] first token hidden vector h_[CLS] in R^768 as the sequence summary."),
        ("Classification Head", "Linear(768 -> 768) -> ReLU -> Dropout(p=0.2) -> Linear(768 -> 2 classes)."),
        ("Output Formulation", "Softmax applied over binary logits [z_neg, z_pos] to yield calibrated probabilities.")
    ]
    for s_title, s_detail in specs:
        p = tf_a.add_paragraph()
        run1 = p.add_run()
        run1.text = f"•  {s_title:24}: "
        run1.font.bold = True
        run1.font.size = Pt(13)
        run1.font.color.rgb = BLUE_ACCENT
        run2 = p.add_run()
        run2.text = s_detail
        run2.font.size = Pt(13)
        run2.font.color.rgb = TEXT_LIGHT
        p.space_before = Pt(8)

    # ================= SLIDE 7: FINE-TUNING METHODOLOGY =================
    slide7 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide7)
    add_header(slide7, "6. Fine-Tuning Methodology & Hyperparameters")

    # 4 grid cards
    hp_cards = [
        ("Optimization Protocol", [
            "Optimizer: AdamW (Weight Decay: 0.01)",
            "Base Learning Rate: 2e-5",
            "Learning Rate Scheduler: Linear Warmup with Linear Decay",
            "Warmup Ratio: 10% of total training steps (25 steps)"
        ], BLUE_ACCENT),
        ("Batch & Sequence Configuration", [
            "Batch Size: 16 sequences per step",
            "Max Sequence Length: 128 tokens",
            "Truncation: Enabled (Head-end preserved)",
            "Padding: Dynamic batch padding to max length in mini-batch"
        ], GREEN_ACCENT),
        ("Loss Function & Gradient Clipping", [
            "Loss Function: Cross-Entropy Loss",
            "Gradient Clipping: max_norm = 1.0 (prevents exploding gradients)",
            "Epochs: 2 complete passes over 2,000 training reviews",
            "Total Optimization Steps: 250 steps"
        ], RGBColor(245, 158, 11)),
        ("Compute Infrastructure", [
            "Compute Hardware: Apple Silicon Metal Performance Shaders (MPS)",
            "Total Fine-Tuning Duration: 88.90 seconds (1.48 minutes)",
            "Throughput: ~22.5 sequences / second",
            "Checkpointing: Best validation F1 model saved to disk"
        ], RGBColor(168, 85, 247))
    ]

    coords = [
        (0.8, 1.6), (6.9, 1.6),
        (0.8, 4.3), (6.9, 4.3)
    ]

    for (title, items, border_col), (x, y) in zip(hp_cards, coords):
        c = slide7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(5.6), Inches(2.6))
        c.fill.solid()
        c.fill.fore_color.rgb = NAVY_CARD
        c.line.color.rgb = border_col

        tb = slide7.shapes.add_textbox(Inches(x + 0.2), Inches(y + 0.15), Inches(5.2), Inches(2.3))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(15)
        p.font.bold = True
        p.font.color.rgb = border_col

        for it in items:
            p_it = tf.add_paragraph()
            p_it.text = f"• {it}"
            p_it.font.size = Pt(11)
            p_it.font.color.rgb = TEXT_LIGHT
            p_it.space_before = Pt(4)

    # ================= SLIDE 8: SYSTEM ARCHITECTURE =================
    slide8 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide8)
    add_header(slide8, "7. End-to-End System Architecture")

    card_sys = slide8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.3))
    card_sys.fill.solid()
    card_sys.fill.fore_color.rgb = NAVY_CARD
    card_sys.line.color.rgb = BLUE_ACCENT

    tb_sys = slide8.shapes.add_textbox(Inches(1.2), Inches(1.8), Inches(11.0), Inches(4.8))
    tf_s = tb_sys.text_frame
    tf_s.word_wrap = True
    p = tf_s.paragraphs[0]
    p.text = "Modular Full-Stack System Pipeline"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT

    pipeline_steps = [
        ("Layer 1: Model & Training Layer (train.py)", "Loads dataset from Hugging Face, performs stratified tokenization, fine-tunes DistilBERT with AdamW scheduler, and serializes model/tokenizer weights to models/finetuned_distilbert/."),
        ("Layer 2: Evaluation & Benchmarking (evaluate.py)", "Executes rigorous side-by-side evaluation of baseline pretrained checkpoint vs fine-tuned model on 1,000 held-out test reviews, exporting CSV, JSON metrics, and confusion matrix plots."),
        ("Layer 3: Production Inference Engine (inference.py)", "Encapsulates SentimentClassifier class for thread-safe single and vectorized batch inference, calculating softmax confidence, class probabilities, and intensity calibration."),
        ("Layer 4: Application Server (app.py)", "Flask REST backend exposing /api/predict (single review inference), /api/batch-predict (CSV upload & dataframe analytics), /api/metrics, and /api/health."),
        ("Layer 5: Responsive Analytics Dashboard (UI)", "HTML5 + Bootstrap 5 + Chart.js dashboard providing single-review analysis, batch CSV processing with live donut & bar charts, session history ledger, and model comparison cards.")
    ]
    for s_title, s_detail in pipeline_steps:
        p = tf_s.add_paragraph()
        run1 = p.add_run()
        run1.text = f"{s_title}: "
        run1.font.bold = True
        run1.font.size = Pt(13)
        run1.font.color.rgb = GREEN_ACCENT
        run2 = p.add_run()
        run2.text = s_detail
        run2.font.size = Pt(12)
        run2.font.color.rgb = TEXT_LIGHT
        p.space_before = Pt(8)

    # ================= SLIDE 9: ACTUAL EMPIRICAL RESULTS =================
    slide9 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide9)
    add_header(slide9, "8. Actual Empirical Results (Held-Out Test Set N=1,000)")

    # Left: Results Table
    c_tbl = slide9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(6.8), Inches(5.3))
    c_tbl.fill.solid()
    c_tbl.fill.fore_color.rgb = NAVY_CARD
    c_tbl.line.color.rgb = BLUE_ACCENT

    tb_t = slide9.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(6.4), Inches(4.8))
    tf_t = tb_t.text_frame
    tf_t.word_wrap = True
    p = tf_t.paragraphs[0]
    p.text = "Benchmark Comparison: Real Test Results"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT

    res_lines = [
        "Metric                 | Baseline Checkpoint | Fine-Tuned DistilBERT | Absolute Delta",
        "--------------------------------------------------------------------------------",
        "Accuracy               | 40.00%              | 84.10%                | +44.10% LIFT",
        "Precision (Binary)     | 0.4091              | 0.8683                | +0.4592",
        "Recall (Binary)        | 0.4500              | 0.8040                | +0.3540",
        "F1-Score (Binary)      | 0.4286              | 0.8349                | +0.4063 LIFT",
        "Loss Convergence       | 0.6885 (Initial)    | 0.3428 (Epoch 2)      | -0.3457 drop",
        "Validation Accuracy    | 50.00% (Random)     | 82.20% (Epoch 2)      | +32.20%"
    ]
    for r in res_lines:
        p = tf_t.add_paragraph()
        p.text = r
        p.font.size = Pt(11)
        p.font.name = "Courier New"
        p.font.color.rgb = GREEN_ACCENT if "+44" in r or "84" in r else TEXT_LIGHT
        p.space_before = Pt(5)

    # Right: Confusion Matrix Breakdown
    c_cm = slide9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.9), Inches(1.6), Inches(4.6), Inches(5.3))
    c_cm.fill.solid()
    c_cm.fill.fore_color.rgb = NAVY_CARD
    c_cm.line.color.rgb = GREEN_ACCENT

    tb_cm = slide9.shapes.add_textbox(Inches(8.1), Inches(1.8), Inches(4.2), Inches(4.8))
    tf_cm = tb_cm.text_frame
    tf_cm.word_wrap = True
    p = tf_cm.paragraphs[0]
    p.text = "Confusion Matrix Breakdown"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = GREEN_ACCENT

    cm_metrics = [
        ("True Negatives (TN):", "439 reviews (87.8% specificity)"),
        ("False Positives (FP):", "61 reviews (Type I error)"),
        ("False Negatives (FN):", "98 reviews (Type II error)"),
        ("True Positives (TP):", "402 reviews (80.4% sensitivity)"),
        ("Macro F1-Score:", "0.8409"),
        ("Weighted F1-Score:", "0.8410")
    ]
    for k, v in cm_metrics:
        p = tf_cm.add_paragraph()
        run1 = p.add_run()
        run1.text = f"{k} "
        run1.font.bold = True
        run1.font.size = Pt(12)
        run1.font.color.rgb = BLUE_ACCENT
        run2 = p.add_run()
        run2.text = v
        run2.font.size = Pt(12)
        run2.font.color.rgb = TEXT_LIGHT
        p.space_before = Pt(8)

    # ================= SLIDE 10: WEB APPLICATION & DEMONSTRATION =================
    slide10 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide10)
    add_header(slide10, "9. Web Application Features & Viva Demonstration")

    card_app = slide10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.3))
    card_app.fill.solid()
    card_app.fill.fore_color.rgb = NAVY_CARD
    card_app.line.color.rgb = BLUE_ACCENT

    tb_a = slide10.shapes.add_textbox(Inches(1.2), Inches(1.8), Inches(11.0), Inches(4.8))
    tf_app = tb_a.text_frame
    tf_app.word_wrap = True
    p = tf_app.paragraphs[0]
    p.text = "Interactive Dashboard Capabilities (Flask + Chart.js)"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT

    demo_features = [
        ("Single Review Inference", "Instant tokenization and sentiment classification with real-time softmax confidence gauge and probability breakdown."),
        ("Pre-Baked Sample Buttons", "One-click injection of diverse reviews (Masterpiece, Disaster, Balanced, Sarcastic) for live viva demonstration."),
        ("Batch CSV Ingestion", "Upload any review dataset. Auto-detects text columns, processes in batches, and renders KPI totals, positive/negative percentages, and average confidence."),
        ("Interactive Chart.js Visualizations", "Live Donut chart displaying sentiment polarity split and Bar chart showing confidence score distribution bins."),
        ("Enriched CSV Export", "Allows downloading predicted labels, confidence scores, and probability splits directly from the browser."),
        ("Session History Ledger", "Maintains an active ledger of queries analyzed during the viva session with re-load capabilities.")
    ]
    for f_title, f_desc in demo_features:
        p = tf_app.add_paragraph()
        run1 = p.add_run()
        run1.text = f"•  {f_title}: "
        run1.font.bold = True
        run1.font.size = Pt(13)
        run1.font.color.rgb = BLUE_ACCENT
        run2 = p.add_run()
        run2.text = f_desc
        run2.font.size = Pt(13)
        run2.font.color.rgb = TEXT_LIGHT
        p.space_before = Pt(8)

    # ================= SLIDE 11: LIMITATIONS & ETHICAL CONSIDERATIONS =================
    slide11 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide11)
    add_header(slide11, "10. Limitations & Ethical Considerations")

    c11_l = slide11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.3))
    c11_l.fill.solid()
    c11_l.fill.fore_color.rgb = NAVY_CARD
    c11_l.line.color.rgb = RGBColor(239, 68, 68)

    tb11_l = slide11.shapes.add_textbox(Inches(1.1), Inches(1.8), Inches(5.0), Inches(4.8))
    tf11_l = tb11_l.text_frame
    tf11_l.word_wrap = True
    p = tf11_l.paragraphs[0]
    p.text = "Technical Limitations"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = RGBColor(239, 68, 68)

    limitations = [
        "Sequence Truncation: Fixed max sequence length (128 tokens) truncates lengthy essays, potentially missing late-stage critiques.",
        "Sarcasm & Subtle Irony: Complex figurative language without explicit negative keywords can result in misclassification.",
        "Domain Specificity: Fine-tuned on movie reviews; transfer to financial or medical sentiment requires domain adaptation.",
        "Architectural Scope Note: DistilBERT is an encoder foundation model. If generative LLM is strictly demanded, PEFT/LoRA on Qwen2.5-0.5B is proposed."
    ]
    for lim in limitations:
        p = tf11_l.add_paragraph()
        p.text = f"• {lim}"
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_LIGHT
        p.space_before = Pt(12)

    c11_r = slide11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.9), Inches(1.6), Inches(5.6), Inches(5.3))
    c11_r.fill.solid()
    c11_r.fill.fore_color.rgb = NAVY_CARD
    c11_r.line.color.rgb = RGBColor(245, 158, 11)

    tb11_r = slide11.shapes.add_textbox(Inches(7.2), Inches(1.8), Inches(5.0), Inches(4.8))
    tf11_r = tb11_r.text_frame
    tf11_r.word_wrap = True
    p = tf11_r.paragraphs[0]
    p.text = "Ethical Considerations & AI Calibration"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = RGBColor(245, 158, 11)

    ethics = [
        "Confidence != Factual Truth: High softmax probability reflects statistical pattern alignment, not epistemological truth.",
        "Dataset Bias: IMDb reviews originate from specific demographics and may reflect socio-cultural reviewer biases.",
        "Automated Moderation Risks: Automated sentiment scoring should assist human moderators rather than autonomously silencing reviews.",
        "Explainability Requirement: Providing class probabilities and token counts ensures greater user transparency."
    ]
    for eth in ethics:
        p = tf11_r.add_paragraph()
        p.text = f"• {eth}"
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_LIGHT
        p.space_before = Pt(12)

    # ================= SLIDE 12: CONCLUSION & REFERENCES =================
    slide12 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide12)
    add_header(slide12, "11. Conclusion, Future Scope & References")

    card_c = slide12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.3))
    card_c.fill.solid()
    card_c.fill.fore_color.rgb = NAVY_CARD
    card_c.line.color.rgb = GREEN_ACCENT

    tb_c = slide12.shapes.add_textbox(Inches(1.2), Inches(1.8), Inches(11.0), Inches(4.8))
    tf_c = tb_c.text_frame
    tf_c.word_wrap = True
    p = tf_c.paragraphs[0]
    p.text = "Key Takeaways & Academic References"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = GREEN_ACCENT

    conclusions = [
        "Demonstrated successful adaptation of DistilBERT for sentiment classification, elevating accuracy from 40.0% to 84.10% (+44.1% lift) with F1 of 0.8349.",
        "Enforced strict data partitioning to prevent data leakage, evaluating on 1,000 held-out test reviews with actual, non-fabricated metrics.",
        "Engineered and verified a full-stack Flask web application with real-time confidence gauges, batch CSV intelligence, and Chart.js dashboards.",
        "Future Scope: Aspect-Based Sentiment Analysis (ABSA), multi-class fine-tuning (5-star rating scale), and PEFT/LoRA parameter-efficient adaptation."
    ]
    for conc in conclusions:
        p = tf_c.add_paragraph()
        p.text = f"✔  {conc}"
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_LIGHT
        p.space_before = Pt(8)

    p_ref = tf_c.add_paragraph()
    p_ref.text = "\nAcademic References:"
    p_ref.font.bold = True
    p_ref.font.size = Pt(14)
    p_ref.font.color.rgb = BLUE_ACCENT
    p_ref.space_before = Pt(10)

    refs = [
        "1. Sanh, V., et al. (2019). 'DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.' arXiv:1910.01108.",
        "2. Vaswani, A., et al. (2017). 'Attention Is All You Need.' Advances in Neural Information Processing Systems (NeurIPS).",
        "3. Maas, A. L., et al. (2011). 'Learning Word Vectors for Sentiment Analysis.' Proceedings of ACL: Human Language Technologies."
    ]
    for ref in refs:
        p = tf_c.add_paragraph()
        p.text = ref
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_MUTED
        p.space_before = Pt(4)

    # Save Presentation
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "technical_presentation.pptx")
    prs.save(output_path)
    print(f"[Success] Generated technical presentation at: {output_path}")


if __name__ == '__main__':
    create_deck()
