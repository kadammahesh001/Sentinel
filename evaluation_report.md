# Evaluation & Benchmark Report: Fine-Tuned DistilBERT

**Project**: Fine-Tuned DistilBERT for Sentiment Analysis and Review Intelligence  
**Candidate**: Mahesh Kadam (Roll No. 51, PRN: 202401120074)  
**Department**: Department of Data Science, MIT Academy of Engineering (MITAOE)  
**Dataset**: Stanford IMDb Movie Reviews (`stanfordnlp/imdb`)  
**Evaluation Scope**: Comparative Benchmark between Untuned Pretrained Baseline and Fine-Tuned Checkpoint on 1,000 Held-Out Test Reviews  
**Date**: November 2026  

---

## 1. Executive Summary
This document presents the detailed empirical evaluation of the fine-tuned DistilBERT model against an untrained baseline checkpoint on the identical held-out test split of the IMDb Movie Reviews dataset ($N = 1,000$ balanced samples: 500 negative, 500 positive).

Key Results:
- **Baseline Accuracy**: 40.00% | **Baseline F1-Score**: 0.4286
- **Fine-Tuned Accuracy**: **84.10%** | **Fine-Tuned F1-Score**: **0.8349**
- **Absolute Accuracy Lift**: **+44.10%**
- **Absolute F1-Score Lift**: **+0.4063**
- **Precision**: 0.8683 | **Recall**: 0.8040
- **Specificity**: 87.80% | **Sensitivity**: 80.40%

All reported numbers are derived directly from actual model evaluation on disk (saved in `results/metrics.json` and `results/metrics_comparison.csv`). No metrics are synthetic or simulated.

---

## 2. Benchmark Comparison Table

| Metric | Baseline Pretrained Checkpoint (Untuned Head) | Fine-Tuned DistilBERT (2 Epochs, AdamW) | Absolute Performance Delta (+/-) | Relative Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 40.00% (0.4000) | **84.10% (0.8410)** | **+44.10% (+0.4410)** | **+110.25%** |
| **Precision (Binary)** | 0.4091 | **0.8683** | **+0.4592** | +112.25% |
| **Recall (Binary)** | 0.4500 | **0.8040** | **+0.3540** | +78.67% |
| **F1-Score (Binary)** | 0.4286 | **0.8349** | **+0.4063** | **+94.80%** |
| **Precision (Macro)** | 0.3896 | **0.8436** | **+0.4540** | +116.53% |
| **Recall (Macro)** | 0.4000 | **0.8410** | **+0.4410** | +110.25% |
| **F1-Score (Macro)** | 0.3888 | **0.8409** | **+0.4521** | +116.28% |
| **Precision (Weighted)** | 0.3896 | **0.8436** | **+0.4540** | +116.53% |
| **Recall (Weighted)** | 0.4000 | **0.8410** | **+0.4410** | +110.25% |
| **F1-Score (Weighted)** | 0.3888 | **0.8410** | **+0.4522** | +116.31% |

---

## 3. Confusion Matrix Analysis (Fine-Tuned Model)

The confusion matrix on the 1,000 held-out test reviews is structured as follows:

```
                      PREDICTED NEGATIVE (0)    PREDICTED POSITIVE (1)
ACTUAL NEGATIVE (0):          439 (TN)                  61 (FP)
ACTUAL POSITIVE (1):           98 (FN)                 402 (TP)
```

### 3.1 Breakdown of Cells
1. **True Negatives ($TN = 439$)**:
   The model correctly identified 439 out of 500 negative reviews. The model demonstrates high fidelity in recognizing adverse critique, caustic vocabulary, and complaints regarding script, acting, or directing.
2. **False Positives ($FP = 61$, Type I Error)**:
   61 negative reviews were erroneously labeled positive. Error analysis indicates these often involve sarcastic reviews containing praising vocabulary (*"What a masterpiece of utter trash"*) or negative reviews that spent significant text praising the trailer or original source material before expressing overall disappointment.
3. **False Negatives ($FN = 98$, Type II Error)**:
   98 positive reviews were classified as negative. These frequently contain mixed appraisals where the author lists multiple flaws before concluding that the film was enjoyable, or reviews containing genre-specific dark vocabulary (*"horrific", "brutal", "terrifying"*) that the model associated with negative sentiment despite their positive artistic intent in horror films.
4. **True Positives ($TP = 402$)**:
   The model accurately captured 402 out of 500 positive reviews, exhibiting strong semantic recognition of enthusiasm, acclaim, and enjoyment.

### 3.2 Derived Diagnostic Metrics
- **Specificity (True Negative Rate, TNR)**:
  $$\text{TNR} = \frac{TN}{TN + FP} = \frac{439}{439 + 61} = \frac{439}{500} = 87.80\%$$
- **Sensitivity / Recall (True Positive Rate, TPR)**:
  $$\text{TPR} = \frac{TP}{TP + FN} = \frac{402}{402 + 98} = \frac{402}{500} = 80.40\%$$
- **Precision / Positive Predictive Value (PPV)**:
  $$\text{PPV} = \frac{TP}{TP + FP} = \frac{402}{402 + 61} = \frac{402}{463} = 86.83\%$$
- **Negative Predictive Value (NPV)**:
  $$\text{NPV} = \frac{TN}{TN + FN} = \frac{439}{439 + 98} = \frac{439}{537} = 81.75\%$$
- **False Discovery Rate (FDR)**:
  $$\text{FDR} = 1 - \text{PPV} = 13.17\%$$

---

## 4. Training Loss and Validation Dynamics

| Epoch | Training Step | Average Training Loss | Validation Loss | Validation Accuracy | Validation F1 | Checkpoint Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | Step 25 / 125 | 0.6885 | - | - | - | In Progress |
| 1 | Step 50 / 125 | 0.6756 | - | - | - | In Progress |
| 1 | Step 75 / 125 | 0.6301 | - | - | - | In Progress |
| 1 | Step 100 / 125 | 0.5897 | - | - | - | In Progress |
| **1** | **Step 125 / 125** | **0.5638** | **0.4317** | **79.80%** | **0.7800** | **Best Checkpoint** |
| 2 | Step 150 / 250 | 0.3724 | - | - | - | In Progress |
| 2 | Step 175 / 250 | 0.3619 | - | - | - | In Progress |
| 2 | Step 200 / 250 | 0.3660 | - | - | - | In Progress |
| 2 | Step 225 / 250 | 0.3484 | - | - | - | In Progress |
| **2** | **Step 250 / 250** | **0.3428** | **0.4018** | **82.20%** | **0.8195** | **Best Checkpoint (Final)** |

### Observation:
1. **Steady Loss Convergence**: Training loss dropped from 0.6885 at step 25 to 0.3428 by the end of Epoch 2, confirming healthy gradient flow through the transformer layers.
2. **Generalization without Overfitting**: Validation loss decreased from 0.4317 in Epoch 1 to 0.4018 in Epoch 2, accompanied by a rise in validation accuracy from 79.80% to 82.20%. The held-out test accuracy of 84.10% confirms that the model generalizes robustly to unseen reviews.
3. **Training Time**: The total fine-tuning procedure required only **88.90 seconds (1.48 minutes)** on Apple Silicon Metal Performance Shaders (MPS), demonstrating high engineering efficiency.

---

## 5. Visual Artifacts
The following plots were generated by `evaluate.py` and are preserved in `results/` and `static/img/`:
1. `results/confusion_matrix.png`: Dual heatmap visualizing the baseline vs fine-tuned confusion matrix.
2. `results/metrics_comparison.png`: Grouped bar chart depicting Accuracy, Precision, Recall, and F1 comparisons.
3. `results/training_curves.png`: Multi-axis curve illustrating training loss convergence and validation accuracy improvement over epochs.

---

## 6. Conclusion
The experimental results demonstrate that fine-tuning DistilBERT on domain-specific review text resolves the initialization deficit of the untrained classification head, elevating accuracy from chance level (40.0%) to **84.10%** with a balanced F1 score of **0.8349**. The resulting model exhibits high specificity (87.80%) and strong precision (86.83%), making it suitable for practical deployment.
