"""
evaluate.py
Evaluation and Baseline Comparison for Fine-Tuned DistilBERT
Candidate: Mahesh Kadam (Roll 51, PRN: 202401120074)
MIT Academy of Engineering (MITAOE) • Department of Data Science

Evaluates the original pretrained checkpoint (untuned head baseline) and the fine-tuned model
on the exact same held-out test split of IMDb movie reviews.
Calculates Accuracy, Precision, Recall, F1, Confusion Matrix, and generates comparison tables and charts.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


class TestDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def run_evaluation(model, dataloader, device):
    """Executes evaluation loop and collects predictions, probabilities, and labels."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds)
            all_probs.extend(probs[:, 1])  # probability of positive class
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    p_bin, r_bin, f1_bin, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'accuracy': float(acc),
        'precision_binary': float(p_bin),
        'recall_binary': float(r_bin),
        'f1_binary': float(f1_bin),
        'precision_macro': float(p_macro),
        'recall_macro': float(r_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(p_weighted),
        'recall_weighted': float(r_weighted),
        'f1_weighted': float(f1_weighted),
        'confusion_matrix': cm.tolist(),
        'tn': int(cm[0, 0]),
        'fp': int(cm[0, 1]),
        'fn': int(cm[1, 0]),
        'tp': int(cm[1, 1]),
        'predictions': all_preds,
        'probabilities': all_probs,
        'true_labels': all_labels
    }


def plot_confusion_matrices(baseline_cm, finetuned_cm, save_path):
    """Plots side-by-side heatmaps of baseline and fine-tuned confusion matrices."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = ['Negative (0)', 'Positive (1)']

    # Baseline CM
    sns.heatmap(baseline_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=labels, yticklabels=labels, cbar=False)
    axes[0].set_title('Baseline DistilBERT (Untuned Head)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=11)
    axes[0].set_ylabel('True Label', fontsize=11)

    # Fine-Tuned CM
    sns.heatmap(finetuned_cm, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=labels, yticklabels=labels, cbar=False)
    axes[1].set_title('Fine-Tuned DistilBERT (IMDb Adapted)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=11)
    axes[1].set_ylabel('True Label', fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved confusion matrix visualization to: {save_path}")


def plot_metrics_comparison(metrics_df, save_path):
    """Plots grouped bar chart comparing performance metrics."""
    plt.figure(figsize=(9, 5))
    bar_width = 0.35
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    # Locate rows robustly
    baseline_row = [idx for idx in metrics_df.index if 'Baseline' in str(idx)][0]
    finetuned_row = [idx for idx in metrics_df.index if 'Fine-Tuned' in str(idx)][0]

    baseline_vals = [
        metrics_df.loc[baseline_row, 'Accuracy'],
        metrics_df.loc[baseline_row, 'Precision'],
        metrics_df.loc[baseline_row, 'Recall'],
        metrics_df.loc[baseline_row, 'F1-Score']
    ]
    finetuned_vals = [
        metrics_df.loc[finetuned_row, 'Accuracy'],
        metrics_df.loc[finetuned_row, 'Precision'],
        metrics_df.loc[finetuned_row, 'Recall'],
        metrics_df.loc[finetuned_row, 'F1-Score']
    ]

    r1 = np.arange(len(metrics))
    r2 = [x + bar_width for x in r1]

    plt.bar(r1, baseline_vals, width=bar_width, color='#6c757d', label='Baseline Pretrained', edgecolor='grey')
    plt.bar(r2, finetuned_vals, width=bar_width, color='#0d6efd', label='Fine-Tuned DistilBERT', edgecolor='grey')

    plt.xlabel('Evaluation Metrics', fontweight='bold', fontsize=11)
    plt.ylabel('Score (0.0 - 1.0)', fontweight='bold', fontsize=11)
    plt.xticks([r + bar_width / 2 for r in range(len(metrics))], metrics, fontsize=11)
    plt.title('Baseline vs Fine-Tuned DistilBERT on Held-Out Test Set', fontsize=13, fontweight='bold')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(frameon=True)

    for i in range(len(metrics)):
        plt.text(r1[i], baseline_vals[i] + 0.02, f"{baseline_vals[i]:.2%}", ha='center', fontsize=9, fontweight='bold')
        plt.text(r2[i], finetuned_vals[i] + 0.02, f"{finetuned_vals[i]:.2%}", ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved metrics comparison chart to: {save_path}")


def plot_training_curves(history_data, save_path):
    """Plots training and validation loss & accuracy progression over epochs."""
    history = history_data.get('history', {})
    epochs = history.get('epoch', [])
    train_loss = history.get('train_loss', [])
    val_loss = history.get('val_loss', [])
    val_acc = history.get('val_accuracy', [])

    if not epochs:
        return

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = '#dc3545'
    ax1.set_xlabel('Epoch', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Loss', color=color, fontweight='bold', fontsize=11)
    l1 = ax1.plot(epochs, train_loss, 'o-', color='#dc3545', label='Train Loss', linewidth=2)
    l2 = ax1.plot(epochs, val_loss, 's--', color='#fd7e14', label='Val Loss', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(epochs)

    ax2 = ax1.twinx()
    color = '#198754'
    ax2.set_ylabel('Validation Accuracy', color=color, fontweight='bold', fontsize=11)
    l3 = ax2.plot(epochs, val_acc, '^-.', color='#198754', label='Val Accuracy', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1.05)

    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', frameon=True)

    plt.title('Fine-Tuning Convergence & Validation Dynamics', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved training curves to: {save_path}")


def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(project_dir, 'dataset')
    results_dir = os.path.join(project_dir, 'results')
    models_dir = os.path.join(project_dir, 'models', 'finetuned_distilbert')

    os.makedirs(results_dir, exist_ok=True)

    # 1. Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[Device] Evaluation running on: {device}")

    # 2. Load held-out test data
    test_csv_path = os.path.join(dataset_dir, "held_out_test.csv")
    if not os.path.exists(test_csv_path):
        print(f"[Error] Held-out test split not found at {test_csv_path}. Please run train.py first.")
        sys.exit(1)

    test_df = pd.read_csv(test_csv_path)
    test_texts = test_df['text'].tolist()
    test_labels = test_df['label'].tolist()
    print(f"[Dataset] Evaluated on {len(test_texts)} held-out IMDb reviews.")

    # 3. Tokenize test set
    model_name = "distilbert/distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)
    test_dataset = TestDataset(test_encodings, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 4. Evaluate Baseline (untrained classification head)
    print("\n--- [1/2] Evaluating Baseline Pretrained Checkpoint ---")
    torch.manual_seed(42)
    baseline_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    baseline_model.to(device)
    baseline_results = run_evaluation(baseline_model, test_loader, device)

    print(f"  Baseline Accuracy:  {baseline_results['accuracy']:.4f} ({baseline_results['accuracy']*100:.2f}%)")
    print(f"  Baseline Precision: {baseline_results['precision_binary']:.4f}")
    print(f"  Baseline Recall:    {baseline_results['recall_binary']:.4f}")
    print(f"  Baseline F1-Score:  {baseline_results['f1_binary']:.4f}")

    # 5. Evaluate Fine-Tuned Model
    print("\n--- [2/2] Evaluating Fine-Tuned DistilBERT Checkpoint ---")
    if not os.path.exists(models_dir):
        print(f"[Error] Fine-tuned model directory not found at {models_dir}. Please run train.py first.")
        sys.exit(1)

    finetuned_model = AutoModelForSequenceClassification.from_pretrained(models_dir)
    finetuned_model.to(device)
    finetuned_results = run_evaluation(finetuned_model, test_loader, device)

    print(f"  Fine-Tuned Accuracy:  {finetuned_results['accuracy']:.4f} ({finetuned_results['accuracy']*100:.2f}%)")
    print(f"  Fine-Tuned Precision: {finetuned_results['precision_binary']:.4f}")
    print(f"  Fine-Tuned Recall:    {finetuned_results['recall_binary']:.4f}")
    print(f"  Fine-Tuned F1-Score:  {finetuned_results['f1_binary']:.4f}")

    # 6. Save Comparison Metrics
    metrics_summary = {
        'test_samples': len(test_texts),
        'baseline': {
            'model_name': 'distilbert-base-uncased (Untuned Head)',
            'accuracy': round(baseline_results['accuracy'], 4),
            'precision': round(baseline_results['precision_binary'], 4),
            'recall': round(baseline_results['recall_binary'], 4),
            'f1_score': round(baseline_results['f1_binary'], 4),
            'confusion_matrix': baseline_results['confusion_matrix'],
            'tn': baseline_results['tn'],
            'fp': baseline_results['fp'],
            'fn': baseline_results['fn'],
            'tp': baseline_results['tp']
        },
        'finetuned': {
            'model_name': 'Fine-Tuned DistilBERT (IMDb Adapted)',
            'accuracy': round(finetuned_results['accuracy'], 4),
            'precision': round(finetuned_results['precision_binary'], 4),
            'recall': round(finetuned_results['recall_binary'], 4),
            'f1_score': round(finetuned_results['f1_binary'], 4),
            'confusion_matrix': finetuned_results['confusion_matrix'],
            'tn': finetuned_results['tn'],
            'fp': finetuned_results['fp'],
            'fn': finetuned_results['fn'],
            'tp': finetuned_results['tp']
        },
        'delta': {
            'accuracy_gain': round(finetuned_results['accuracy'] - baseline_results['accuracy'], 4),
            'f1_gain': round(finetuned_results['f1_binary'] - baseline_results['f1_binary'], 4)
        }
    }

    # Save JSON
    metrics_json_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"\n[Artifact] Saved detailed metrics to: {metrics_json_path}")

    # Save CSV comparison table
    comparison_df = pd.DataFrame({
        'Model': ['Baseline (Untuned Head)', 'Fine-Tuned DistilBERT', 'Absolute Delta (+/-)'],
        'Accuracy': [
            baseline_results['accuracy'],
            finetuned_results['accuracy'],
            finetuned_results['accuracy'] - baseline_results['accuracy']
        ],
        'Precision': [
            baseline_results['precision_binary'],
            finetuned_results['precision_binary'],
            finetuned_results['precision_binary'] - baseline_results['precision_binary']
        ],
        'Recall': [
            baseline_results['recall_binary'],
            finetuned_results['recall_binary'],
            finetuned_results['recall_binary'] - baseline_results['recall_binary']
        ],
        'F1-Score': [
            baseline_results['f1_binary'],
            finetuned_results['f1_binary'],
            finetuned_results['f1_binary'] - baseline_results['f1_binary']
        ]
    })
    comparison_csv_path = os.path.join(results_dir, "metrics_comparison.csv")
    comparison_df.to_csv(comparison_csv_path, index=False)
    print(f"[Artifact] Saved metrics comparison table to: {comparison_csv_path}")

    # 7. Generate Plots
    cm_plot_path = os.path.join(results_dir, "confusion_matrix.png")
    plot_confusion_matrices(
        np.array(baseline_results['confusion_matrix']),
        np.array(finetuned_results['confusion_matrix']),
        cm_plot_path
    )

    plot_df = comparison_df.set_index('Model')
    metrics_bar_path = os.path.join(results_dir, "metrics_comparison.png")
    plot_metrics_comparison(plot_df, metrics_bar_path)

    history_json_path = os.path.join(results_dir, "training_history.json")
    if os.path.exists(history_json_path):
        with open(history_json_path, 'r') as f:
            history_data = json.load(f)
        training_curves_path = os.path.join(results_dir, "training_curves.png")
        plot_training_curves(history_data, training_curves_path)

    print("\n=======================================================")
    print("                EVALUATION BENCHMARK SUMMARY           ")
    print("=======================================================")
    print(comparison_df.to_string(index=False))
    print("=======================================================\n")


if __name__ == '__main__':
    main()
