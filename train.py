"""
train.py
Fine-Tuned DistilBERT for Sentiment Analysis and Review Intelligence
Candidate: Mahesh Kadam (Roll 51, PRN: 202401120074)
MIT Academy of Engineering (MITAOE) • Department of Data Science

This script fine-tunes a pretrained DistilBERT model ('distilbert/distilbert-base-uncased')
on the IMDb Movie Reviews dataset (stanfordnlp/imdb) using PyTorch and Hugging Face Transformers.
Includes split strategy to prevent data leakage, training history recording, and model checkpointing.
"""

import os
import sys
import time
import json
import argparse
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class IMDbDataset(Dataset):
    """PyTorch Dataset wrapper for tokenized IMDb reviews."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def prepare_balanced_subset(dataset, n_samples_per_class=1000, seed=42):
    """
    Creates a balanced subset containing equal numbers of positive and negative reviews.
    Prevents class imbalance during training and evaluation.
    """
    df = pd.DataFrame(dataset)
    pos_df = df[df['label'] == 1].sample(n=n_samples_per_class, random_state=seed)
    neg_df = df[df['label'] == 0].sample(n=n_samples_per_class, random_state=seed)
    balanced_df = pd.concat([pos_df, neg_df]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return balanced_df['text'].tolist(), balanced_df['label'].tolist()


def evaluate_model(model, dataloader, device):
    """Calculates loss and accuracy on validation or test set."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(len(dataloader), 1)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)

    return {
        'loss': avg_loss,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def train_model(args):
    set_seed(args.seed)
    project_dir = os.path.dirname(os.path.abspath(__file__))
    model_save_dir = os.path.join(project_dir, args.save_dir)
    results_dir = os.path.join(project_dir, args.results_dir)
    dataset_dir = os.path.join(project_dir, 'dataset')

    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    # 1. Device selection
    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[Device] Using compute device: {device}")

    # 2. Load dataset
    print("[Dataset] Loading stanfordnlp/imdb from Hugging Face...")
    raw_dataset = load_dataset("stanfordnlp/imdb")

    train_full = raw_dataset["train"]
    test_full = raw_dataset["test"]
    print(f"[Dataset] Full official train count: {len(train_full)}, test count: {len(test_full)}")

    # 3. Stratified Train / Validation / Test split strategy
    # To prevent data leakage:
    # - Test set is strictly held-out from the official 'test' split.
    # - Validation set is partitioned strictly from the official 'train' split.
    print(f"[Sampling] Selecting reproducible subset (Train: {args.train_samples}, Val: {args.val_samples}, Test: {args.test_samples})...")
    
    half_train = args.train_samples // 2
    half_val = args.val_samples // 2
    half_test = args.test_samples // 2

    # Partition training split into pool
    train_df = pd.DataFrame(train_full)
    pos_train_pool = train_df[train_df['label'] == 1].sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    neg_train_pool = train_df[train_df['label'] == 0].sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # Validation from train split
    val_pos = pos_train_pool.iloc[:half_val]
    val_neg = neg_train_pool.iloc[:half_val]
    val_df = pd.concat([val_pos, val_neg]).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # Train from remaining train split (no overlap with validation!)
    train_pos = pos_train_pool.iloc[half_val:half_val + half_train]
    train_neg = neg_train_pool.iloc[half_val:half_val + half_train]
    train_df_subset = pd.concat([train_pos, train_neg]).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # Held-out Test strictly from official test split
    test_texts, test_labels = prepare_balanced_subset(test_full, n_samples_per_class=half_test, seed=args.seed)

    train_texts, train_labels = train_df_subset['text'].tolist(), train_df_subset['label'].tolist()
    val_texts, val_labels = val_df['text'].tolist(), val_df['label'].tolist()

    print(f"[Dataset Partition] Train: {len(train_texts)} | Val: {len(val_texts)} | Held-Out Test: {len(test_texts)}")

    # Save a sample test CSV for batch upload demo
    sample_csv_path = os.path.join(dataset_dir, "sample_test_reviews.csv")
    sample_demo_df = pd.DataFrame({
        'review_id': range(1, 51),
        'text': test_texts[:50],
        'ground_truth': ['Positive' if l == 1 else 'Negative' for l in test_labels[:50]]
    })
    sample_demo_df.to_csv(sample_csv_path, index=False)
    print(f"[Artifact] Saved demo sample CSV with 50 reviews to {sample_csv_path}")

    # Also save the exact held-out test split for evaluate.py
    held_out_test_path = os.path.join(dataset_dir, "held_out_test.csv")
    pd.DataFrame({'text': test_texts, 'label': test_labels}).to_csv(held_out_test_path, index=False)
    print(f"[Artifact] Saved held-out test dataset to {held_out_test_path}")

    # 4. Tokenizer & Tokenization
    model_name = "distilbert/distilbert-base-uncased"
    print(f"[Tokenizer] Initializing {model_name} tokenizer (max_length={args.max_length})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("[Tokenization] Encoding training and validation sequences...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=args.max_length)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=args.max_length)

    train_dataset = IMDbDataset(train_encodings, train_labels)
    val_dataset = IMDbDataset(val_encodings, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 5. Initialize Model
    print(f"[Model] Instantiating sequence classification model from {model_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1}
    )
    model.to(device)

    # 6. Optimizer and Learning Rate Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    print(f"[Training] Total Optimization Steps: {total_steps} (Warmup Steps: {warmup_steps})")
    print(f"[Training] Epochs: {args.epochs} | Batch Size: {args.batch_size} | Base LR: {args.learning_rate}")

    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': []
    }

    start_time = time.time()
    best_val_f1 = 0.0

    # 7. Training Loop
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            if step % 25 == 0 or step == len(train_loader):
                step_loss = running_loss / step
                print(f"  Epoch [{epoch}/{args.epochs}] | Step [{step}/{len(train_loader)}] | Avg Train Loss: {step_loss:.4f}")

        avg_train_loss = running_loss / len(train_loader)

        # Validation
        val_metrics = evaluate_model(model, val_loader, device)
        epoch_duration = time.time() - epoch_start

        history['epoch'].append(epoch)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])

        print(f"--> Epoch {epoch} Complete ({epoch_duration:.1f}s) | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']*100:.2f}% | Val F1: {val_metrics['f1']:.4f}")

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            print("    [Checkpoint] Best validation F1 achieved. Saving candidate weights...")
            model.save_pretrained(model_save_dir)
            tokenizer.save_pretrained(model_save_dir)

    total_training_time = time.time() - start_time
    print(f"\n[Training Complete] Total elapsed time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} mins)")

    # 8. Save Final Model and Tokenizer
    model.save_pretrained(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)
    print(f"[Save] Model and tokenizer saved to: {model_save_dir}")

    # 9. Save Training History & Configuration
    training_summary = {
        'model_name': model_name,
        'base_architecture': 'DistilBertForSequenceClassification',
        'train_samples': len(train_texts),
        'val_samples': len(val_texts),
        'test_samples': len(test_texts),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'max_sequence_length': args.max_length,
        'device': str(device),
        'seed': args.seed,
        'total_training_time_seconds': round(total_training_time, 2),
        'total_training_time_minutes': round(total_training_time / 60, 2),
        'best_val_f1': round(best_val_f1, 4),
        'history': history
    }

    history_json_path = os.path.join(results_dir, "training_history.json")
    with open(history_json_path, 'w') as f:
        json.dump(training_summary, f, indent=4)
    print(f"[Save] Training summary saved to: {history_json_path}")

    # 10. Smoke Test Inference
    print("\n[Inference Verification] Running quick test on sample statements:")
    test_samples = [
        "This film is a breathtaking masterpiece of modern cinema!",
        "An absolute disaster with wooden acting and a nonsensical plot."
    ]
    model.eval()
    for text in test_samples:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = np.argmax(probs)
            label = "POSITIVE" if pred == 1 else "NEGATIVE"
            confidence = probs[pred]
        print(f"  Review: '{text}' --> {label} (Confidence: {confidence*100:.2f}%)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for Sentiment Analysis")
    parser.add_argument('--train_samples', type=int, default=2000, help='Total training reviews (balanced)')
    parser.add_argument('--val_samples', type=int, default=500, help='Total validation reviews (balanced)')
    parser.add_argument('--test_samples', type=int, default=1000, help='Total held-out test reviews (balanced)')
    parser.add_argument('--epochs', type=int, default=2, help='Number of fine-tuning epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and evaluation')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='AdamW learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='AdamW weight decay')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum token sequence length')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='', help='Compute device override (e.g. cpu, mps, cuda)')
    parser.add_argument('--save_dir', type=str, default='models/finetuned_distilbert', help='Directory to save fine-tuned model')
    parser.add_argument('--results_dir', type=str, default='results', help='Directory to save training logs')

    args = parser.parse_args()
    train_model(args)
