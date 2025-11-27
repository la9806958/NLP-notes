"""
FinBERT Classifier for EarningsFilteredResults2.csv

This script trains a FinBERT model using k-fold cross-validation to classify financial documents
(earnings call transcripts) as having 'Positive' or 'Negative' sentiment based on future stock returns.
Specifically designed to work with EarningsFilteredResults2.csv dataset.

Required packages:
- transformers
- torch
- pandas
- scikit-learn
- accelerate

You can install them using pip:
pip install transformers torch pandas scikit-learn accelerate
"""

import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, f1_score, balanced_accuracy_score, matthews_corrcoef
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import os
import sys

# --- Configuration ---
DATA_PATH = "EarningsFilteredResults2.csv"
MODEL_NAME = "ProsusAI/finbert"
OUTPUT_DIR = "./finbert_results"
LOGGING_DIR = "./finbert_logs"
N_FOLDS = 5  # Number of folds for cross-validation
RETURN_COLUMN = "future_3bday_cum_return"  # Default return column to use

# --- 1. Load and Prepare Data ---
def load_and_prepare_data(file_path: str, return_column: str = RETURN_COLUMN):
    """
    Loads the EarningsFilteredResults2.csv data and creates binary labels based on future returns.
    """
    print("Loading and preparing data...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        return None, None

    print(f"Original dataset size: {len(df)} rows")

    # Check available columns
    print(f"Available columns: {list(df.columns)}")

    # Drop rows with missing transcripts or returns
    df.dropna(subset=['transcript', return_column], inplace=True)

    print(f"After dropping missing values: {len(df)} rows")

    # Create binary labels: 1 for Positive return, 0 for Negative/Zero return
    df['label'] = (df[return_column] > 0).astype(int)

    print(f"Dataset prepared: {len(df)} samples")
    print(f"  - Positive samples ({return_column} > 0): {df['label'].sum()}")
    print(f"  - Negative samples ({return_column} <= 0): {len(df) - df['label'].sum()}")
    print(f"  - Label distribution: {df['label'].value_counts().to_dict()}")

    return df['transcript'].tolist(), df['label'].tolist()


# --- 2. Create PyTorch Dataset ---
class TranscriptDataset(torch.utils.data.Dataset):
    """
    Custom PyTorch Dataset for loading tokenized transcripts.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# --- 3. Define Evaluation Metrics ---
def compute_metrics(pred):
    """
    Computes accuracy, F1, precision, recall, and MCC for evaluation.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    f1_macro = f1_score(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    bal_acc = balanced_accuracy_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    return {
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'f1': f1,
        'f1_macro': f1_macro,
        'precision': precision,
        'recall': recall,
        'mcc': mcc
    }


# --- 4. K-Fold Cross-Validation Training ---
def train_and_evaluate_fold(fold_idx, train_texts, train_labels, val_texts, val_labels, tokenizer):
    """
    Trains a model on one fold and returns predictions for the validation set.
    """
    print(f"\n--- Training Fold {fold_idx + 1}/{N_FOLDS} ---")

    # Tokenize data for this fold
    print("Tokenizing training data...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    print("Tokenizing validation data...")
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

    train_dataset = TranscriptDataset(train_encodings, train_labels)
    val_dataset = TranscriptDataset(val_encodings, val_labels)

    # Load fresh model for this fold
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        ignore_mismatched_sizes=True
    )

    # Set up training arguments for this fold
    fold_output_dir = os.path.join(OUTPUT_DIR, f"fold_{fold_idx}")
    fold_logging_dir = os.path.join(LOGGING_DIR, f"fold_{fold_idx}")

    training_args = TrainingArguments(
        output_dir=fold_output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,  # Increased for CPU optimization
        per_device_eval_batch_size=16,   # Increased for CPU optimization
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=fold_logging_dir,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,  # Reduce memory usage
        fp16=False,  # Disable mixed precision for CPU
        dataloader_num_workers=16,  # Use multiple CPU cores
        report_to="none",  # Disable wandb/tensorboard
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the model
    print("Starting training...")
    trainer.train()

    # Get predictions on validation set
    print("Getting predictions...")
    predictions = trainer.predict(val_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)

    # Compute metrics for this fold
    fold_metrics = compute_metrics(predictions)

    print(f"Fold {fold_idx + 1} Results:")
    print(f"  Accuracy: {fold_metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {fold_metrics['balanced_accuracy']:.4f}")
    print(f"  Macro F1-Score: {fold_metrics['f1_macro']:.4f}")
    print(f"  F1-Score: {fold_metrics['f1']:.4f}")
    print(f"  Precision: {fold_metrics['precision']:.4f}")
    print(f"  Recall: {fold_metrics['recall']:.4f}")
    print(f"  MCC: {fold_metrics['mcc']:.4f}")

    return pred_labels, val_labels, fold_metrics


def main():
    # Parse command line arguments for return column
    return_column = RETURN_COLUMN
    if len(sys.argv) > 1:
        return_column = sys.argv[1]
        print(f"Using return column: {return_column}")

    # --- Check for CUDA ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"✅ CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️ CUDA not available. Running on CPU.")

    # --- Load Data ---
    texts, labels = load_and_prepare_data(DATA_PATH, return_column)
    if texts is None:
        return

    # --- Initialize Tokenizer ---
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # --- Create output directories ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGGING_DIR, exist_ok=True)

    # --- K-Fold Cross-Validation ---
    print(f"\nStarting {N_FOLDS}-fold cross-validation...")

    # Initialize arrays to store all predictions and true labels
    all_predictions = []
    all_true_labels = []
    fold_metrics_list = []

    # Initialize k-fold splitter
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    # Train and evaluate on each fold
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        # Split data for this fold
        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        print(f"Fold {fold_idx + 1} - Train: {len(train_texts)}, Val: {len(val_texts)}")

        # Train model and get predictions for this fold
        pred_labels, true_labels, fold_metrics = train_and_evaluate_fold(
            fold_idx, train_texts, train_labels, val_texts, val_labels, tokenizer
        )

        # Store results
        all_predictions.extend(pred_labels)
        all_true_labels.extend(true_labels)
        fold_metrics_list.append(fold_metrics)

    # --- Compute Overall Results ---
    print(f"\n{'='*60}")
    print("OVERALL CROSS-VALIDATION RESULTS")
    print(f"Using return column: {return_column}")
    print(f"{'='*60}")

    # Convert to numpy arrays for easier computation
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)

    # Overall metrics
    overall_accuracy = accuracy_score(all_true_labels, all_predictions)
    overall_bal_acc = balanced_accuracy_score(all_true_labels, all_predictions)
    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
        all_true_labels, all_predictions, average='binary'
    )
    overall_f1_macro = f1_score(all_true_labels, all_predictions, average='macro')
    overall_mcc = matthews_corrcoef(all_true_labels, all_predictions)

    print(f"Overall Performance (across all {len(all_predictions)} samples):")
    print(f"  Accuracy: {overall_accuracy:.4f}")
    print(f"  Balanced Accuracy: {overall_bal_acc:.4f}")
    print(f"  Macro F1-Score: {overall_f1_macro:.4f}")
    print(f"  F1-Score: {overall_f1:.4f}")
    print(f"  Precision: {overall_precision:.4f}")
    print(f"  Recall: {overall_recall:.4f}")
    print(f"  MCC: {overall_mcc:.4f}")

    # Average fold metrics
    avg_accuracy = np.mean([m['accuracy'] for m in fold_metrics_list])
    avg_bal_acc = np.mean([m['balanced_accuracy'] for m in fold_metrics_list])
    avg_f1 = np.mean([m['f1'] for m in fold_metrics_list])
    avg_f1_macro = np.mean([m['f1_macro'] for m in fold_metrics_list])
    avg_mcc = np.mean([m['mcc'] for m in fold_metrics_list])

    print(f"\nAverage Performance (across {N_FOLDS} folds):")
    print(f"  Accuracy: {avg_accuracy:.4f} ± {np.std([m['accuracy'] for m in fold_metrics_list]):.4f}")
    print(f"  Balanced Accuracy: {avg_bal_acc:.4f} ± {np.std([m['balanced_accuracy'] for m in fold_metrics_list]):.4f}")
    print(f"  Macro F1-Score: {avg_f1_macro:.4f} ± {np.std([m['f1_macro'] for m in fold_metrics_list]):.4f}")
    print(f"  F1-Score: {avg_f1:.4f} ± {np.std([m['f1'] for m in fold_metrics_list]):.4f}")
    print(f"  MCC: {avg_mcc:.4f} ± {np.std([m['mcc'] for m in fold_metrics_list]):.4f}")

    # Detailed classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(all_true_labels, all_predictions, target_names=['Negative', 'Positive']))

    # Save results
    results_df = pd.DataFrame({
        'true_label': all_true_labels,
        'predicted_label': all_predictions,
        'correct': all_true_labels == all_predictions
    })

    results_path = os.path.join(OUTPUT_DIR, f'cross_validation_results_{return_column}.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")

    # Save summary results
    summary_results = {
        'return_column': return_column,
        'total_samples': len(all_predictions),
        'positive_samples': int(np.sum(all_true_labels)),
        'negative_samples': int(len(all_true_labels) - np.sum(all_true_labels)),
        'overall_accuracy': float(overall_accuracy),
        'overall_balanced_accuracy': float(overall_bal_acc),
        'overall_f1': float(overall_f1),
        'overall_f1_macro': float(overall_f1_macro),
        'overall_precision': float(overall_precision),
        'overall_recall': float(overall_recall),
        'overall_mcc': float(overall_mcc),
        'avg_accuracy': float(avg_accuracy),
        'avg_f1': float(avg_f1),
        'avg_f1_macro': float(avg_f1_macro),
        'avg_mcc': float(avg_mcc)
    }

    summary_path = os.path.join(OUTPUT_DIR, f'summary_results_{return_column}.csv')
    pd.DataFrame([summary_results]).to_csv(summary_path, index=False)
    print(f"Summary results saved to: {summary_path}")

    print(f"\n✅ K-fold cross-validation complete! Each sample was predicted exactly once.")

if __name__ == "__main__":
    main()