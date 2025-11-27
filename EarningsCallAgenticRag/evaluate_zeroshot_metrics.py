#!/usr/bin/env python3
"""
Evaluate Zero-Shot Classifier Metrics
Calculate weighted macro F1 and MCC scores for various return lags
"""

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef, classification_report
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def evaluate_metrics(file_path, return_column):
    """Calculate F1 and MCC metrics for a specific lag"""

    # Load the results
    df = pd.read_csv(file_path)

    # Filter out rows with valid predictions (directional_score) and actual returns
    df_valid = df.dropna(subset=['directional_score', 'actual_return'])

    # Convert directional scores to classes (1-10 to 3 classes: bearish 1-3, neutral 4-7, bullish 8-10)
    def score_to_class(score):
        if score <= 3:
            return 0  # bearish
        elif score <= 7:
            return 1  # neutral
        else:
            return 2  # bullish

    # Convert returns to classes based on threshold
    def return_to_class(ret, threshold=0.02):
        if ret < -threshold:
            return 0  # bearish
        elif ret > threshold:
            return 2  # bullish
        else:
            return 1  # neutral

    y_pred = df_valid['directional_score'].apply(score_to_class)
    y_true = df_valid['actual_return'].apply(return_to_class)

    # Calculate metrics
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_true, y_pred)

    # Get additional statistics
    total_samples = len(df)
    valid_samples = len(df_valid)
    accuracy = (y_true == y_pred).mean()

    # Get class distribution
    unique_classes = sorted(set(y_true) | set(y_pred))
    class_dist = y_true.value_counts().sort_index()
    pred_dist = y_pred.value_counts().sort_index()

    return {
        'return_column': return_column,
        'total_samples': total_samples,
        'valid_samples': valid_samples,
        'weighted_macro_f1': weighted_f1,
        'mcc': mcc,
        'accuracy': accuracy,
        'true_class_distribution': class_dist.to_dict(),
        'pred_class_distribution': pred_dist.to_dict()
    }

def main():
    """Process all available result files"""

    results_dir = Path("./zeroshot_results")

    # Define the return columns and their corresponding files
    return_columns = [
        ('future_3bday_cum_return', 'zeroshot_results_future_3bday_cum_return.csv'),
        ('return_3d', 'zeroshot_results_return_3d.csv'),
        ('return_7d', 'zeroshot_results_return_7d.csv'),
        ('return_15d', 'zeroshot_results_return_15d.csv'),
        ('return_30d', 'zeroshot_results_return_30d.csv')
    ]

    print("=" * 80)
    print("Zero-Shot Classifier Evaluation - Weighted Macro F1 and MCC")
    print("=" * 80)
    print()

    # Collect all metrics
    all_metrics = []

    for return_col, filename in return_columns:
        file_path = results_dir / filename

        if not file_path.exists():
            print(f"⚠️  File not found: {file_path}")
            continue

        print(f"Processing {return_col}...")
        metrics = evaluate_metrics(file_path, return_col)
        all_metrics.append(metrics)

        # Print detailed results
        print(f"\n📊 {return_col}:")
        print(f"  Total samples: {metrics['total_samples']:,}")
        print(f"  Valid samples: {metrics['valid_samples']:,}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Weighted Macro F1: {metrics['weighted_macro_f1']:.4f}")
        print(f"  MCC: {metrics['mcc']:.4f}")
        print(f"  True class distribution: {metrics['true_class_distribution']}")
        print(f"  Pred class distribution: {metrics['pred_class_distribution']}")

    # Create summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    # Format as table
    print(f"\n{'Return Column':<30} {'Samples':<10} {'Accuracy':<10} {'Weighted F1':<12} {'MCC':<10}")
    print("-" * 72)

    for metrics in all_metrics:
        print(f"{metrics['return_column']:<30} "
              f"{metrics['valid_samples']:<10} "
              f"{metrics['accuracy']:<10.4f} "
              f"{metrics['weighted_macro_f1']:<12.4f} "
              f"{metrics['mcc']:<10.4f}")

    # Calculate average metrics
    if all_metrics:
        avg_f1 = np.mean([m['weighted_macro_f1'] for m in all_metrics])
        avg_mcc = np.mean([m['mcc'] for m in all_metrics])
        avg_acc = np.mean([m['accuracy'] for m in all_metrics])

        print("-" * 72)
        print(f"{'Average':<30} {'':<10} {avg_acc:<10.4f} {avg_f1:<12.4f} {avg_mcc:<10.4f}")

    # Save summary to CSV
    summary_df = pd.DataFrame(all_metrics)[['return_column', 'valid_samples', 'accuracy', 'weighted_macro_f1', 'mcc']]
    summary_df.to_csv('zeroshot_evaluation_summary.csv', index=False)
    print(f"\n✅ Summary saved to zeroshot_evaluation_summary.csv")

if __name__ == "__main__":
    main()