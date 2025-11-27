"""
Loughran-McDonald Dictionary-based Sentiment Classifier for EarningsFilteredResults2.csv

This script applies the Loughran-McDonald financial sentiment dictionary to classify
earnings call transcripts and evaluate performance against different return horizons.
Supports comprehensive evaluation across all return columns in EarningsFilteredResults2.csv.

Usage:
    python baseline/lm_earnings_classifier.py [return_column]

Available return columns:
- future_3bday_cum_return (default)
- return_3d
- return_7d
- return_15d
- return_30d
"""

import pandas as pd
import re
import sys
import os
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, classification_report
)
import numpy as np

# --- Configuration ---
LM_DICT_PATH = "baseline/LM/LM_Master.csv"
DATA_PATH = "EarningsFilteredResults2.csv"
OUTPUT_DIR = "./lm_results"
DEFAULT_RETURN_COLUMN = "future_3bday_cum_return"

# Available return columns for evaluation
RETURN_COLUMNS = [
    'future_3bday_cum_return',
    'return_3d',
    'return_7d',
    'return_15d',
    'return_30d'
]

def load_lm_dictionary(path: str) -> tuple[set[str], set[str]]:
    """
    Loads the Loughran-McDonald sentiment word lists.
    The dictionary is expected to have 'Word', 'Positive', and 'Negative' columns.
    """
    print(f"Loading Loughran-McDonald dictionary from: {path}")
    try:
        lm_df = pd.read_csv(path)

        # Check required columns
        required_cols = ['Word', 'Positive', 'Negative']
        if not all(col in lm_df.columns for col in required_cols):
            print(f"Error: Dictionary must contain columns: {required_cols}")
            return set(), set()

        # Words are converted to uppercase for case-insensitive matching
        positive_words = set(lm_df[lm_df['Positive'] != 0]['Word'].str.upper())
        negative_words = set(lm_df[lm_df['Negative'] != 0]['Word'].str.upper())

        print(f"Loaded {len(positive_words)} positive words and {len(negative_words)} negative words.")
        return positive_words, negative_words
    except FileNotFoundError:
        print(f"Error: Loughran-McDonald dictionary not found at {path}")
        return set(), set()
    except Exception as e:
        print(f"Error loading dictionary: {e}")
        return set(), set()

def analyze_sentiment(transcript: str, positive_words: set[str], negative_words: set[str]) -> dict:
    """
    Analyzes the sentiment of a single transcript using LM dictionary.

    Returns:
        dict: Contains positive_count, negative_count, polarity, sentiment, and word_count
    """
    if not isinstance(transcript, str) or not transcript.strip():
        return {
            'positive_count': 0,
            'negative_count': 0,
            'polarity': 0,
            'sentiment': 'Neutral',
            'word_count': 0,
            'sentiment_ratio': 0.0
        }

    # Tokenize: convert to uppercase and split by word boundaries
    words = re.findall(r'\b\w+\b', transcript.upper())
    word_count = len(words)

    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)

    polarity = positive_count - negative_count

    # Calculate sentiment ratio (net sentiment per word)
    sentiment_ratio = polarity / word_count if word_count > 0 else 0.0

    # Determine sentiment classification
    if polarity > 0:
        sentiment = 'Positive'
    elif polarity < 0:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'

    return {
        'positive_count': positive_count,
        'negative_count': negative_count,
        'polarity': polarity,
        'sentiment': sentiment,
        'word_count': word_count,
        'sentiment_ratio': sentiment_ratio
    }

def load_earnings_data(path: str, return_column: str) -> pd.DataFrame:
    """
    Load and prepare EarningsFilteredResults2.csv data.
    """
    print(f"Loading earnings data from: {path}")
    try:
        df = pd.read_csv(path)
        print(f"Original dataset size: {len(df)} rows")
        print(f"Available columns: {list(df.columns)}")

        # Check required columns
        if 'transcript' not in df.columns:
            print("Error: 'transcript' column not found")
            return None

        if return_column not in df.columns:
            print(f"Error: return column '{return_column}' not found")
            print(f"Available return columns: {[col for col in df.columns if 'return' in col.lower()]}")
            return None

        # Drop rows with missing transcript or return data
        original_size = len(df)
        df = df.dropna(subset=['transcript', return_column])
        print(f"After dropping missing values: {len(df)} rows (removed {original_size - len(df)})")

        return df

    except FileNotFoundError:
        print(f"Error: Data file not found at {path}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def evaluate_predictions(y_true, y_pred, return_column: str) -> dict:
    """
    Comprehensive evaluation of predictions.
    """
    results = {
        'return_column': return_column,
        'total_samples': len(y_true),
        'positive_samples': int(np.sum(y_true)),
        'negative_samples': int(len(y_true) - np.sum(y_true)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='binary', zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average='binary', zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, average='binary', zero_division=0)),
        'macro_precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'macro_recall': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'mcc': float(matthews_corrcoef(y_true, y_pred))
    }

    return results

def run_lm_classification(return_column: str = DEFAULT_RETURN_COLUMN):
    """
    Run LM dictionary classification for a specific return column.
    """
    print(f"\n{'='*60}")
    print(f"Loughran-McDonald Classification - Return Column: {return_column}")
    print(f"{'='*60}")

    # Load LM dictionary
    positive_words, negative_words = load_lm_dictionary(LM_DICT_PATH)
    if not positive_words and not negative_words:
        print("Cannot proceed without sentiment dictionary.")
        return None

    # Load earnings data
    df = load_earnings_data(DATA_PATH, return_column)
    if df is None:
        return None

    # Create binary labels: 1 for positive returns, 0 for negative/zero
    df['label'] = (df[return_column] > 0).astype(int)

    print(f"Label distribution:")
    print(f"  - Positive returns (>0): {df['label'].sum()} ({df['label'].mean():.1%})")
    print(f"  - Negative/Zero returns (<=0): {(1-df['label']).sum()} ({(1-df['label']).mean():.1%})")

    # Apply sentiment analysis
    print("Analyzing sentiment...")
    tqdm.pandas(desc="Processing transcripts")

    sentiment_results = df['transcript'].progress_apply(
        lambda t: analyze_sentiment(t, positive_words, negative_words)
    )

    # Combine results
    sentiment_df = pd.DataFrame(sentiment_results.tolist())
    results_df = pd.concat([df, sentiment_df], axis=1)

    # Show sentiment distribution
    sentiment_counts = results_df['sentiment'].value_counts()
    print(f"\nSentiment distribution:")
    for sentiment, count in sentiment_counts.items():
        print(f"  - {sentiment}: {count} ({count/len(results_df):.1%})")

    # Evaluation Strategy 1: Only Positive vs Negative (exclude Neutral)
    print(f"\n--- Evaluation 1: Positive vs Negative Only ---")
    non_neutral_df = results_df[results_df['sentiment'] != 'Neutral'].copy()

    if len(non_neutral_df) > 0:
        non_neutral_df['pred'] = (non_neutral_df['sentiment'] == 'Positive').astype(int)

        eval_results_1 = evaluate_predictions(
            non_neutral_df['label'],
            non_neutral_df['pred'],
            return_column
        )
        eval_results_1['method'] = 'LM_Pos_vs_Neg'
        eval_results_1['samples_used'] = len(non_neutral_df)

        print(f"Samples used: {eval_results_1['samples_used']}/{len(results_df)} ({eval_results_1['samples_used']/len(results_df):.1%})")
        print(f"Accuracy: {eval_results_1['accuracy']:.4f}")
        print(f"Balanced Accuracy: {eval_results_1['balanced_accuracy']:.4f}")
        print(f"F1-Score: {eval_results_1['f1_score']:.4f}")
        print(f"MCC: {eval_results_1['mcc']:.4f}")
    else:
        print("No non-neutral predictions found!")
        eval_results_1 = None

    # Evaluation Strategy 2: All predictions with Neutral as Negative
    print(f"\n--- Evaluation 2: All Predictions (Neutral as Negative) ---")
    results_df['pred_all'] = (results_df['sentiment'] == 'Positive').astype(int)

    eval_results_2 = evaluate_predictions(
        results_df['label'],
        results_df['pred_all'],
        return_column
    )
    eval_results_2['method'] = 'LM_All_Neutral_as_Neg'
    eval_results_2['samples_used'] = len(results_df)

    print(f"Samples used: {eval_results_2['samples_used']}/{len(results_df)} (100.0%)")
    print(f"Accuracy: {eval_results_2['accuracy']:.4f}")
    print(f"Balanced Accuracy: {eval_results_2['balanced_accuracy']:.4f}")
    print(f"F1-Score: {eval_results_2['f1_score']:.4f}")
    print(f"MCC: {eval_results_2['mcc']:.4f}")

    # Evaluation Strategy 3: Polarity-based threshold
    print(f"\n--- Evaluation 3: Polarity-based Classification ---")
    # Use sentiment ratio for more nuanced classification
    results_df['pred_polarity'] = (results_df['sentiment_ratio'] > 0).astype(int)

    eval_results_3 = evaluate_predictions(
        results_df['label'],
        results_df['pred_polarity'],
        return_column
    )
    eval_results_3['method'] = 'LM_Polarity_Threshold'
    eval_results_3['samples_used'] = len(results_df)

    print(f"Samples used: {eval_results_3['samples_used']}/{len(results_df)} (100.0%)")
    print(f"Accuracy: {eval_results_3['accuracy']:.4f}")
    print(f"Balanced Accuracy: {eval_results_3['balanced_accuracy']:.4f}")
    print(f"F1-Score: {eval_results_3['f1_score']:.4f}")
    print(f"MCC: {eval_results_3['mcc']:.4f}")

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save detailed results
    output_file = os.path.join(OUTPUT_DIR, f'lm_results_{return_column}.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")

    # Save evaluation metrics
    all_results = []
    if eval_results_1:
        all_results.append(eval_results_1)
    all_results.extend([eval_results_2, eval_results_3])

    metrics_file = os.path.join(OUTPUT_DIR, f'lm_metrics_{return_column}.csv')
    metrics_df = pd.DataFrame(all_results)
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Evaluation metrics saved to: {metrics_file}")

    return all_results

def run_all_return_columns():
    """
    Run LM classification for all available return columns.
    """
    print("Running Loughran-McDonald classification for all return columns...")

    all_results = []

    for return_col in RETURN_COLUMNS:
        try:
            results = run_lm_classification(return_col)
            if results:
                all_results.extend(results)
        except Exception as e:
            print(f"Error processing {return_col}: {e}")
            continue

    if all_results:
        # Save combined results
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        combined_file = os.path.join(OUTPUT_DIR, 'lm_combined_results.csv')
        combined_df = pd.DataFrame(all_results)
        combined_df.to_csv(combined_file, index=False)

        # Display comparison table
        print(f"\n{'='*80}")
        print("LOUGHRAN-MCDONALD PERFORMANCE COMPARISON")
        print(f"{'='*80}")

        # Create comparison table
        comparison_cols = ['return_column', 'method', 'samples_used', 'balanced_accuracy', 'f1_score', 'mcc']
        comparison_df = combined_df[comparison_cols].round(4)
        print(comparison_df.to_string(index=False))

        # Find best performing methods
        best_f1 = combined_df.loc[combined_df['f1_score'].idxmax()]
        best_mcc = combined_df.loc[combined_df['mcc'].idxmax()]

        print(f"\n🏆 Best F1 Score: {best_f1['method']} on {best_f1['return_column']} (F1: {best_f1['f1_score']:.4f})")
        print(f"🏆 Best MCC Score: {best_mcc['method']} on {best_mcc['return_column']} (MCC: {best_mcc['mcc']:.4f})")

        print(f"\n💾 Combined results saved to: {combined_file}")

def main():
    """
    Main function - supports both single return column and all columns analysis.
    """
    if len(sys.argv) > 1:
        return_column = sys.argv[1]
        if return_column not in RETURN_COLUMNS:
            print(f"Error: Invalid return column '{return_column}'")
            print(f"Available columns: {RETURN_COLUMNS}")
            sys.exit(1)
        run_lm_classification(return_column)
    else:
        run_all_return_columns()

if __name__ == "__main__":
    main()