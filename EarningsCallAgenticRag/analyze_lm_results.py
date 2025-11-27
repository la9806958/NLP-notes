"""
Analysis script for Loughran-McDonald classifier results.
Adapted from merge_and_analyze.py to work with LM dictionary results.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef
import os

def load_lm_results():
    """Load LM classifier results for all return columns"""
    lm_dir = './lm_results'

    if not os.path.exists(lm_dir):
        print(f"Error: {lm_dir} directory not found!")
        print("Please run baseline/lm_earnings_classifier.py first.")
        return None

    # Look for detailed results files
    result_files = []
    return_columns = ['future_3bday_cum_return', 'return_3d', 'return_7d', 'return_15d', 'return_30d']

    for return_col in return_columns:
        file_path = os.path.join(lm_dir, f'lm_results_{return_col}.csv')
        if os.path.exists(file_path):
            result_files.append((return_col, file_path))

    if not result_files:
        print(f"No LM result files found in {lm_dir}")
        return None

    print(f"Found {len(result_files)} LM result files")
    return result_files

def analyze_lm_predictions(return_column, df):
    """Analyze LM predictions for a specific return column - mimicking merge_and_analyze.py style"""
    print(f"\n--- Analyzing {return_column} ---")

    # Check required columns
    required_cols = ['sentiment', 'polarity', 'sentiment_ratio', 'label']
    if not all(col in df.columns for col in required_cols):
        print(f"Missing required columns. Available: {list(df.columns)}")
        return None

    valid_data = df.dropna(subset=required_cols)
    print(f"Valid data points: {len(valid_data)}")

    if len(valid_data) == 0:
        print("No valid data points found!")
        return None

    # Show sample predictions - mimicking merge_and_analyze.py style
    print(f"\nFirst 5 examples:")
    # Create prediction direction: 1 for Positive, -1 for Negative, 0 for Neutral
    valid_data = valid_data.copy()
    valid_data['pred_direction'] = valid_data['sentiment'].map({'Positive': 1, 'Negative': -1, 'Neutral': 0})
    valid_data['return_direction'] = (valid_data[return_column] > 0).astype(int) * 2 - 1  # 1 for positive, -1 for negative

    # Find available columns for display
    available_cols = ['pred_direction', 'return_direction', return_column]
    if 'ticker' in df.columns:
        available_cols.insert(0, 'ticker')

    print(valid_data[available_cols].head())

    results = {}

    # Strategy 1: Positive vs Negative (excluding Neutral) - matches merge_and_analyze.py approach
    non_neutral_data = valid_data[valid_data['sentiment'] != 'Neutral']
    if len(non_neutral_data) > 0:
        # Convert to binary: positive sentiment = 1, negative = 0 for predictions
        # positive return = 1, negative/zero = 0 for true labels
        y_pred = (non_neutral_data['sentiment'] == 'Positive').astype(int)
        y_true = (non_neutral_data[return_column] > 0).astype(int)

        f1 = f1_score(y_true, y_pred, average='macro')  # Using macro F1 like requested
        mcc = matthews_corrcoef(y_true, y_pred)

        results['LM_Pos_vs_Neg'] = {
            'n_samples': len(non_neutral_data),
            'f1_score': f1,
            'mcc': mcc
        }

        print(f"\nResults for {return_column} (Pos vs Neg only):")
        print(f"F1 Score: {f1:.4f}")
        print(f"MCC: {mcc:.4f}")
        print(f"Number of samples: {len(non_neutral_data)}")

    # Strategy 2: All predictions (Neutral as Negative)
    y_pred_all = (valid_data['sentiment'] == 'Positive').astype(int)
    y_true_all = (valid_data[return_column] > 0).astype(int)

    f1_all = f1_score(y_true_all, y_pred_all, average='macro')
    mcc_all = matthews_corrcoef(y_true_all, y_pred_all)

    results['LM_All_Neutral_as_Neg'] = {
        'n_samples': len(valid_data),
        'f1_score': f1_all,
        'mcc': mcc_all
    }

    print(f"\nResults for {return_column} (All predictions):")
    print(f"F1 Score: {f1_all:.4f}")
    print(f"MCC: {mcc_all:.4f}")
    print(f"Number of samples: {len(valid_data)}")

    # Strategy 3: Polarity-based threshold
    y_pred_polarity = (valid_data['sentiment_ratio'] > 0).astype(int)

    f1_polarity = f1_score(y_true_all, y_pred_polarity, average='macro')
    mcc_polarity = matthews_corrcoef(y_true_all, y_pred_polarity)

    results['LM_Polarity_Threshold'] = {
        'n_samples': len(valid_data),
        'f1_score': f1_polarity,
        'mcc': mcc_polarity
    }

    print(f"\nResults for {return_column} (Polarity threshold):")
    print(f"F1 Score: {f1_polarity:.4f}")
    print(f"MCC: {mcc_polarity:.4f}")
    print(f"Number of samples: {len(valid_data)}")

    return results

def main():
    print("Loading LM classifier results...")
    result_files = load_lm_results()

    if not result_files:
        return

    all_results = []

    for return_column, file_path in result_files:
        print(f"\n{'='*60}")
        print(f"ANALYZING RETURN COLUMN: {return_column}")
        print(f"{'='*60}")

        try:
            # Load LM results
            df = pd.read_csv(file_path)
            print(f"Loaded {len(df)} samples from {file_path}")

            # Analyze predictions
            results = analyze_lm_predictions(return_column, df)

            if results:
                # Store results for comparison
                for strategy, metrics in results.items():
                    result_entry = {
                        'return_column': return_column,
                        'strategy': strategy,
                        **metrics
                    }
                    all_results.append(result_entry)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    if all_results:
        # Create summary - matching merge_and_analyze.py style exactly
        print(f"\n{'='*50}")
        print("SUMMARY")
        print(f"{'='*50}")

        # Convert results to match merge_and_analyze.py format
        results_dict = {}
        for result in all_results:
            return_col = result['return_column']
            strategy = result['strategy']
            key = f"{return_col}_{strategy}"
            results_dict[key] = {
                'n_samples': result['n_samples'],
                'f1_score': result['f1_score'],
                'mcc': result['mcc']
            }

        # Print results by return column like merge_and_analyze.py
        return_columns = ['future_3bday_cum_return', 'return_3d', 'return_7d', 'return_15d', 'return_30d']

        for return_col in return_columns:
            if any(key.startswith(return_col) for key in results_dict.keys()):
                print(f"\n{return_col}:")
                for strategy in ['LM_Pos_vs_Neg', 'LM_All_Neutral_as_Neg', 'LM_Polarity_Threshold']:
                    key = f"{return_col}_{strategy}"
                    if key in results_dict:
                        metrics = results_dict[key]
                        print(f"  Samples: {metrics['n_samples']}")
                        print(f"  F1 Score: {metrics['f1_score']:.4f}")
                        print(f"  MCC: {metrics['mcc']:.4f}")
                        print()

        # Save detailed results
        results_df = pd.DataFrame(all_results)
        output_file = './lm_analysis_results.csv'
        results_df.to_csv(output_file, index=False)
        print(f"Detailed results saved to: {output_file}")

if __name__ == "__main__":
    main()