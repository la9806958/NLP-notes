"""
Analysis script for Mock Zero-Shot GPT results
Adapted to work with the generated mock results files
"""

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef
import os

def analyze_mock_zeroshot_results():
    """Analyze mock zero-shot results in merge_and_analyze.py style"""

    results_dir = './zeroshot_results'
    return_columns = ['future_3bday_cum_return', 'return_3d', 'return_7d', 'return_15d', 'return_30d']

    print("Loading Mock Zero-Shot GPT results...")

    all_results = []

    for return_col in return_columns:
        file_path = os.path.join(results_dir, f'zeroshot_results_{return_col}.csv')

        if not os.path.exists(file_path):
            print(f"❌ Results file not found: {file_path}")
            continue

        print(f"\n{'='*60}")
        print(f"ANALYZING RETURN COLUMN: {return_col}")
        print(f"{'='*60}")

        try:
            # Load results
            df = pd.read_csv(file_path)
            print(f"Loaded {len(df)} samples from {file_path}")

            # Filter valid results
            valid_df = df.dropna(subset=['directional_score', 'actual_return']).copy()
            print(f"Valid data points: {len(valid_df)}")

            if len(valid_df) == 0:
                print("No valid data points found!")
                continue

            # Convert to binary predictions
            valid_df['pred_direction'] = (valid_df['directional_score'] > 5).astype(int)
            valid_df['return_direction'] = (valid_df['actual_return'] > 0).astype(int)

            print(f"\n--- Analyzing {return_col} ---")
            print(f"Valid data points: {len(valid_df)}")

            # Show examples
            print(f"\nFirst 5 examples:")
            sample_cols = ['ticker', 'directional_score', 'pred_direction', 'return_direction', 'actual_return']
            print(valid_df[sample_cols].head())

            # Compute metrics
            y_true = valid_df['return_direction'].values
            y_pred = valid_df['pred_direction'].values

            f1 = f1_score(y_true, y_pred, average='macro')
            mcc = matthews_corrcoef(y_true, y_pred)

            print(f"\nResults for {return_col}:")
            print(f"F1 Score: {f1:.4f}")
            print(f"MCC: {mcc:.4f}")
            print(f"Number of samples: {len(valid_df)}")

            # Store results
            result = {
                'return_column': return_col,
                'method': 'Mock_ZeroShot_GPT',
                'n_samples': len(valid_df),
                'f1_score': f1,
                'mcc': mcc
            }
            all_results.append(result)

            # Show distribution
            score_dist = valid_df['directional_score'].value_counts().sort_index()
            print(f"\nDirectional Score Distribution:")
            for score, count in score_dist.items():
                pct = count / len(valid_df) * 100
                print(f"  Score {score}: {count} samples ({pct:.1f}%)")

        except Exception as e:
            print(f"❌ Error processing {return_col}: {e}")
            continue

    # Summary in merge_and_analyze.py style
    if all_results:
        print(f"\n{'='*50}")
        print("SUMMARY")
        print(f"{'='*50}")

        for result in all_results:
            print(f"\n{result['return_column']}:")
            print(f"  Samples: {result['n_samples']}")
            print(f"  F1 Score: {result['f1_score']:.4f}")
            print(f"  MCC: {result['mcc']:.4f}")

        # Save summary
        summary_df = pd.DataFrame(all_results)
        summary_file = './mock_zeroshot_analysis_results.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"\nDetailed results saved to: {summary_file}")

        # Performance insights
        avg_f1 = summary_df['f1_score'].mean()
        avg_mcc = summary_df['mcc'].mean()
        best_f1 = summary_df.loc[summary_df['f1_score'].idxmax()]
        best_mcc = summary_df.loc[summary_df['mcc'].idxmax()]

        print(f"\n📊 Mock Zero-Shot GPT Performance:")
        print(f"Average F1 Score: {avg_f1:.4f}")
        print(f"Average MCC: {avg_mcc:.4f}")
        print(f"Best F1: {best_f1['return_column']} (F1: {best_f1['f1_score']:.4f})")
        print(f"Best MCC: {best_mcc['return_column']} (MCC: {best_mcc['mcc']:.4f})")

        print(f"\n🔬 Comparison with other methods:")
        print(f"LM Dictionary Average: F1=0.40, MCC=0.075")
        print(f"Mock Zero-Shot GPT: F1={avg_f1:.3f}, MCC={avg_mcc:.3f}")

        if avg_mcc > 0.075:
            print("✅ Mock GPT outperforms LM dictionary baseline")
        else:
            print("⚠️ Mock GPT similar to LM dictionary performance")

if __name__ == "__main__":
    analyze_mock_zeroshot_results()