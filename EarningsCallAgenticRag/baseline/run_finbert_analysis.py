"""
Script to run FinBERT analysis on EarningsFilteredResults2.csv with different return columns.
This script allows you to easily compare FinBERT performance across different return horizons.

Usage:
python baseline/run_finbert_analysis.py [return_column]

Available return columns:
- future_3bday_cum_return (default)
- return_3d
- return_7d
- return_15d
- return_30d
"""

import sys
import os
import subprocess
import pandas as pd

# Available return columns
RETURN_COLUMNS = [
    'future_3bday_cum_return',
    'return_3d',
    'return_7d',
    'return_15d',
    'return_30d'
]

def run_finbert_classifier(return_column):
    """
    Run the FinBERT classifier with the specified return column.
    """
    print(f"\n{'='*60}")
    print(f"Running FinBERT classifier with return column: {return_column}")
    print(f"{'='*60}")

    # Change to the project directory
    os.chdir('/home/lichenhui/EarningsCallAgenticRag')

    # Activate virtual environment and run the classifier
    cmd = f"source venv/bin/activate && python baseline/finbert_earnings_classifier.py {return_column}"

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3600)  # 1 hour timeout

        if result.returncode == 0:
            print("✅ FinBERT classifier completed successfully!")
            print("\nOutput:")
            print(result.stdout)
        else:
            print("❌ FinBERT classifier failed!")
            print("Error output:")
            print(result.stderr)

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print("❌ FinBERT classifier timed out after 1 hour!")
        return False

def analyze_results():
    """
    Analyze and compare results from all return columns.
    """
    print(f"\n{'='*60}")
    print("ANALYZING RESULTS ACROSS ALL RETURN COLUMNS")
    print(f"{'='*60}")

    results_dir = '/home/lichenhui/EarningsCallAgenticRag/baseline/finbert_earnings_results'

    if not os.path.exists(results_dir):
        print("❌ Results directory not found!")
        return

    # Look for summary files
    summary_files = []
    for return_col in RETURN_COLUMNS:
        summary_file = os.path.join(results_dir, f'summary_results_{return_col}.csv')
        if os.path.exists(summary_file):
            summary_files.append((return_col, summary_file))

    if not summary_files:
        print("❌ No summary files found!")
        return

    # Load and combine results
    all_results = []
    for return_col, file_path in summary_files:
        try:
            df = pd.read_csv(file_path)
            df['return_column'] = return_col
            all_results.append(df)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)

        # Display comparison table
        print("\nFinBERT Performance Comparison:")
        print("-" * 80)

        comparison_df = combined_df[['return_column', 'total_samples', 'overall_accuracy',
                                     'overall_f1', 'overall_f1_macro', 'overall_mcc']].round(4)

        print(comparison_df.to_string(index=False))

        # Save combined results
        combined_path = os.path.join(results_dir, 'combined_finbert_results.csv')
        combined_df.to_csv(combined_path, index=False)
        print(f"\n💾 Combined results saved to: {combined_path}")

        # Find best performing return column
        best_f1 = combined_df.loc[combined_df['overall_f1'].idxmax()]
        best_mcc = combined_df.loc[combined_df['overall_mcc'].idxmax()]

        print(f"\n🏆 Best F1 Score: {best_f1['return_column']} (F1: {best_f1['overall_f1']:.4f})")
        print(f"🏆 Best MCC Score: {best_mcc['return_column']} (MCC: {best_mcc['overall_mcc']:.4f})")

def main():
    """
    Main function to run FinBERT analysis.
    """
    if len(sys.argv) > 1:
        # Run with specific return column
        return_column = sys.argv[1]
        if return_column not in RETURN_COLUMNS:
            print(f"❌ Invalid return column: {return_column}")
            print(f"Available columns: {RETURN_COLUMNS}")
            sys.exit(1)

        success = run_finbert_classifier(return_column)
        if success:
            analyze_results()
    else:
        # Run with all return columns
        print("Running FinBERT analysis with all return columns...")

        successful_runs = 0
        for return_column in RETURN_COLUMNS:
            success = run_finbert_classifier(return_column)
            if success:
                successful_runs += 1
            else:
                print(f"❌ Failed to run FinBERT with {return_column}")

        print(f"\n📊 Completed {successful_runs}/{len(RETURN_COLUMNS)} runs successfully")

        if successful_runs > 0:
            analyze_results()

if __name__ == "__main__":
    main()