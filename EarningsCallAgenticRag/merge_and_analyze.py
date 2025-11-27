import pandas as pd
import numpy as np
import re
from sklearn.metrics import f1_score, matthews_corrcoef

def extract_directional_score(research_note):
    """Extract directional score from research_note text."""
    if pd.isna(research_note) or research_note == "":
        return np.nan

    # Look for patterns like "Direction: 8" or "directional score 8" or "score: 8"
    patterns = [
        r'[Dd]irection[:\s]*(\d+)',
        r'[Dd]irectional\s*score[:\s]*(\d+)',
        r'[Ss]core[:\s]*(\d+)',
        r'Direction:\s*(\d+)',
        r'directional\s*score:\s*(\d+)',
        r'score:\s*(\d+)'
    ]

    for pattern in patterns:
        match = re.search(pattern, research_note)
        if match:
            score = int(match.group(1))
            if 0 <= score <= 10:
                return score

    # Look for the pattern at the end like "Direction: 8"
    end_pattern = r'Direction:\s*(\d+)$'
    match = re.search(end_pattern, research_note)
    if match:
        score = int(match.group(1))
        if 0 <= score <= 10:
            return score

    return np.nan

def direction_from_score(score):
    """Convert directional score to direction: -1 if <=5, 1 if >5."""
    if pd.isna(score):
        return np.nan
    return 1 if score > 5 else -1

def direction_from_return(return_val):
    """Convert return value to direction: 1 if positive, -1 if negative or zero."""
    if pd.isna(return_val):
        return np.nan
    return 1 if return_val > 0 else -1

def main():
    print("Loading CSV files...")

    # Load the CSV files
    final_results = pd.read_csv('FinalResults.csv')
    earnings_filtered = pd.read_csv('EarningsFilteredResults2.csv')

    print(f"FinalResults.csv shape: {final_results.shape}")
    print(f"EarningsFilteredResults2.csv shape: {earnings_filtered.shape}")

    print("\nColumns in FinalResultsQoQ.csv:")
    print(final_results.columns.tolist())
    print("\nColumns in EarningsFilteredResults2.csv:")
    print(earnings_filtered.columns.tolist())

    # Merge the dataframes on ticker and actual_return
    print("\nMerging dataframes on ticker and actual_return...")

    # Check if actual_return column exists in both dataframes
    if 'actual_return' not in final_results.columns:
        print("Warning: 'actual_return' column not found in FinalResultsQoQ.csv")
        # Look for similar column names
        for col in final_results.columns:
            if 'return' in col.lower():
                print(f"Found return column: {col}")

    if 'actual_return' not in earnings_filtered.columns:
        print("Warning: 'actual_return' column not found in EarningsFilteredResults2.csv")
        # Look for similar column names
        for col in earnings_filtered.columns:
            if 'return' in col.lower():
                print(f"Found return column: {col}")

    # Use the correct return column names based on what we found
    final_return_col = 'actual_return' if 'actual_return' in final_results.columns else None
    earnings_return_cols = [col for col in earnings_filtered.columns if 'return' in col.lower()]

    if final_return_col is None:
        print("No return column found in FinalResults.csv")
        return

    print(f"Return columns in EarningsFilteredResults2.csv: {earnings_return_cols}")

    # Merge actual_return with future_3bday_cum_return first
    print(f"\nMerging actual_return with future_3bday_cum_return...")

    # Primary merge on actual_return = future_3bday_cum_return
    merged_df = pd.merge(
        final_results[['ticker', final_return_col, 'research_note']],
        earnings_filtered[['ticker', 'future_3bday_cum_return', 'return_3d', 'return_7d', 'return_15d', 'return_30d']],
        left_on=final_return_col,
        right_on='future_3bday_cum_return',
        how='inner'
    )

    print(f"Base merged dataframe shape: {merged_df.shape}")

    if merged_df.empty:
        print("No matches found between actual_return and future_3bday_cum_return")
        return

    # Extract directional scores from research_note
    print("Extracting directional scores from research_note...")
    merged_df['directional_score'] = merged_df['research_note'].apply(extract_directional_score)
    merged_df['score_direction'] = merged_df['directional_score'].apply(direction_from_score)

    # Remove rows with missing directional scores
    valid_base_data = merged_df.dropna(subset=['score_direction'])
    print(f"Records with valid directional scores: {len(valid_base_data)}")

    if len(valid_base_data) == 0:
        print("No valid directional scores found")
        return

    # Now analyze each return column from the same merged dataset
    results = {}
    all_return_cols = ['future_3bday_cum_return', 'return_3d', 'return_7d', 'return_15d', 'return_30d']

    for return_col in all_return_cols:
        print(f"\n--- Analyzing {return_col} ---")

        # Create return direction for this column
        valid_data = valid_base_data.copy()
        valid_data['return_direction'] = valid_data[return_col].apply(direction_from_return)

        # Remove rows with missing return values
        valid_data = valid_data.dropna(subset=['return_direction'])
        print(f"Valid data points: {len(valid_data)}")

        if len(valid_data) == 0:
            print(f"No valid data points for {return_col}")
            continue

        # Print some examples
        print("\nFirst 5 examples:")
        # Find available columns
        available_cols = ['directional_score', 'score_direction', return_col, 'return_direction']
        if 'ticker_x' in valid_data.columns:
            available_cols.insert(0, 'ticker_x')
        elif 'ticker_y' in valid_data.columns:
            available_cols.insert(0, 'ticker_y')
        elif 'ticker' in valid_data.columns:
            available_cols.insert(0, 'ticker')

        print(valid_data[available_cols].head())

        # Compute F1 and MCC
        y_true = valid_data['return_direction'].values
        y_pred = valid_data['score_direction'].values

        f1 = f1_score(y_true, y_pred, average='weighted')
        mcc = matthews_corrcoef(y_true, y_pred)

        results[return_col] = {
            'n_samples': len(valid_data),
            'f1_score': f1,
            'mcc': mcc
        }

        print(f"\nResults for {return_col}:")
        print(f"F1 Score: {f1:.4f}")
        print(f"MCC: {mcc:.4f}")
        print(f"Number of samples: {len(valid_data)}")

    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)

    for return_col, metrics in results.items():
        print(f"\n{return_col}:")
        print(f"  Samples: {metrics['n_samples']}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        print(f"  MCC: {metrics['mcc']:.4f}")

if __name__ == "__main__":
    main()