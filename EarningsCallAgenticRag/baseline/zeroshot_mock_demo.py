#!/usr/bin/env python3
"""
Mock Demo of Zero-Shot GPT Classifier for EarningsFilteredResults2.csv
====================================================================
This script demonstrates how the zero-shot classifier would work
without requiring actual OpenAI API calls. It uses mock responses
to show the analysis format and evaluation approach.
"""

import pandas as pd
import numpy as np
import os
import re
from sklearn.metrics import f1_score, matthews_corrcoef
from typing import Optional, Dict

# Mock responses that simulate GPT output
MOCK_RESPONSES = [
    "The company showed strong revenue growth and raised guidance for the next quarter. Management expressed confidence in market expansion. Direction: 8",
    "Disappointing earnings miss and lowered outlook due to supply chain challenges. Market conditions remain uncertain. Direction: 3",
    "Mixed results with some positive indicators but concerning margin compression. Management cautious about near-term prospects. Direction: 5",
    "Exceptional quarter with record profits and strong cash flow generation. New product launches showing promise. Direction: 9",
    "Revenue declined year-over-year with increased competition pressuring market share. Cost cutting measures implemented. Direction: 2",
    "Solid performance meeting expectations with stable demand trends. No major surprises in the quarter. Direction: 6",
    "Outstanding results exceeding all guidance metrics. Strong momentum across all business segments. Direction: 10",
    "Challenging quarter with regulatory headwinds and operational difficulties. Future outlook remains cloudy. Direction: 4",
    "Strong execution of strategic initiatives with improving operational efficiency. Market position strengthening. Direction: 7",
    "Significant revenue beat but margin pressures from input costs. Management optimistic about recovery. Direction: 6"
]

def mock_gpt_analysis(transcript: str, sentiment_bias: float = 0.0) -> str:
    """
    Generate mock GPT analysis based on transcript characteristics
    sentiment_bias: -1 to 1, affects the direction score
    """
    # Simple sentiment analysis based on transcript length and content
    transcript_lower = transcript.lower()

    # Look for positive/negative keywords
    positive_words = ['growth', 'strong', 'increase', 'beat', 'exceed', 'optimistic', 'confident']
    negative_words = ['decline', 'challenge', 'difficult', 'miss', 'concern', 'pressure', 'weak']

    pos_count = sum(1 for word in positive_words if word in transcript_lower)
    neg_count = sum(1 for word in negative_words if word in transcript_lower)

    # Base score influenced by word counts and bias
    base_score = 5 + (pos_count - neg_count) * 0.5 + sentiment_bias * 2
    base_score = max(0, min(10, int(base_score + np.random.normal(0, 1))))

    # Select appropriate mock response
    response_idx = min(len(MOCK_RESPONSES) - 1, max(0, base_score))
    mock_response = MOCK_RESPONSES[response_idx]

    # Adjust the direction score in the response
    mock_response = re.sub(r'Direction: \d+', f'Direction: {base_score}', mock_response)

    return mock_response

def extract_directional_score(analysis: str) -> Optional[int]:
    """Extract directional score from analysis text"""
    if pd.isna(analysis) or analysis == "":
        return None

    patterns = [
        r'[Dd]irection[:\s]*(\d+)',
        r'[Dd]irectional[:\s]*(\d+)',
        r'[Ss]core[:\s]*(\d+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, analysis)
        if match:
            score = int(match.group(1))
            if 0 <= score <= 10:
                return score
    return None

def run_mock_zeroshot_analysis(return_column: str = 'future_3bday_cum_return', n_samples: int = 100):
    """Run mock zero-shot analysis on sample data"""

    print(f"Mock Zero-Shot Analysis for {return_column}")
    print("=" * 60)

    # Load sample data
    if not os.path.exists('EarningsFilteredResults2.csv'):
        print("❌ EarningsFilteredResults2.csv not found")
        return None

    df = pd.read_csv('EarningsFilteredResults2.csv')
    print(f"Loaded {len(df)} total samples")

    # Take sample for demo
    sample_df = df.sample(n=min(n_samples, len(df)), random_state=42).copy()
    print(f"Processing {len(sample_df)} samples for demo...")

    # Generate mock analyses
    analyses = []
    directional_scores = []

    for idx, row in sample_df.iterrows():
        # Add some bias based on actual return to simulate realistic performance
        actual_return = row[return_column]
        sentiment_bias = 0.3 * np.sign(actual_return) if abs(actual_return) > 0.05 else 0

        # Generate mock analysis
        mock_analysis = mock_gpt_analysis(row['transcript'][:500], sentiment_bias)  # Use first 500 chars
        directional_score = extract_directional_score(mock_analysis)

        analyses.append(mock_analysis)
        directional_scores.append(directional_score)

    # Add results to dataframe
    sample_df['analysis'] = analyses
    sample_df['directional_score'] = directional_scores

    # Filter valid results
    valid_results = sample_df.dropna(subset=['directional_score', return_column]).copy()
    print(f"Valid results: {len(valid_results)}")

    if len(valid_results) == 0:
        print("No valid results generated")
        return None

    # Convert to binary predictions and labels
    valid_results['pred_direction'] = (valid_results['directional_score'] > 5).astype(int)
    valid_results['return_direction'] = (valid_results[return_column] > 0).astype(int)

    print(f"\n--- Analyzing {return_column} ---")
    print(f"Valid data points: {len(valid_results)}")

    # Show sample predictions
    print(f"\nFirst 5 examples:")
    sample_cols = ['ticker', 'directional_score', 'pred_direction', 'return_direction', return_column]
    print(valid_results[sample_cols].head())

    # Compute metrics
    y_true = valid_results['return_direction'].values
    y_pred = valid_results['pred_direction'].values

    f1 = f1_score(y_true, y_pred, average='macro')
    mcc = matthews_corrcoef(y_true, y_pred)

    print(f"\nResults for {return_column}:")
    print(f"F1 Score: {f1:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"Number of samples: {len(valid_results)}")

    # Show distribution
    score_dist = valid_results['directional_score'].value_counts().sort_index()
    print(f"\nDirectional Score Distribution:")
    for score, count in score_dist.items():
        print(f"  Score {score}: {count} samples")

    return {
        'return_column': return_column,
        'n_samples': len(valid_results),
        'f1_score': f1,
        'mcc': mcc
    }

def main():
    """Run mock analysis for demonstration"""
    print("🧠 Mock Zero-Shot GPT Classifier Demo")
    print("====================================")
    print("This demo simulates how the zero-shot classifier would work")
    print("without requiring actual OpenAI API calls.\n")

    # Run analysis for different return columns
    return_columns = ['future_3bday_cum_return', 'return_3d', 'return_7d', 'return_15d', 'return_30d']

    all_results = []

    for return_col in return_columns[:2]:  # Limit to 2 for demo
        result = run_mock_zeroshot_analysis(return_col, n_samples=50)
        if result:
            all_results.append(result)
        print("\n" + "-" * 60 + "\n")

    # Summary
    if all_results:
        print("MOCK ANALYSIS SUMMARY")
        print("=" * 50)

        for result in all_results:
            print(f"\n{result['return_column']}:")
            print(f"  Samples: {result['n_samples']}")
            print(f"  F1 Score: {result['f1_score']:.4f}")
            print(f"  MCC: {result['mcc']:.4f}")

    print(f"\n💡 To run the real zero-shot classifier:")
    print(f"1. Add your OpenAI API key to credentials.json")
    print(f"2. Run: python baseline/zeroshot_earnings_classifier.py")
    print(f"3. Note: Real API calls will be much slower and cost money!")

if __name__ == "__main__":
    main()