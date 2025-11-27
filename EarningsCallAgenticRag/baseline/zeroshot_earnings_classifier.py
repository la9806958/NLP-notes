#!/usr/bin/env python3
"""
Zero-Shot GPT Classifier for EarningsFilteredResults2.csv
========================================================
Adapted from baseline/zeroShotGPT/zeroShot.py to work with EarningsFilteredResults2.csv
and evaluate performance across different return horizons.

Features:
- Works with EarningsFilteredResults2.csv format
- Supports all return columns (future_3bday_cum_return, return_3d, return_7d, return_15d, return_30d)
- Incremental processing with resume capability
- Comprehensive evaluation with F1 and MCC metrics
- Rate limiting and error handling
- Compatible analysis format similar to merge_and_analyze.py

Usage:
    python baseline/zeroshot_earnings_classifier.py [return_column]

Available return columns:
- future_3bday_cum_return (default)
- return_3d
- return_7d
- return_15d
- return_30d
"""

import os
import json
import pandas as pd
import openai
import sys
import re
import time
import numpy as np
from typing import Dict, Optional
from pathlib import Path
from sklearn.metrics import f1_score, matthews_corrcoef

# --- Configuration ---
CREDENTIALS_FILE = "credentials.json"
DATA_PATH = "EarningsFilteredResults2.csv"
OUTPUT_DIR = "./zeroshot_results"
MODEL = "gpt-3.5-turbo"
RETURN_THRESHOLD = 0.02  # 2% threshold for processing priority
DEFAULT_RETURN_COLUMN = "future_3bday_cum_return"

# Available return columns for evaluation
RETURN_COLUMNS = [
    'future_3bday_cum_return',
    'return_3d',
    'return_7d',
    'return_15d',
    'return_30d'
]

# Rate limiting
REQUESTS_PER_MINUTE = 50
REQUEST_DELAY = 60.0 / REQUESTS_PER_MINUTE  # ~1.2 seconds between requests

class ZeroShotEarningsClassifier:
    def __init__(self, return_column: str = DEFAULT_RETURN_COLUMN):
        self.return_column = return_column
        self.output_dir = OUTPUT_DIR
        self.log_path = os.path.join(self.output_dir, f'zeroshot_results_{return_column}.csv')

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize OpenAI
        self._setup_openai()

        # Load data
        self.df = self._load_data()
        self.processed_df = self._load_or_create_log()

        print(f"Initialized ZeroShot classifier for return column: {return_column}")
        print(f"Dataset size: {len(self.df)} samples")
        print(f"Previously processed: {len(self.processed_df)} samples")

    def _setup_openai(self):
        """Initialize OpenAI API"""
        if os.path.exists(CREDENTIALS_FILE):
            try:
                creds = json.loads(Path(CREDENTIALS_FILE).read_text())
                openai.api_key = creds.get("openai_api_key")
                print("✅ OpenAI API key loaded from credentials.json")
            except Exception as e:
                print(f"❌ Error loading credentials: {e}")
                print("Please ensure credentials.json exists with your OpenAI API key")
                sys.exit(1)
        else:
            print(f"❌ Credentials file not found: {CREDENTIALS_FILE}")
            print("Please create credentials.json with your OpenAI API key")
            sys.exit(1)

    def _load_data(self) -> pd.DataFrame:
        """Load EarningsFilteredResults2.csv"""
        if not os.path.exists(DATA_PATH):
            print(f"❌ Data file not found: {DATA_PATH}")
            sys.exit(1)

        df = pd.read_csv(DATA_PATH)
        print(f"Loaded {len(df)} rows from {DATA_PATH}")

        # Check required columns
        required_cols = ['ticker', 'transcript', self.return_column]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
            sys.exit(1)

        # Drop rows with missing data
        original_size = len(df)
        df = df.dropna(subset=required_cols)
        if len(df) < original_size:
            print(f"Dropped {original_size - len(df)} rows with missing data")

        return df

    def _load_or_create_log(self) -> pd.DataFrame:
        """Load existing results or create new log"""
        if os.path.exists(self.log_path):
            processed_df = pd.read_csv(self.log_path)
            print(f"📄 Loaded existing results from {self.log_path}")
        else:
            processed_df = pd.DataFrame(columns=[
                "record_id", "ticker", "analysis", "directional_score",
                "error", "actual_return", "timestamp"
            ])
            print(f"🆕 Creating new results file: {self.log_path}")

        return processed_df

    def baseline_prompt(self, transcript: str) -> str:
        """Generate prompt for zero-shot classification"""
        return f"""
You are a portfolio manager analyzing an earnings call transcript.
Decide whether the stock price is likely to **increase ("Up") or decrease ("Down")**
in the next few trading days after the earnings call.

Assign a **Direction score** from 0 to 10:
- 0-2: Strong conviction of decline
- 3-4: Moderate decline expected
- 5: Neutral/uncertain
- 6-7: Moderate increase expected
- 8-10: Strong conviction of increase

---
TRANSCRIPT:
{transcript}
---

Instructions:
1. Analyze the key financial metrics, guidance, and management tone
2. Consider market context and competitive positioning
3. Assign a confidence score (0-10) based on your analysis

Respond in **exactly** this format:

<Brief analysis in 2-3 sentences>
Direction: <0-10>

""".strip()

    def call_gpt(self, prompt: str) -> str:
        """Call GPT API with rate limiting and error handling"""
        try:
            time.sleep(REQUEST_DELAY)  # Rate limiting

            resp = openai.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=500
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"OpenAI API error: {e}")

    def extract_directional_score(self, analysis: str) -> Optional[int]:
        """Extract directional score from GPT response"""
        if pd.isna(analysis) or analysis == "":
            return None

        # Look for patterns like "Direction: 8" or "Direction : 8"
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

    def already_processed(self, record_id: str) -> bool:
        """Check if record already processed"""
        if 'record_id' not in self.processed_df.columns:
            return False
        return record_id in self.processed_df['record_id'].values

    def needs_processing(self, record_id: str, actual_return: float) -> bool:
        """Determine if record needs (re)processing"""
        if not self.already_processed(record_id):
            return True

        # Check if we need to reprocess due to missing analysis or high return
        row = self.processed_df[self.processed_df['record_id'] == record_id]
        if row.empty:
            return True

        existing_analysis = row['analysis'].iloc[0]
        existing_score = row['directional_score'].iloc[0]

        no_analysis = pd.isna(existing_analysis) or existing_analysis.strip() == ""
        no_score = pd.isna(existing_score)
        high_return = abs(actual_return) >= RETURN_THRESHOLD

        return (no_analysis or no_score) and high_return

    def append_or_update_log(self, row_dict: Dict) -> None:
        """Add or update record in log"""
        record_id = row_dict['record_id']
        mask = self.processed_df['record_id'] == record_id

        if mask.any():
            # Update existing record
            for key, value in row_dict.items():
                self.processed_df.loc[mask, key] = value
        else:
            # Append new record
            self.processed_df = pd.concat([
                self.processed_df,
                pd.DataFrame([row_dict])
            ], ignore_index=True)

        # Save to disk
        self.processed_df.to_csv(self.log_path, index=False)

    def process_earnings_calls(self, max_samples: Optional[int] = None):
        """Process earnings calls with GPT analysis"""
        print(f"\nStarting zero-shot processing for {self.return_column}...")

        processed_count = 0
        skipped_count = 0
        error_count = 0

        for idx, row in self.df.iterrows():
            if max_samples and processed_count >= max_samples:
                print(f"Reached maximum sample limit: {max_samples}")
                break

            record_id = str(row.get('record_id', f'row_{idx}'))
            ticker = row['ticker']
            actual_return = row[self.return_column]

            # Check if processing needed
            if not self.needs_processing(record_id, actual_return):
                skipped_count += 1
                if skipped_count % 100 == 0:
                    print(f"⚡ Skipped {skipped_count} already processed records...")
                continue

            try:
                print(f"🧠 Processing {ticker} (record {record_id})...")

                # Generate prompt and get analysis
                prompt = self.baseline_prompt(row['transcript'])
                analysis = self.call_gpt(prompt)

                # Extract directional score
                directional_score = self.extract_directional_score(analysis)

                # Log result
                result_dict = {
                    "record_id": record_id,
                    "ticker": ticker,
                    "analysis": analysis,
                    "directional_score": directional_score,
                    "error": "",
                    "actual_return": actual_return,
                    "timestamp": pd.Timestamp.now().isoformat()
                }

                self.append_or_update_log(result_dict)
                processed_count += 1

                print(f"✅ {ticker} processed (Score: {directional_score})")

            except Exception as e:
                error_count += 1
                print(f"❌ Error processing {ticker}: {e}")

                # Log error
                error_dict = {
                    "record_id": record_id,
                    "ticker": ticker,
                    "analysis": "",
                    "directional_score": None,
                    "error": str(e),
                    "actual_return": actual_return,
                    "timestamp": pd.Timestamp.now().isoformat()
                }

                self.append_or_update_log(error_dict)

        print(f"\n🎯 Processing complete!")
        print(f"  New samples processed: {processed_count}")
        print(f"  Skipped (already done): {skipped_count}")
        print(f"  Errors: {error_count}")

    def evaluate_performance(self):
        """Evaluate classifier performance using F1 and MCC"""
        print(f"\n--- Analyzing {self.return_column} ---")

        # Load results
        if not os.path.exists(self.log_path):
            print("No results file found. Run processing first.")
            return None

        results_df = pd.read_csv(self.log_path)

        # Filter valid results with directional scores
        valid_results = results_df.dropna(subset=['directional_score', 'actual_return'])
        print(f"Valid data points: {len(valid_results)}")

        if len(valid_results) == 0:
            print("No valid results found!")
            return None

        # Convert to binary predictions (score > 5 = positive prediction)
        valid_results = valid_results.copy()
        valid_results['pred_direction'] = (valid_results['directional_score'] > 5).astype(int)
        valid_results['return_direction'] = (valid_results['actual_return'] > 0).astype(int)

        # Show examples
        print(f"\nFirst 5 examples:")
        sample_cols = ['ticker', 'directional_score', 'pred_direction', 'return_direction', self.return_column]
        if 'ticker' in valid_results.columns:
            print(valid_results[sample_cols].head())

        # Compute metrics
        y_true = valid_results['return_direction'].values
        y_pred = valid_results['pred_direction'].values

        f1 = f1_score(y_true, y_pred, average='macro')
        mcc = matthews_corrcoef(y_true, y_pred)

        results = {
            'return_column': self.return_column,
            'n_samples': len(valid_results),
            'f1_score': f1,
            'mcc': mcc
        }

        print(f"\nResults for {self.return_column}:")
        print(f"F1 Score: {f1:.4f}")
        print(f"MCC: {mcc:.4f}")
        print(f"Number of samples: {len(valid_results)}")

        return results

def main():
    """Main function - supports both single return column and all columns analysis"""

    # Parse command line arguments
    return_column = DEFAULT_RETURN_COLUMN
    if len(sys.argv) > 1:
        return_column = sys.argv[1]
        if return_column not in RETURN_COLUMNS:
            print(f"Error: Invalid return column '{return_column}'")
            print(f"Available columns: {RETURN_COLUMNS}")
            sys.exit(1)

    # Initialize classifier
    classifier = ZeroShotEarningsClassifier(return_column)

    # Process earnings calls - set max_samples=None for full dataset
    max_samples = None  # Full dataset - change to smaller number for testing
    classifier.process_earnings_calls(max_samples=max_samples)

    # Evaluate performance
    results = classifier.evaluate_performance()

    if results:
        print(f"\n{'='*50}")
        print("ZERO-SHOT GPT RESULTS")
        print(f"{'='*50}")
        print(f"Return column: {results['return_column']}")
        print(f"Samples: {results['n_samples']}")
        print(f"F1 Score: {results['f1_score']:.4f}")
        print(f"MCC: {results['mcc']:.4f}")

if __name__ == "__main__":
    main()