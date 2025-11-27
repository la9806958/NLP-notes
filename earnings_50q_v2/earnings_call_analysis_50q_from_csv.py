#!/usr/bin/env python3
"""
Earnings Call Analysis Agent with 50 Questions - CSV Processing Version
Modified version that processes the "analysis" column from earnings_analysis_results_pre_2023.csv.

This agent takes in earnings_analysis_results_pre_2023.csv, processes each row's "analysis" column,
and extracts scores for 50 different growth-focused questions using historical context from previous calls.
Output CSV contains all original columns plus 50 score columns (q1-q50).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from openai import OpenAI
import os
import json
import csv
from typing import Dict, List, Optional, Tuple
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import re
import threading
import fcntl
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EarningsCallAnalysisAgent:
    def __init__(self, credentials_path: str = None, api_key: str = None, log_file: str = "earnings_analysis_prompts.log"):
        """Initialize the earnings analysis agent with OpenAI API key from credentials file or direct key."""
        if credentials_path:
            with open(credentials_path, 'r') as f:
                credentials = json.load(f)
            api_key = credentials.get('openai_api_key')
            if not api_key:
                raise ValueError("openai_api_key not found in credentials.json")
        elif not api_key:
            # Try to load from base folder credentials.json as default
            try:
                with open('/home/lichenhui/credentials.json', 'r') as f:
                    credentials = json.load(f)
                api_key = credentials.get('openai_api_key')
                if not api_key:
                    raise ValueError("openai_api_key not found in credentials.json")
            except FileNotFoundError:
                raise ValueError("Either credentials_path or api_key must be provided, or credentials.json must exist in base folder")

        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"

        # Set up prompt logging
        self.prompt_log_file = log_file
        # Thread lock for safe file operations
        self.file_lock = threading.Lock()

        # Clear the log file at start
        with open(self.prompt_log_file, 'w') as f:
            f.write(f"Earnings Call Analysis Agent Prompt Log - Started at {datetime.now()}\n")
            f.write("="*80 + "\n\n")

        # Set up token usage logging
        self.token_log_file = log_file.replace('.log', '_tokens.log')
        with open(self.token_log_file, 'w') as f:
            f.write(f"Earnings Call Analysis Agent Token Usage Log - Started at {datetime.now()}\n")
            f.write("timestamp,step,ticker,date,input_tokens,output_tokens,total_tokens\n")

    def load_earnings_data(self, csv_path: str, columns_to_load: List[str] = None, minimal: bool = False) -> pd.DataFrame:
        """Load and preprocess the earnings analysis CSV with only required columns."""
        try:
            # Define required columns for processing
            required_columns = ['ticker', 'et_timestamp', 'analysis']

            if minimal:
                # For minimal load, only load the required columns for processing
                usecols = required_columns
            else:
                # If specific columns requested, use them; otherwise load only required columns
                if columns_to_load is None:
                    # First, read just the header to get all column names
                    with open(csv_path, 'r') as f:
                        reader = csv.reader(f)
                        all_columns = next(reader)
                    columns_to_load = all_columns

                # Always include required columns
                usecols = list(set(columns_to_load) | set(required_columns))

            # Load CSV with only necessary columns
            df = pd.read_csv(csv_path, usecols=usecols)

            # Ensure required columns exist
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in CSV: {missing_columns}")

            # Convert et_timestamp column to datetime
            df['et_timestamp'] = pd.to_datetime(df['et_timestamp'])

            # Sort by ticker and et_timestamp for easier processing
            df = df.sort_values(['ticker', 'et_timestamp']).reset_index(drop=True)

            logger.info(f"Loaded {len(df)} earnings call analysis entries with {len(df.columns)} columns")
            logger.info(f"Timestamp range: {df['et_timestamp'].min()} to {df['et_timestamp'].max()}")
            logger.info(f"Unique tickers: {df['ticker'].nunique()}")

            return df

        except Exception as e:
            logger.error(f"Error loading earnings data: {e}")
            raise

    def get_historical_context(self, df: pd.DataFrame, current_ticker: str, current_timestamp: datetime,
                              max_history: int = 4) -> List[Dict]:
        """Get up to 4 historical earnings analyses for the ticker that predate the current analysis."""
        # Filter for same ticker and timestamps before current timestamp
        historical = df[
            (df['ticker'] == current_ticker) &
            (df['et_timestamp'] < current_timestamp)
        ].copy()

        # Sort by timestamp descending (most recent first) and take up to max_history
        historical = historical.sort_values('et_timestamp', ascending=False).head(max_history)

        # Convert to list of dictionaries
        context = []
        for _, row in historical.iterrows():
            context.append({
                'ticker': row['ticker'],
                'timestamp': row['et_timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'analysis': row['analysis']
            })

        # Sort by timestamp ascending (oldest first) for chronological context
        context = sorted(context, key=lambda x: x['timestamp'])

        logger.debug(f"Found {len(context)} historical analyses for {current_ticker} before {current_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        return context

    def create_earnings_analysis_prompt(self, ticker: str, current_date: str, current_analysis: str,
                                      historical_context: List[Dict]) -> str:
        """Create the prompt for earnings call analysis with historical context."""

        prompt = f"""You are a portfolio manager who is reading notes on an earnings call and you have access to a history of notes on the particular firm:

Your notes on the current call you are reading is:

TICKER: {ticker}
DATE: {current_date}
CURRENT ANALYSIS:
{current_analysis}

You have access to a history of previous notes:"""

        # Add historical context if available
        if historical_context:
            prompt += "\n\nPREVIOUS NOTES:\n"
            for i, hist in enumerate(historical_context, 1):
                prompt += f"\n--- Previous Analysis {i} ---\n"
                prompt += f"Timestamp: {hist['timestamp']}\n"
                prompt += f"Analysis: {hist['analysis']}\n"
        else:
            prompt += "\n\nPREVIOUS NOTES: No historical data available for this ticker.\n"

        prompt += """

1. Element of Surprise (Direction-Focused)
1. Did revenue growth surprise to the upside or downside versus expectations? (0 = sharply below, 10 = sharply above)
2. Was the direction of margin change more favorable or unfavorable than anticipated? (0 = severe contraction, 10 = strong expansion)
3. Did volume or demand trends accelerate or decelerate unexpectedly? (0 = major deceleration, 10 = major acceleration)
4. Was the growth trajectory of key segments stronger or weaker than the market modeled? (0 = much weaker, 10 = much stronger)
5. Did the pace of customer acquisition/retention move in an unexpected direction? (0 = sharply lower, 10 = sharply higher)
6. Were cost trends more favorable or unfavorable than expected in terms of growth impact? (0 = large negative surprise, 10 = large positive surprise)
7. Did operating leverage come in stronger or weaker than anticipated? (0 = weaker, 10 = stronger)
8. Was the direction of free cash flow growth a positive or negative surprise? (0 = strong decline, 10 = strong increase)
9. Did international or regional growth outperform or underperform expectations? (0 = materially underperformed, 10 = materially outperformed)
10. Did management characterize the growth direction (accelerating vs slowing) in line with reality? (0 = misleading, 10 = fully aligned and transparent)

2. Forward-Looking Statements & Delivery
1. How credible are management's claims about sustaining growth momentum? (0 = not credible, 10 = highly credible)
2. How strong is implied acceleration or stability in guidance relative to history? (0 = strong deceleration, 10 = strong acceleration)
3. How consistent has management been in delivering on growth forecasts of this type? (0 = poor history, 10 = excellent track record)
4. How well are forward growth rates supported by operational detail? (0 = vague, 10 = fully detailed and credible)
5. How aligned are investments with growth ambitions? (0 = misaligned, 10 = perfectly aligned)
6. How resilient are the targets to macro assumptions? (0 = fragile, 10 = highly resilient)
7. How openly did management acknowledge risks to growth? (0 = no acknowledgement, 10 = fully transparent)
8. How specific were commitments around timing and scale of growth? (0 = vague, 10 = highly specific)
9. How strong is historical precedent for executing on similar growth plans? (0 = none, 10 = strong precedent)
10. To what extent do forward statements shift the long-term growth trajectory? (0 = negative inflection, 10 = highly positive inflection)

3. Financial Performance
1. How strong was absolute revenue growth versus last quarter/year? (0 = contraction, 10 = very strong growth)
2. Did margins expand or contract in a meaningful way? (0 = sharp contraction, 10 = sharp expansion)
3. How aligned was cash flow with reported earnings growth? (0 = large divergence, 10 = fully aligned and positive)
4. Was EBITDA growth stronger than top-line growth (operating leverage)? (0 = negative leverage, 10 = strong positive leverage)
5. Did balance sheet movements strengthen growth potential? (0 = deteriorated, 10 = materially improved)
6. Was gross profit growth faster than revenue growth? (0 = lagged materially, 10 = led materially)
7. How clearly did management explain sequential revenue/cost changes? (0 = vague, 10 = highly transparent)
8. Was net income trajectory consistent and stable? (0 = volatile, 10 = smooth and strong)
9. How sustainable were growth rates achieved this quarter? (0 = unsustainable, 10 = highly sustainable)
10. How well do results align with longer-term growth averages? (0 = below average, 10 = well above average)

4. Key Events
1. Did M&A transactions strengthen growth prospects? (0 = highly dilutive, 10 = highly accretive)
2. How meaningful were new product launches for growth acceleration? (0 = insignificant, 10 = highly transformative)
3. Did partnerships/joint ventures materially enhance growth trajectory? (0 = immaterial, 10 = highly material)
4. Were divestitures framed as value-destructive or growth-enhancing? (0 = destructive, 10 = strongly positive refocus)
5. How impactful were regulatory developments on growth? (0 = severe drag, 10 = major tailwind)
6. Were macroeconomic factors framed as slowing or accelerating growth? (0 = strong headwind, 10 = strong tailwind)
7. Did management frame headwinds as temporary vs. structural? (0 = structural drag, 10 = short-term only)
8. How well quantified was the growth impact of strategic initiatives? (0 = unquantified, 10 = clearly quantified)
9. How credible were timelines for growth from new initiatives? (0 = unrealistic, 10 = highly credible)
10. Did management present shocks as growth opportunities? (0 = defensive posture, 10 = opportunistic and positive)

5. Narrative & Language Quality, Q&A Integrity
1. How directly did management address growth acceleration/deceleration? (0 = evasive, 10 = fully direct)
2. Was there consistency between prepared remarks and Q&A on growth? (0 = inconsistent, 10 = highly consistent)
3. Did tone convey confidence in sustaining growth? (0 = weak, 10 = highly confident)
4. How transparent was management about headwinds? (0 = avoided, 10 = candid)
5. How concrete were explanations for growth changes? (0 = vague, 10 = highly specific)
6. How well did management quantify growth drivers? (0 = qualitative only, 10 = fully quantitative)
7. Were terminology and descriptors consistent with past calls? (0 = inconsistent, 10 = consistent)
8. How responsive were executives when pressed on growth? (0 = avoided, 10 = fully responsive)
9. Did management reframe negative growth as "normalization" convincingly? (0 = unconvincing, 10 = highly convincing)
10. How consistent was the language around guidance vs. numbers provided? (0 = contradictory, 10 = fully consistent)

Provide your ratings in a list of values from 0 - 10.

Please provide your response in the following JSON format:
{
    "ticker": \"""" + ticker + """\",
    "date": \"""" + current_date + """\",
    "scores": [score1, score2, score3, ..., score50]
}

Provide only the JSON response with exactly 50 numerical scores (0-10) in the scores array.
"""
        return prompt

    def get_earnings_response(self, ticker: str, current_date: str, current_analysis: str,
                             historical_context: List[Dict]) -> Optional[Dict]:
        """Get response for earnings call analysis using LLM."""
        prompt = self.create_earnings_analysis_prompt(ticker, current_date, current_analysis, historical_context)

        # Log the prompt to file for examination
        self.log_prompt_to_file("EARNINGS_ANALYSIS", prompt, ticker, current_date)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert financial analyst and portfolio manager. Provide analysis in the requested JSON format with exactly 50 numerical scores (0-10)."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=2000
            )

            # Log token usage
            usage = response.usage
            self.log_token_usage("EARNINGS_ANALYSIS", ticker, current_date, usage.prompt_tokens, usage.completion_tokens)

            # Parse the JSON response
            response_text = response.choices[0].message.content.strip()

            logger.debug(f"Raw LLM response for {ticker} on {current_date}: {response_text[:200]}...")

            # Extract JSON from response
            json_str = self.extract_json(response_text)
            if not json_str:
                logger.warning(f"Could not extract JSON from response for {ticker} on {current_date}")
                return None

            # Validate JSON
            try:
                result = json.loads(json_str)
                # Validate structure
                if 'ticker' in result and 'date' in result and 'scores' in result:
                    scores = result['scores']
                    if isinstance(scores, list) and len(scores) == 50:
                        # Validate all scores are numbers between 0-10
                        if all(isinstance(score, (int, float)) and 0 <= score <= 10 for score in scores):
                            logger.info(f"Successfully parsed response for {ticker} on {current_date}")
                            return result
                        else:
                            logger.warning(f"Invalid scores in response for {ticker} on {current_date}")
                    else:
                        logger.warning(f"Expected 50 scores, got {len(scores) if isinstance(scores, list) else 'non-list'} for {ticker} on {current_date}")
                else:
                    logger.warning(f"Missing required fields in response for {ticker} on {current_date}")
                return None

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing error for {ticker} on {current_date}: {e}")
                return None

        except Exception as e:
            logger.error(f"Error getting response for {ticker} on {current_date}: {e}")
            return None

    def extract_json(self, response_text: str) -> Optional[str]:
        """Extract JSON from LLM response."""
        # First try to find JSON in code blocks
        if "```json" in response_text:
            start_marker = "```json"
            end_marker = "```"
            start_idx = response_text.find(start_marker) + len(start_marker)
            end_idx = response_text.find(end_marker, start_idx)
            if start_idx != -1 and end_idx != -1:
                return response_text[start_idx:end_idx].strip()

        # Fallback to brace extraction
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        if start_idx != -1 and end_idx != -1:
            return response_text[start_idx:end_idx]

        return None

    def log_prompt_to_file(self, step: str, prompt: str, ticker: str = None, date: str = None):
        """Log prompt to the dedicated prompt log file with thread safety."""
        with self.file_lock:
            with open(self.prompt_log_file, 'a', encoding='utf-8') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"TIMESTAMP: {datetime.now()}\n")
                    f.write(f"STEP: {step}\n")
                    if ticker:
                        f.write(f"TICKER: {ticker}\n")
                    if date:
                        f.write(f"DATE: {date}\n")
                    f.write(f"{'='*80}\n")
                    f.write(f"PROMPT:\n")
                    f.write(prompt)
                    f.write(f"\n{'='*80}\n\n")
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def log_token_usage(self, step: str, ticker: str, date: str, input_tokens: int, output_tokens: int):
        """Log token usage to CSV file with thread safety."""
        total_tokens = input_tokens + output_tokens
        timestamp = datetime.now().isoformat()
        with self.file_lock:
            with open(self.token_log_file, 'a') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(f"{timestamp},{step},{ticker},{date},{input_tokens},{output_tokens},{total_tokens}\n")
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def write_csv_header(self, output_path: str, original_columns: List[str]):
        """Write CSV header for results - includes all original columns plus 50 score columns."""
        score_columns = [f'q{i}' for i in range(1, 51)]
        header = original_columns + score_columns
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    def append_to_csv(self, output_path: str, original_row: Dict, scores: List[float], original_columns: List[str]):
        """Append results to CSV file with thread safety - includes all original columns plus scores."""
        # Build row with original columns in order, then scores
        row = [original_row.get(col, '') for col in original_columns] + scores
        with self.file_lock:
            with open(output_path, 'a', newline='', encoding='utf-8') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    writer = csv.writer(f)
                    writer.writerow(row)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def process_earnings_calls(self, csv_path: str, output_path: str = "earnings_analysis_50q_results.csv",
                             max_workers: int = None, parallel: bool = True) -> int:
        """Process all earnings calls from the CSV file."""
        # Load the earnings data - load ALL columns first to get original_columns
        logger.info("Loading earnings data for metadata...")
        df_full = self.load_earnings_data(csv_path, minimal=False)

        # Store original columns for output
        original_columns = df_full.columns.tolist()

        # Now reload with only required columns for processing
        logger.info("Reloading with minimal columns for processing...")
        df = self.load_earnings_data(csv_path, minimal=True)

        # Free the full dataframe
        del df_full
        gc.collect()

        # Initialize output CSV
        if not os.path.exists(output_path):
            self.write_csv_header(output_path, original_columns)
            logger.info(f"Initialized CSV output file: {output_path}")
        else:
            logger.info(f"Using existing CSV output file: {output_path}")

        # Check for already processed records
        completed_pairs = set()
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            try:
                existing_df = pd.read_csv(output_path)
                if len(existing_df) > 0:
                    completed_pairs = set((row['ticker'], str(pd.to_datetime(row['et_timestamp']))) for _, row in existing_df.iterrows())
                    logger.info(f"Found {len(completed_pairs)} already processed records")
            except Exception as e:
                logger.warning(f"Could not read existing output file: {e}")

        # Build ticker-based processing structure and historical data from minimal df
        ticker_records = {}  # ticker -> list of records to process
        historical_data = {}  # ticker -> list of all historical records

        logger.info("Building ticker-based processing structure and historical data...")
        for idx, row in df.iterrows():
            ticker = row['ticker']
            timestamp = row['et_timestamp']

            # Build historical data for all records
            if ticker not in historical_data:
                historical_data[ticker] = []
            historical_data[ticker].append({
                'ticker': ticker,
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'analysis': row['analysis']
            })

            # Check if this record needs processing
            key = (ticker, str(timestamp))
            if key not in completed_pairs:
                # Need to get original_row with all columns
                original_row = {col: '' for col in original_columns}
                original_row['ticker'] = ticker
                original_row['et_timestamp'] = timestamp
                original_row['analysis'] = row['analysis']

                record = {
                    'index': idx,
                    'ticker': ticker,
                    'date': timestamp.strftime('%Y-%m-%d'),
                    'timestamp': timestamp,
                    'analysis': row['analysis'],
                    'original_row': original_row
                }

                if ticker not in ticker_records:
                    ticker_records[ticker] = []
                ticker_records[ticker].append(record)

        total_records = sum(len(records) for records in ticker_records.values())
        logger.info(f"Processing {total_records} earnings call records across {len(ticker_records)} tickers (skipping {len(completed_pairs)} already processed)")

        # Free the main dataframe - we don't need it anymore
        del df
        gc.collect()
        logger.info("Historical data pre-computed and main dataframe freed from memory")

        if total_records == 0:
            logger.info("All records already processed. Nothing to do.")
            return 0

        # Process records by ticker in parallel
        total_processed = 0
        total_failed = 0

        if parallel and max_workers != 1:
            # Parallel processing by ticker
            if max_workers is None:
                max_workers = min(mp.cpu_count(), 15)

            logger.info(f"Using {max_workers} workers for parallel processing across {len(ticker_records)} tickers")

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Create partial function with fixed parameters
                process_func = partial(
                    process_ticker_worker,
                    historical_data=historical_data,
                    output_path=output_path,
                    api_key=self.api_key,
                    model=self.model,
                    prompt_log_file=self.prompt_log_file,
                    token_log_file=self.token_log_file,
                    original_columns=original_columns
                )

                # Submit all tickers for processing
                future_to_ticker = {
                    executor.submit(process_func, ticker, records): ticker
                    for ticker, records in ticker_records.items()
                }

                # Collect results as they complete
                completed_tickers = 0
                for future in as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    try:
                        ticker_name, processed, failed = future.result(timeout=3600)  # 1 hour timeout per ticker
                        total_processed += processed
                        total_failed += failed
                        completed_tickers += 1
                        logger.info(f"Completed ticker {ticker_name}: {processed} processed, {failed} failed ({completed_tickers}/{len(ticker_records)} tickers, {total_processed}/{total_records} records)")
                    except Exception as e:
                        logger.error(f"Error processing ticker {ticker}: {e}")
                        # Count all records for this ticker as failed
                        total_failed += len(ticker_records[ticker])
                    finally:
                        # Force garbage collection periodically
                        if completed_tickers % 10 == 0:
                            gc.collect()

                # Final garbage collect
                gc.collect()
        else:
            # Sequential processing by ticker
            logger.info("Using sequential processing")
            for ticker_idx, (ticker, records) in enumerate(ticker_records.items()):
                logger.info(f"Processing ticker {ticker} ({ticker_idx + 1}/{len(ticker_records)}) with {len(records)} records")

                ticker_name, processed, failed = process_ticker_worker(
                    ticker, records, historical_data, output_path,
                    self.api_key, self.model, self.prompt_log_file,
                    self.token_log_file, original_columns
                )

                total_processed += processed
                total_failed += failed
                logger.info(f"Completed ticker {ticker_name}: {processed} processed, {failed} failed")

                # Force garbage collection every 10 tickers
                if (ticker_idx + 1) % 10 == 0:
                    gc.collect()

        logger.info(f"Completed processing. Total records processed: {total_processed}, failed: {total_failed}")
        logger.info(f"Results saved to: {output_path}")

        # Final garbage collection
        gc.collect()

        return total_processed


def process_ticker_worker(ticker: str, ticker_records: List[Dict], historical_data: Dict[str, List[Dict]],
                         output_path: str, api_key: str, model: str, prompt_log_file: str,
                         token_log_file: str, original_columns: List[str]) -> Tuple[str, int, int]:
    """Worker function to process all records for a single ticker in parallel."""
    processed = 0
    failed = 0

    try:
        # Create a new agent instance for this worker
        worker_log_file = f"{prompt_log_file}.{ticker}"
        agent = EarningsCallAnalysisAgent(api_key=api_key, log_file=worker_log_file)
        agent.model = model

        # Get historical data for this ticker
        ticker_history = historical_data.get(ticker, [])

        # Sort records by timestamp
        ticker_records = sorted(ticker_records, key=lambda x: x['timestamp'])

        # Process each record for this ticker
        for record in ticker_records:
            try:
                current_ts = pd.to_datetime(record['timestamp'])

                # Get historical context from pre-computed data
                historical_context = [
                    h for h in ticker_history
                    if pd.to_datetime(h['timestamp']) < current_ts
                ]
                historical_context = sorted(historical_context, key=lambda x: x['timestamp'])[-4:]

                # Get LLM response
                response = agent.get_earnings_response(
                    record['ticker'],
                    record['date'],
                    record['analysis'],
                    historical_context
                )

                if response and 'scores' in response:
                    # Log results to CSV
                    agent.append_to_csv(
                        output_path,
                        record['original_row'],
                        response['scores'],
                        original_columns
                    )
                    processed += 1
                else:
                    failed += 1
                    logger.warning(f"Failed to get response for {ticker} on {record['date']}")

            except Exception as e:
                failed += 1
                logger.error(f"Error processing {ticker} on {record['date']}: {e}")

        # Clean up
        del agent
        gc.collect()

        return (ticker, processed, failed)

    except Exception as e:
        logger.error(f"Worker error for ticker {ticker}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        gc.collect()
        return (ticker, processed, failed)


def main():
    parser = argparse.ArgumentParser(description='Earnings Call Analysis Agent with 50 Questions - CSV Processing Version')
    parser.add_argument('--credentials', default='credentials.json',
                       help='Path to credentials JSON file (default: credentials.json)')
    parser.add_argument('--api-key', help='OpenAI API key (alternative to credentials file)')
    parser.add_argument('--input', default='/home/lichenhui/earnings_analysis_results_pre_2023.csv',
                       help='Path to input CSV file with earnings analysis')
    parser.add_argument('--output', default='earnings_analysis_50q_results_pre_2023.csv',
                       help='Output CSV file path')
    parser.add_argument('--parallel', action='store_true', default=True,
                       help='Enable parallel processing (default: True)')
    parser.add_argument('--sequential', action='store_true',
                       help='Force sequential processing (overrides --parallel)')
    parser.add_argument('--max-workers', type=int, default=8,
                       help='Maximum number of workers for parallel processing (default: 4)')
    parser.add_argument('--log-dir', default='.',
                       help='Directory for log files (default: current directory)')

    args = parser.parse_args()

    # Create log file path
    log_file_path = os.path.join(args.log_dir, "earnings_analysis_prompts.log")

    # Initialize agent with credentials file or API key
    if args.api_key:
        agent = EarningsCallAnalysisAgent(api_key=args.api_key, log_file=log_file_path)
    else:
        agent = EarningsCallAnalysisAgent(credentials_path=args.credentials, log_file=log_file_path)

    # Determine if parallel processing should be used
    use_parallel = args.parallel and not args.sequential

    total_processed = agent.process_earnings_calls(
        args.input,
        args.output,
        max_workers=args.max_workers,
        parallel=use_parallel
    )

    print(f"Processed {total_processed} earnings call records. Results saved to {args.output}")


if __name__ == "__main__":
    main()
