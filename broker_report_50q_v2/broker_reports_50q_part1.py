#!/usr/bin/env python3
"""
Script to process financial reports using OpenAI GPT-4o-mini model.

This script:
1. Gets US tickers from close_to_close_returns_matrix.csv
2. Filters optimized_financial_results.csv for files containing multiple US tickers
3. Processes text files with GPT-4o-mini model via OpenAI API
"""

import pandas as pd
import os
import re
import time
import csv
import json
from collections import defaultdict
import logging
from openai import OpenAI
from multiprocessing import Pool, cpu_count
from functools import partial
import concurrent.futures

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for continuous logging
CSV_OUTPUT_FILE = '/home/lichenhui/financial_analysis_4o_mini_results.csv'
CSV_HEADERS = [
    'timestamp', 'file_path', 'ticker', 'analysis_output', 'total_latency_ms',
    'api_latency_ms', 'input_tokens', 'output_tokens', 'total_tokens',
    'text_length_chars', 'cost_usd'
]

# OpenAI API pricing for GPT-4o-mini (as of 2024)
PRICING_INPUT_PER_1K = 0.00015  # $0.15 per 1M tokens
PRICING_OUTPUT_PER_1K = 0.0006  # $0.60 per 1M tokens

def init_continuous_logging():
    """Initialize the continuous logging CSV file if it doesn't exist."""
    if not os.path.exists(CSV_OUTPUT_FILE):
        # Match the original broker_info_extraction format
        headers = ['timestamp', 'file_path', 'full_directory', 'filename', 'extracted_date',
                  'firm_from_directory', 'extracted_firm', 'gpt_response', 'total_latency_ms',
                  'api_latency_ms', 'input_tokens', 'output_tokens', 'total_tokens',
                  'text_length_chars', 'cost_usd']

        with open(CSV_OUTPUT_FILE, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
        logger.info(f"Initialized continuous logging to {CSV_OUTPUT_FILE}")
    else:
        logger.info(f"Using existing file {CSV_OUTPUT_FILE}")

def calculate_cost(input_tokens, output_tokens):
    """Calculate the cost of the API call."""
    input_cost = (input_tokens / 1000) * PRICING_INPUT_PER_1K
    output_cost = (output_tokens / 1000) * PRICING_OUTPUT_PER_1K
    return input_cost + output_cost

def log_result_continuously(result_data):
    """Append a single result to the continuous logging CSV."""
    with open(CSV_OUTPUT_FILE, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            result_data['timestamp'],
            result_data['file_path'],
            result_data['full_directory'],
            result_data['filename'],
            result_data['extracted_date'],
            result_data['firm_from_directory'],
            result_data['extracted_firm'],
            result_data['gpt_response'],
            result_data['total_latency_ms'],
            result_data['api_latency_ms'],
            result_data['input_tokens'],
            result_data['output_tokens'],
            result_data['total_tokens'],
            result_data['text_length_chars'],
            result_data['cost_usd']
        ])

def get_processed_file_paths():
    """Get already processed file_paths from existing results."""
    processed_files = set()
    if os.path.exists(CSV_OUTPUT_FILE):
        try:
            # Try to read with error handling for malformed CSV
            with open(CSV_OUTPUT_FILE, 'r') as f:
                lines = f.readlines()
                for line in lines[1:]:  # Skip header
                    parts = line.split(',')
                    if len(parts) >= 2:
                        processed_files.add(parts[1])  # file_path is 2nd column
            logger.info(f"Found {len(processed_files)} already processed file paths")
        except Exception as e:
            logger.warning(f"Error reading existing results file: {e}")
            # Fallback to empty set if we can't read the file
            processed_files = set()
    return processed_files

def clean_ticker(ticker_str):
    """Clean ticker string to extract just the ticker symbol."""
    if pd.isna(ticker_str) or ticker_str == 'N/A' or ticker_str == '':
        return None

    # Remove common suffixes and extract just the ticker
    ticker = str(ticker_str).strip()

    # Handle various formats like "ZOZO (3092.T)", "4507.JP", etc.
    # Extract the main ticker before any parentheses or dots
    ticker = re.split(r'[\s\(\.]', ticker)[0]

    return ticker if ticker and ticker != 'N/A' else None

def load_and_join_data():
    """Load and join the broker info and extracted tickers data."""
    print("📂 Loading broker_info_extraction_4o_mini_final_enhanced.csv...")
    logger.info("Loading broker_info_extraction_4o_mini_final_enhanced.csv...")
    broker_df = pd.read_csv('/home/lichenhui/broker_info_extraction_4o_mini_final_enhanced.csv')
    print(f"   Loaded {len(broker_df)} rows from broker data")

    print("📂 Loading extracted_tickers.csv...")
    logger.info("Loading extracted_tickers.csv...")
    ticker_df = pd.read_csv('/home/lichenhui/extracted_tickers.csv')
    print(f"   Loaded {len(ticker_df)} rows from ticker data")

    # Join on full_file_path (from ticker_df) and file_path (from broker_df)
    print("🔗 Joining data on file directory...")
    logger.info("Joining data on file directory...")
    joined_df = pd.merge(ticker_df, broker_df,
                        left_on='full_file_path', right_on='file_path',
                        how='inner')

    print(f"   Joined data has {len(joined_df)} rows")
    logger.info(f"Joined data has {len(joined_df)} rows")

    # Clean ticker column to remove NaN and invalid entries
    print("🧹 Cleaning ticker data...")
    before_clean = len(joined_df)
    joined_df = joined_df[joined_df['ticker'].notna() & (joined_df['ticker'] != 'N/A') & (joined_df['ticker'] != '')]
    print(f"   Removed {before_clean - len(joined_df)} invalid ticker entries")

    # Deduplicate by unique file names and group tickers
    print("📋 Deduplicating by unique file names...")
    logger.info("Deduplicating by unique file names...")
    file_tickers = joined_df.groupby(['full_file_path', 'extracted_date', 'extracted_firm'])['ticker'].apply(lambda x: list(set(x))).reset_index()
    print(f"   Found {len(file_tickers)} unique files")

    # Filter to only files with multiple tickers
    multi_ticker_files = file_tickers[file_tickers['ticker'].apply(len) > 1]
    print(f"   Found {len(multi_ticker_files)} files with multiple tickers")

    logger.info(f"Found {len(multi_ticker_files)} unique files with multiple tickers after deduplication")

    return multi_ticker_files, joined_df

def load_credentials():
    """Load credentials from credentials.json file."""
    credentials_file = '/home/lichenhui/credentials.json'

    if not os.path.exists(credentials_file):
        raise FileNotFoundError(f"Credentials file not found: {credentials_file}")

    try:
        with open(credentials_file, 'r') as f:
            credentials = json.load(f)

        if 'openai_api_key' not in credentials:
            raise KeyError("openai_api_key not found in credentials file")

        return credentials
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in credentials file: {e}")

def init_openai_client():
    """Initialize OpenAI client."""
    logger.info("Initializing OpenAI client...")

    # Try to load API key from credentials file first
    try:
        credentials = load_credentials()
        api_key = credentials['openai_api_key']
        logger.info("API key loaded from credentials.json")
    except (FileNotFoundError, KeyError, ValueError) as e:
        logger.warning(f"Failed to load credentials from file: {e}")
        # Fallback to environment variable
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in credentials.json or environment variable")
        logger.info("API key loaded from environment variable")

    client = OpenAI(api_key=api_key)
    logger.info("OpenAI client initialized successfully")
    return client

def generate_multi_ticker_analysis(client, text, tickers, processed_tickers=None, report_summary=None):
    """Generate analysis for multiple tickers incrementally to save input tokens."""
    start_time = time.time()

    if processed_tickers is None:
        processed_tickers = {}

    # Create incremental prompt to save input tokens
    if not processed_tickers:
        # First ticker - establish context with full report and create summary for future use
        current_ticker = tickers[0]
        prompt = f"You are a portfolio manager reviewing a sector equity research report covering multiple tickers. For {current_ticker}, write a concise yet information-rich summary describing how its price is likely to evolve following the publication of the report, with particular attention to differences relative to other tickers discussed. Incorporate as many relevant details as possible from the report for {current_ticker}, including the tone of analyst commentary, valuation arguments, catalysts, risks, and any forward-looking guidance, and clearly contrast these with the coverage of the other securities."

        system_message = "You are an expert financial analyst and portfolio manager."
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Research Report:\n{text}\n\nPrompt:\n{prompt}"}
        ]
    else:
        # Subsequent tickers - use only previous analyses and report summary, NO full report text
        current_ticker = tickers[len(processed_tickers)]
        previous_analyses = "\n\n".join([f"{ticker}: {analysis}" for ticker, analysis in processed_tickers.items()])

        # Build context-aware prompt without full report text
        prompt = f"You are a portfolio manager who previously analyzed a sector equity research report covering tickers: {', '.join(tickers)}. Based on your previous analyses below from the same report, now analyze {current_ticker}. Write a concise yet information-rich summary for {current_ticker}, clearly contrasting it with your previous analyses. Focus on relative differences in analyst tone, valuation arguments, catalysts, and risks.\n\nPrevious analyses from the same report:\n{previous_analyses}\n\nNow provide analysis for {current_ticker} with clear contrasts to the above."

        messages = [
            {"role": "system", "content": "You are an expert financial analyst. You have already analyzed other tickers from this report. Now analyze the next ticker using context from your previous analyses."},
            {"role": "user", "content": prompt}
        ]

    # Log token savings
    if processed_tickers:
        logger.info(f"    TOKEN SAVING: For {current_ticker} (ticker {len(processed_tickers)+1}/{len(tickers)}), using incremental prompt (NO full report text)")
    else:
        logger.info(f"    FULL CONTEXT: For {current_ticker} (ticker 1/{len(tickers)}), using full report text")

    # Make API call
    api_start = time.time()
    try:
        print(f"      🤖 Calling OpenAI API for {current_ticker}...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=400,
            temperature=0.7,
            top_p=1.0
        )
        api_time = time.time() - api_start
        print(f"      ✅ API call completed in {api_time:.2f}s")

        # Extract response
        analysis = response.choices[0].message.content.strip()

        # Extract token usage
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens

        # Calculate cost
        cost = calculate_cost(input_tokens, output_tokens)

        total_time = time.time() - start_time

        # Log the generated output
        print(f"      📝 Generated analysis for {current_ticker}: {len(analysis)} chars")
        logger.info(f"    OUTPUT for {current_ticker}: {analysis[:100]}...")  # Truncate log for readability

        # Log timing and cost information with token savings info
        if processed_tickers:
            logger.info(f"    TOKENS SAVED: Input={input_tokens} (vs ~8000+ with full report)")
        logger.info(f"    Timing for {current_ticker}: Total={total_time:.3f}s, API={api_time:.3f}s, Cost=${cost:.6f}")
        logger.info(f"    Tokens for {current_ticker}: Input={input_tokens}, Output={output_tokens}, Total={total_tokens}")

        return current_ticker, analysis, {
            'total_latency_ms': total_time * 1000,
            'api_latency_ms': api_time * 1000,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'cost_usd': cost
        }

    except Exception as e:
        logger.error(f"Error calling OpenAI API for {current_ticker}: {e}")
        total_time = time.time() - start_time
        return current_ticker, f"Error: {str(e)}", {
            'total_latency_ms': total_time * 1000,
            'api_latency_ms': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0,
            'cost_usd': 0
        }

def process_single_file_all_tickers(args):
    """Process all tickers in a single file incrementally to save input tokens."""
    file_path, tickers, extracted_date, firm, api_key = args

    print(f"\n🔄 Processing file: {os.path.basename(file_path)}")
    print(f"   Tickers: {tickers}")
    print(f"   Firm: {firm}")

    try:
        # Initialize OpenAI client for this worker
        client = OpenAI(api_key=api_key)

        # Check if file exists and read it
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # Limit text length to avoid excessive API costs and context limits
            if len(text) > 15000:
                text = text[:15000] + "... [truncated]"

            print(f"   📄 File length: {len(text)} characters")
            logger.info(f"Processing file {file_path} with {len(tickers)} tickers: {', '.join(tickers)}")
            logger.info(f"TOKEN OPTIMIZATION: First ticker gets full context, subsequent {len(tickers)-1} tickers use incremental approach")
            print(f"   🎯 Processing {len(tickers)} tickers with token optimization")

            results = []
            processed_tickers = {}

            # Process each ticker incrementally
            for i in range(len(tickers)):
                current_ticker, analysis, timing_info = generate_multi_ticker_analysis(
                    client, text, tickers, processed_tickers
                )

                # Store this analysis for context in next ticker
                processed_tickers[current_ticker] = analysis

                # Prepare result data matching original broker format
                result_data = {
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'file_path': file_path,
                    'full_directory': os.path.dirname(file_path),
                    'filename': os.path.basename(file_path),
                    'extracted_date': extracted_date,
                    'firm_from_directory': 'Not found',  # Keep consistent with original format
                    'extracted_firm': firm,
                    'gpt_response': analysis.replace('\n', ' ').replace('\r', ' '),
                    'total_latency_ms': timing_info['total_latency_ms'],
                    'api_latency_ms': timing_info['api_latency_ms'],
                    'input_tokens': timing_info['input_tokens'],
                    'output_tokens': timing_info['output_tokens'],
                    'total_tokens': timing_info['total_tokens'],
                    'text_length_chars': len(text),
                    'cost_usd': timing_info['cost_usd']
                }

                results.append(result_data)

                # Small delay between tickers in same file
                time.sleep(0.1)

            return results

        else:
            logger.warning(f"File not found: {file_path}")
            return []

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return []

def process_files_parallel(multi_ticker_data, processed_files, api_key):
    """Process files in parallel, handling all tickers per file incrementally."""
    print("\n🔄 Preparing files for parallel processing...")
    # Prepare arguments for each unique file (not ticker-file combinations)
    process_args = []

    for _, row in multi_ticker_data.iterrows():
        file_path = row['full_file_path']

        # Skip if already processed
        if file_path in processed_files:
            continue

        tickers = row['ticker']
        extracted_date = row['extracted_date']
        firm = row['extracted_firm']

        # Add entire file with all its tickers as one job
        process_args.append((file_path, tickers, extracted_date, firm, api_key))

    print(f"📋 Prepared {len(process_args)} unique files for processing (deduplicated)")
    logger.info(f"Prepared {len(process_args)} unique files for processing (deduplicated)")
    total_ticker_combinations = sum(len(args[1]) for args in process_args)
    print(f"📊 Total ticker analyses to be performed: {total_ticker_combinations}")
    logger.info(f"Total ticker analyses to be performed: {total_ticker_combinations}")

    if not process_args:
        print("⚠️ No new files to process (all files already processed or skipped)")
        logger.info("No new files to process")
        return []

    all_results = []

    print(f"🚀 Starting parallel processing with 20 workers...")
    # Use ThreadPoolExecutor with 20 workers for IO-bound operations like API calls
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        future_to_args = {executor.submit(process_single_file_all_tickers, args): args for args in process_args}

        completed_files = 0
        for future in concurrent.futures.as_completed(future_to_args):
            completed_files += 1
            args = future_to_args[future]
            file_path = args[0]

            try:
                file_results = future.result()
                if file_results:
                    print(f"✅ [{completed_files}/{len(process_args)}] Completed: {os.path.basename(file_path)} ({len(file_results)} results)")
                    for result in file_results:
                        all_results.append(result)
                        # Log immediately to CSV
                        log_result_continuously(result)
                else:
                    print(f"⚠️ [{completed_files}/{len(process_args)}] No results: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"❌ [{completed_files}/{len(process_args)}] Failed: {os.path.basename(file_path)} - {e}")

    print(f"\n🏁 Parallel processing complete!")
    return all_results

def save_results(results):
    """Save results to a CSV file."""
    if results:
        df = pd.DataFrame(results)
        output_file = '/home/lichenhui/financial_analysis_4o_mini_results.csv'
        df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")

        # Also save a human-readable version
        with open('/home/lichenhui/financial_analysis_4o_mini_readable.txt', 'w') as f:
            for result in results:
                f.write(f"File: {result['file_path']}\n")
                f.write(f"Filename: {result['filename']}\n")
                f.write(f"Firm: {result['extracted_firm']}\n")
                f.write(f"Date: {result['extracted_date']}\n")
                f.write(f"Analysis: {result['gpt_response']}\n")
                f.write(f"Cost: ${result['cost_usd']:.6f}\n")
                f.write(f"Tokens: {result['total_tokens']} (in: {result['input_tokens']}, out: {result['output_tokens']})\n")
                f.write("-" * 80 + "\n\n")

        logger.info("Human-readable results saved to financial_analysis_4o_mini_readable.txt")
    else:
        logger.warning("No results to save")

def main():
    """Main function to run the entire process."""
    print("=== STARTING FINANCIAL REPORT PROCESSING ===")
    logger.info("Starting financial report processing with GPT-4o-mini...")
    print(f"Working directory: {os.getcwd()}")
    print(f"Output file will be: {CSV_OUTPUT_FILE}")

    try:
        # Initialize continuous logging
        init_continuous_logging()

        # Step 1: Initialize OpenAI client
        client = init_openai_client()

        # Step 2: Get processed file paths to avoid reprocessing
        processed_files = get_processed_file_paths()

        # Step 3: Load and join data from CSVs
        print("\n=== LOADING AND JOINING DATA ===")
        multi_ticker_data, joined_df = load_and_join_data()
        print(f"Multi-ticker data shape: {multi_ticker_data.shape}")
        print(f"First few multi-ticker files:")
        for idx, row in multi_ticker_data.head(3).iterrows():
            print(f"  File: {row['full_file_path']}")
            print(f"  Tickers: {row['ticker']}")
            print(f"  Firm: {row['extracted_firm']}")
            print()

        # Get API key for parallel processing
        try:
            credentials = load_credentials()
            api_key = credentials['openai_api_key']
        except (FileNotFoundError, KeyError, ValueError):
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in credentials.json or environment variable")

        if multi_ticker_data.empty:
            print("\n❌ ERROR: No files found with multiple tickers")
            logger.warning("No files found with multiple tickers")
            print("\nDebugging info:")
            print(f"Joined data shape: {joined_df.shape if 'joined_df' in locals() else 'N/A'}")
            return

        # Step 4: Process files in parallel (full dataset)
        # Filter out already processed files first
        unprocessed_data = multi_ticker_data[~multi_ticker_data['full_file_path'].isin(processed_files)]
        print(f"\n=== PROCESSING FILES ===")
        print(f"Processing {len(unprocessed_data)} files with multiple tickers (FULL DATASET)")
        print(f"Already processed files: {len(processed_files)}")
        logger.info(f"FULL DATASET: Processing {len(unprocessed_data)} files with multiple tickers")

        if len(unprocessed_data) > 10:
            print(f"\nFirst 10 files to be processed:")
            for idx, row in unprocessed_data.head(10).iterrows():
                file_path = row['full_file_path']
                print(f"  🔄 PROCESS: {file_path}")
                print(f"    Tickers: {row['ticker']}")
            print(f"  ... and {len(unprocessed_data) - 10} more files")
        else:
            print("\nFiles to be processed:")
            for idx, row in unprocessed_data.iterrows():
                file_path = row['full_file_path']
                print(f"  🔄 PROCESS: {file_path}")
                print(f"    Tickers: {row['ticker']}")

        results = process_files_parallel(unprocessed_data, processed_files, api_key)
        print(f"\n✅ Processing completed. Got {len(results)} results.")

        # Step 5: Calculate and log statistics
        print(f"\n=== RESULTS SUMMARY ===")
        if results:
            total_latencies = [r['total_latency_ms'] for r in results]
            api_latencies = [r['api_latency_ms'] for r in results]
            total_costs = [r['cost_usd'] for r in results]
            total_input_tokens = sum(r['input_tokens'] for r in results)
            total_output_tokens = sum(r['output_tokens'] for r in results)
            total_cost = sum(total_costs)

            logger.info("=== PROCESSING STATISTICS ===")
            logger.info(f"Total calls processed: {len(results)}")
            logger.info(f"Average total latency: {sum(total_latencies)/len(total_latencies):.1f} ms")
            logger.info(f"Min/Max total latency: {min(total_latencies):.1f} / {max(total_latencies):.1f} ms")
            logger.info(f"Average API latency: {sum(api_latencies)/len(api_latencies):.1f} ms")
            logger.info(f"Min/Max API latency: {min(api_latencies):.1f} / {max(api_latencies):.1f} ms")
            logger.info("=== TOKEN USAGE ===")
            logger.info(f"Total input tokens: {total_input_tokens:,}")
            logger.info(f"Total output tokens: {total_output_tokens:,}")
            logger.info(f"Total tokens: {total_input_tokens + total_output_tokens:,}")
            logger.info("=== COST ANALYSIS ===")
            logger.info(f"Total cost: ${total_cost:.6f}")
            logger.info(f"Average cost per analysis: ${total_cost/len(results):.6f}")
            logger.info(f"Min/Max cost per analysis: ${min(total_costs):.6f} / ${max(total_costs):.6f}")

        # Step 6: Save results
        save_results(results)

        print(f"\n🎉 Processing completed successfully!")
        print(f"📝 Results saved to: {CSV_OUTPUT_FILE}")
        print(f"📖 Readable results in: /home/lichenhui/financial_analysis_4o_mini_readable.txt")
        logger.info(f"Processing completed successfully!")
        logger.info(f"Continuous logging saved to: {CSV_OUTPUT_FILE}")

    except Exception as e:
        print(f"\n❌ ERROR in main process: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Full traceback:\n{traceback.format_exc()}")
        logger.error(f"Error in main process: {e}")
        raise

if __name__ == "__main__":
    main()
