#!/usr/bin/env python3
import pandas as pd
import json
import os
from openai import OpenAI
from datetime import datetime
import logging
import concurrent.futures
import asyncio
import aiofiles
from functools import partial
import time
import tiktoken
import re
from difflib import SequenceMatcher
import pytz

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_credentials():
    """Load OpenAI API credentials"""
    with open('credentials.json', 'r') as f:
        return json.load(f)

def initialize_openai():
    """Initialize OpenAI client"""
    creds = load_credentials()
    return OpenAI(api_key=creds['openai_api_key'])

def load_data():
    """Load broker info, ticker data, and timestamp data"""
    logger.info("Loading data files...")

    # Load broker info
    broker_info = pd.read_csv('csv/broker_info_extraction_4o_mini_final_enhanced.csv')
    logger.info(f"Loaded {len(broker_info)} broker info records")

    # Load ticker data with timestamps
    tickers = pd.read_csv('csv/timestamps_tickers_joined.csv')
    logger.info(f"Loaded {len(tickers)} ticker records with timestamps")

    return broker_info, tickers

def merge_and_deduplicate_data(broker_info, tickers):
    """Merge datasets and deduplicate by filename"""
    logger.info("Merging and deduplicating data...")

    # Merge on filename (broker_info uses 'filename', tickers uses 'filename_timestamps')
    merged = pd.merge(
        broker_info,
        tickers,
        left_on='filename',
        right_on='filename_timestamps',
        how='inner'
    )

    # Deduplicate by unique filenames
    merged_dedup = merged.drop_duplicates(subset=['filename'])
    logger.info(f"After deduplication: {len(merged_dedup)} unique reports")

    return merged_dedup

def filter_single_ticker_reports(merged_data):
    """Filter to reports discussing only one stock and from specific firms"""
    logger.info("Filtering to single-ticker reports from specific firms...")

    # Define specific firms to include (excluding "Other")
    target_firms = [
        'J.P. Morgan', 'Jefferies', 'Barclays', 'Morgan Stanley', 'Bank of America',
        'Citigroup', 'Evercore', 'Goldman Sachs', 'Deutsche Bank', 'Wells Fargo',
        'Guggenheim', 'BNP Paribas', 'UBS', 'Bernstein', 'HSBC', 'Macquarie',
        'Nomura'
    ]

    # Filter by specific firms first
    firm_filtered_data = merged_data[merged_data['extracted_firm'].isin(target_firms)]
    logger.info(f"After firm filtering: {len(firm_filtered_data)} reports")

    # Count tickers per report
    ticker_counts = firm_filtered_data.groupby('filename')['ticker'].nunique().reset_index()
    ticker_counts.columns = ['filename', 'ticker_count']

    # Filter for single ticker reports
    single_ticker_files = ticker_counts[ticker_counts['ticker_count'] == 1]['filename']
    filtered_data = firm_filtered_data[firm_filtered_data['filename'].isin(single_ticker_files)]

    logger.info(f"Single ticker reports from specific firms: {len(filtered_data)}")
    return filtered_data

def find_previous_reports(data):
    """Find previous reports for each (broker, ticker) pair - optimized version"""
    logger.info("Finding previous reports...")

    # Filter out invalid dates and convert
    data_clean = data[data['extracted_date'] != 'Not found'].copy()
    data_clean['extracted_date'] = pd.to_datetime(data_clean['extracted_date'], errors='coerce')

    # Remove rows with invalid dates
    data_clean = data_clean.dropna(subset=['extracted_date'])
    logger.info(f"After cleaning dates: {len(data_clean)} reports")

    # Sort by broker, ticker, and date (ascending for shift operation)
    data_sorted = data_clean.sort_values(['extracted_firm', 'ticker', 'extracted_date']).reset_index(drop=True)

    # Use vectorized operations to find previous reports
    # Shift previous values within each group
    data_sorted['prev_filename'] = data_sorted.groupby(['extracted_firm', 'ticker'])['filename'].shift(1)
    data_sorted['prev_full_directory'] = data_sorted.groupby(['extracted_firm', 'ticker'])['full_directory'].shift(1)
    data_sorted['prev_extracted_date'] = data_sorted.groupby(['extracted_firm', 'ticker'])['extracted_date'].shift(1)
    data_sorted['prev_file_path'] = data_sorted.groupby(['extracted_firm', 'ticker'])['file_path'].shift(1)
    data_sorted['prev_report_timestamp_et'] = data_sorted.groupby(['extracted_firm', 'ticker'])['report_timestamp_et'].shift(1)

    # Filter out rows without previous reports (first report in each group)
    has_previous = data_sorted['prev_filename'].notna()
    pairs_data = data_sorted[has_previous].copy()

    # Create results dataframe directly
    results_df = pd.DataFrame({
        'current_file': pairs_data['filename'],
        'current_directory': pairs_data['full_directory'],
        'current_date': pairs_data['extracted_date'],
        'current_firm': pairs_data['extracted_firm'],
        'ticker': pairs_data['ticker'],
        'current_et_timestamp': pairs_data.get('report_timestamp_et', 'N/A'),
        'previous_file': pairs_data['prev_filename'],
        'previous_directory': pairs_data['prev_full_directory'],
        'previous_date': pairs_data['prev_extracted_date'],
        'previous_et_timestamp': pairs_data.get('prev_report_timestamp_et', 'N/A'),
        'current_full_path': pairs_data['file_path'],
        'previous_full_path': pairs_data['prev_file_path']
    })

    logger.info(f"Found {len(results_df)} report pairs with previous versions")
    return results_df

def read_report_content(file_path):
    """Read report content from file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return ""

async def read_report_content_async(file_path):
    """Async version of read_report_content"""
    try:
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            return await f.read()
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return ""

def process_single_pair(args):
    """Process a single report pair - designed for multiprocessing"""
    idx, row, openai_api_key = args

    # Initialize OpenAI client for this process
    client = OpenAI(api_key=openai_api_key)

    logger.info(f"Processing pair {idx+1}: {row['ticker']} - {row['current_firm']}")

    # Read current and previous reports
    current_content = read_report_content(row['current_full_path'])
    previous_content = read_report_content(row['previous_full_path'])

    if current_content and previous_content:
        # Generate summary
        summary = summarize_differences(
            client,
            current_content,
            previous_content,
            row['ticker'],
            row['current_firm']
        )

        result = {
            'ticker': row['ticker'],
            'firm': row['current_firm'],
            'current_file': row['current_file'],
            'current_directory': row['current_directory'],
            'current_date': row['current_date'],
            'current_et_timestamp': normalize_et_timestamp(row.get('current_et_timestamp', 'N/A')),
            'previous_file': row['previous_file'],
            'previous_directory': row['previous_directory'],
            'previous_date': row['previous_date'],
            'previous_et_timestamp': normalize_et_timestamp(row.get('previous_et_timestamp', 'N/A')),
            'llm_summary': summary,
            'processing_timestamp': datetime.now().isoformat()
        }

        # Immediately append to streaming CSV
        append_result_to_csv(result)

        return result
    else:
        logger.warning(f"Could not read content for pair {idx+1}")
        return None

def clean_text(text):
    """Remove non-English characters and normalize text"""
    if not text:
        return ""

    # Remove non-ASCII characters (keeps only English characters)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text

def split_into_chunks(text, chunk_size=100):
    """Split text into chunks of approximately chunk_size words"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def remove_common_chunks(current_text, previous_text, similarity_threshold=0.8):
    """Remove chunks that are similar between current and previous reports"""
    current_clean = clean_text(current_text)
    previous_clean = clean_text(previous_text)

    current_chunks = split_into_chunks(current_clean)
    previous_chunks = split_into_chunks(previous_clean)

    # Find unique chunks in current report
    unique_current = []
    for curr_chunk in current_chunks:
        is_unique = True
        for prev_chunk in previous_chunks:
            similarity = SequenceMatcher(None, curr_chunk.lower(), prev_chunk.lower()).ratio()
            if similarity >= similarity_threshold:
                is_unique = False
                break
        if is_unique:
            unique_current.append(curr_chunk)

    # Find unique chunks in previous report
    unique_previous = []
    for prev_chunk in previous_chunks:
        is_unique = True
        for curr_chunk in current_chunks:
            similarity = SequenceMatcher(None, prev_chunk.lower(), curr_chunk.lower()).ratio()
            if similarity >= similarity_threshold:
                is_unique = False
                break
        if is_unique:
            unique_previous.append(prev_chunk)

    processed_current = ' '.join(unique_current)
    processed_previous = ' '.join(unique_previous)

    return processed_current, processed_previous

def normalize_et_timestamp(timestamp_str):
    """Normalize ET timestamp to always show 'ET' instead of EST/EDT"""
    if pd.isna(timestamp_str) or timestamp_str == 'N/A':
        return timestamp_str

    # Replace EST or EDT with ET for consistency
    normalized = str(timestamp_str).replace(' EST', ' ET').replace(' EDT', ' ET')
    return normalized

def truncate_to_tokens(text, max_tokens=10000, model="gpt-4o-mini"):
    """Truncate text to a maximum number of tokens"""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base if model not found
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text

    # Truncate to max_tokens and decode back to text
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)

def summarize_differences(client, current_content, previous_content, ticker, firm):
    """Use GPT-4o-mini to summarize differences between reports"""

    # Preprocess the content: remove common chunks and non-English characters
    processed_current, processed_previous = remove_common_chunks(current_content, previous_content)

    # Log preprocessing statistics
    logger.info(f"Preprocessing {ticker}: Original lengths - Current: {len(current_content)}, Previous: {len(previous_content)}")
    logger.info(f"After preprocessing - Current: {len(processed_current)}, Previous: {len(processed_previous)}")

    prompt = f"""Task: Summarize the key differences between the most recent equity research report and the previously published report. Write one structured paragraph for each of the following dimensions:

  1. New Information: Identify specific events (earnings results, regulatory changes, management guidance, acquisitions, market
  developments) introduced in the current report. Quote specific data points, dates, and analyst interpretations. Indicate sentiment
  using explicit language from the reports.

  2. Interesting Perspectives: Highlight any novel analytical frames, thematic shifts, or unique observations in the current report that differentiate it from the prior version. Explain why these perspectives are noteworthy.

3. Financial Projections: Compare specific metrics with exact figures: (examples below but not limited to)
     - Target price: Previous $X → Current $Y (change of Z%)
     - EPS estimates: Previous $A → Current $B for [time period]
     - Revenue projections: Previous vs. Current with % changes
     - Rating changes: [Previous rating] → [Current rating]
  Include the analysts' exact justifications quoted from the reports.

4. Headwinds/Tailwinds: Outline the major challenges (headwinds) and supportive factors (tailwinds) emphasized in the current report. Contrast with the earlier report to show how the balance of risks and opportunities has shifted. In paticular, include:
     - NEW headwinds in current report (not mentioned previously)
     - ONGOING headwinds (status change from previous report)
     - NEW tailwinds in current report
     - ONGOING tailwinds (status change from previous report)
     - Market/competitive risks: [new concerns]
     - Operational risks: [company-specific issues]
     - Balance sheet risks: [debt, liquidity, capital concerns]
  Compare risk emphasis (high/medium/low) between reports.

5. Valuation Methodology: Compare how analysts justify their valuations:
     - Multiples used: P/E, EV/EBITDA, P/B ratios and changes
     - DCF assumptions: discount rate, growth rate changes
     - Peer comparison shifts
     - Asset-based valuations for specific sectors

Keep your overall analysis under 2000 tokens.

Company: {ticker}
Research Firm: {firm}

CURRENT REPORT (unique content after preprocessing):
{truncate_to_tokens(processed_current, max_tokens=10000)}

PREVIOUS REPORT (unique content after preprocessing):
{truncate_to_tokens(processed_previous, max_tokens=10000)}"""

    # Log the prompt to a file
    with open('broker_comparison_prompts.log', 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*100}\n")
        f.write(f"TIMESTAMP: {datetime.now().isoformat()}\n")
        f.write(f"TICKER: {ticker}\n")
        f.write(f"FIRM: {firm}\n")
        f.write(f"PROMPT:\n{prompt}\n")
        f.write(f"{'='*100}\n\n")

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.3
        )

        # Log the response as well
        with open('broker_comparison_prompts.log', 'a', encoding='utf-8') as f:
            f.write(f"RESPONSE:\n{response.choices[0].message.content}\n")
            f.write(f"{'='*100}\n\n")

        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}")
        return f"Error generating summary: {e}"

def process_report_pairs(report_pairs, client):
    """Process all report pairs and generate summaries"""
    logger.info("Processing report pairs...")

    results = []
    for idx, row in report_pairs.iterrows():
        logger.info(f"Processing pair {idx+1}/{len(report_pairs)}: {row['ticker']} - {row['current_firm']}")

        # Read current and previous reports
        current_content = read_report_content(row['current_full_path'])
        previous_content = read_report_content(row['previous_full_path'])

        if current_content and previous_content:
            # Generate summary
            summary = summarize_differences(
                client,
                current_content,
                previous_content,
                row['ticker'],
                row['current_firm']
            )

            results.append({
                'ticker': row['ticker'],
                'firm': row['current_firm'],
                'current_file': row['current_file'],
                'current_directory': row['current_directory'],
                'current_date': row['current_date'],
                'current_et_timestamp': normalize_et_timestamp(row.get('current_et_timestamp', 'N/A')),
                'previous_file': row['previous_file'],
                'previous_directory': row['previous_directory'],
                'previous_date': row['previous_date'],
                'previous_et_timestamp': normalize_et_timestamp(row.get('previous_et_timestamp', 'N/A')),
                'llm_summary': summary,
                'processing_timestamp': datetime.now().isoformat()
            })
        else:
            logger.warning(f"Could not read content for pair {idx+1}")

    return pd.DataFrame(results)

def save_results(results_df, filename='broker_report_comparisons.csv'):
    """Save results to CSV"""
    results_df.to_csv(filename, index=False)
    logger.info(f"Results saved to {filename}")

def append_result_to_csv(result_dict, filename='broker_report_comparisons_streaming.csv'):
    """Append a single result to CSV file for continuous logging"""
    import os
    import pandas as pd

    # Create DataFrame from single result
    result_df = pd.DataFrame([result_dict])

    # If file doesn't exist, create with headers
    if not os.path.exists(filename):
        result_df.to_csv(filename, index=False)
    else:
        # Append without headers
        result_df.to_csv(filename, mode='a', header=False, index=False)

    logger.info(f"Appended result for {result_dict['ticker']} - {result_dict['firm']} to {filename}")

def load_processed_pairs(filename='broker_report_comparisons_streaming.csv'):
    """Load already processed pairs to enable resume functionality"""
    import os
    import pandas as pd

    if not os.path.exists(filename):
        logger.info("No existing processed pairs file found. Starting fresh.")
        return set()

    try:
        processed_df = pd.read_csv(filename)
        # Create unique keys based on ticker, firm, current_file, and previous_file
        processed_df['pair_key'] = (processed_df['ticker'].astype(str) + '|' +
                                   processed_df['firm'].astype(str) + '|' +
                                   processed_df['current_file'].astype(str) + '|' +
                                   processed_df['previous_file'].astype(str))

        processed_keys = set(processed_df['pair_key'].unique())
        logger.info(f"Loaded {len(processed_keys)} already processed pairs for resume functionality")
        return processed_keys
    except Exception as e:
        logger.error(f"Error loading processed pairs: {e}")
        return set()

def filter_unprocessed_pairs(report_pairs, processed_keys):
    """Filter out already processed pairs to enable resume functionality"""
    if not processed_keys:
        logger.info("No processed pairs to filter. Processing all pairs.")
        return report_pairs

    # Create pair keys for all report pairs (using current_firm instead of firm)
    report_pairs['pair_key'] = (report_pairs['ticker'].astype(str) + '|' +
                               report_pairs['current_firm'].astype(str) + '|' +
                               report_pairs['current_file'].astype(str) + '|' +
                               report_pairs['previous_file'].astype(str))

    # Filter out already processed pairs
    unprocessed_pairs = report_pairs[~report_pairs['pair_key'].isin(processed_keys)]

    # Remove the temporary pair_key column
    unprocessed_pairs = unprocessed_pairs.drop('pair_key', axis=1)

    logger.info(f"Filtered pairs: {len(report_pairs)} total → {len(unprocessed_pairs)} unprocessed")
    return unprocessed_pairs

def main():
    """Main execution function"""
    logger.info("Starting broker report comparison framework with resume functionality")

    # Load and process data (full dataset)
    broker_info, tickers = load_data()

    # Process all data
    merged_data = merge_and_deduplicate_data(broker_info, tickers)
    single_ticker_data = filter_single_ticker_reports(merged_data)

    # Find report pairs
    report_pairs = find_previous_reports(single_ticker_data)

    if report_pairs.empty:
        logger.warning("No report pairs found with previous versions")
        return

    # Load already processed pairs for resume functionality
    processed_keys = load_processed_pairs()

    # Filter out already processed pairs
    unprocessed_pairs = filter_unprocessed_pairs(report_pairs, processed_keys)

    if unprocessed_pairs.empty:
        logger.info("All pairs have already been processed! No work remaining.")
        print("=== ALL PROCESSING COMPLETE ===")
        print(f"Total pairs found: {len(report_pairs)}")
        print(f"Already processed: {len(processed_keys)}")
        print("No additional processing needed.")
        return

    # Log and print the number of pairs to be processed
    total_pairs = len(report_pairs)
    remaining_pairs = len(unprocessed_pairs)
    completed_pairs = len(processed_keys)

    logger.info(f"Found {total_pairs} total report pairs")
    logger.info(f"Already processed: {completed_pairs} pairs ({completed_pairs/total_pairs*100:.1f}%)")
    logger.info(f"Remaining to process: {remaining_pairs} pairs ({remaining_pairs/total_pairs*100:.1f}%)")

    print(f"=== RESUME PROCESSING PLAN ===")
    print(f"Total report pairs found: {total_pairs:,}")
    print(f"Already processed: {completed_pairs:,} ({completed_pairs/total_pairs*100:.1f}%)")
    print(f"Remaining to process: {remaining_pairs:,} ({remaining_pairs/total_pairs*100:.1f}%)")
    print(f"Starting parallel processing with 25 workers...")

    # Use the filtered unprocessed pairs for processing
    report_pairs = unprocessed_pairs

    # Process remaining report pairs in parallel
    logger.info(f"Processing {remaining_pairs} remaining report pairs in parallel...")

    # Load credentials for parallel processing
    creds = load_credentials()
    openai_api_key = creds['openai_api_key']

    # Prepare arguments for parallel processing
    process_args = [(idx, row, openai_api_key) for idx, row in report_pairs.iterrows()]

    # Use ProcessPoolExecutor for parallel processing
    start_time = time.time()
    results = []

    # Use 25 workers for maximum parallelization
    max_workers = 25
    logger.info(f"Using {max_workers} parallel workers")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_args = {executor.submit(process_single_pair, args): args for args in process_args}

        # Collect results as they complete with progress tracking
        completed = 0
        total_tasks = len(future_to_args)

        for future in concurrent.futures.as_completed(future_to_args):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                args = future_to_args[future]
                logger.error(f"Error processing pair {args[0]}: {e}")

            completed += 1
            if completed % max(1, total_tasks // 10) == 0 or completed == total_tasks:
                progress = (completed / total_tasks) * 100
                logger.info(f"Progress: {completed}/{total_tasks} pairs processed ({progress:.1f}%)")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    save_results(results_df)

    end_time = time.time()
    processing_time = end_time - start_time

    # Display summary stats
    print(f"\n=== PROCESSING COMPLETE ===")
    print(f"Total report pairs processed: {len(results_df)}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Average time per pair: {processing_time/len(results_df):.2f} seconds")
    print(f"Results saved to: broker_report_comparisons.csv")

    logger.info(f"Broker report comparison processing completed in {processing_time:.2f} seconds")

if __name__ == "__main__":
    main()
