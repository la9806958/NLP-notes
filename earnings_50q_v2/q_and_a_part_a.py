import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from openai import OpenAI
import json
import logging
from datetime import timedelta
import multiprocessing
import sys

# Configure logging for multiprocessing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(message)s',
    handlers=[
        logging.FileHandler('process_earnings_calls_modified.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

with open('credentials.json', 'r') as f:
    credentials = json.load(f)

OUTPUT_FILE = "earnings_call_analysis_results_modified.csv"
# Use multiprocessing Lock instead of threading Lock
LOCK = multiprocessing.Lock()

def analyze_qna_with_llm(current_qna, previous_qnas, ticker, current_datetime, previous_datetimes, api_key):
    previous_text = "\n\n---\n\n".join([f"Previous Call {i+1} ({previous_datetimes[i]}):\n{qna}" for i, qna in enumerate(previous_qnas)])

    prompt = f"""You are analysing Q&A sections of earnings calls. Comment only on the current call vs the previous calls.

Previous Calls:
{previous_text}

Current Call ({current_datetime}):
{current_qna}

Perform the following tasks:

Task 1:
For each speaker, compare their responses in the current call with the previous call. Summarize (1) the topics they addressed with well-qualified answers supported by numbers, and (2) the topics they addressed less convincingly with vaguer explanations. You must include both types in your summary.

Task 2:
Summarise the topics analysts are curious about and the market sentiment of analysts.

Write a paragraph for each. Keep your solution for both within 500 tokens.

Respond in the following format:

Task 1:
[Your Task 1 response here - one paragraph]

Task 2:
[Your Task 2 response here - one paragraph]"""

    try:
        # Create client with timeout for this specific call
        client_with_timeout = OpenAI(
            api_key=api_key,
            timeout=60.0  # 1 minute timeout
        )

        response = client_with_timeout.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=500  # Limiting to 500 tokens as specified
        )

        content = response.choices[0].message.content.strip()

        # Parse the response to extract Task 1 and Task 2
        task1_start = content.find("Task 1:")
        task2_start = content.find("Task 2:")

        if task1_start == -1 or task2_start == -1:
            # Fallback parsing if markers not found
            lines = content.split('\n\n')
            task1_text = lines[0] if len(lines) > 0 else content[:250]
            task2_text = lines[1] if len(lines) > 1 else content[250:]
        else:
            task1_text = content[task1_start + len("Task 1:"):task2_start].strip()
            task2_text = content[task2_start + len("Task 2:"):].strip()

        return {
            "task1_speaker_qualification": task1_text,
            "task2_analyst_sentiment": task2_text
        }
    except Exception as e:
        logging.error(f"Error analyzing ticker {ticker}: {e}")
        return {
            "task1_speaker_qualification": f"Error: {str(e)}",
            "task2_analyst_sentiment": f"Error: {str(e)}"
        }

def process_single_call(row_data, previous_calls_df, api_key):
    ticker = row_data['ticker']
    et_timestamp = row_data['et_timestamp']
    current_qna = row_data['qna_section']

    # Get previous calls for comparison (must be at least 30 days before current call)
    et_timestamp_minus_30 = et_timestamp - timedelta(days=30)
    previous_calls = previous_calls_df[
        (previous_calls_df['ticker'] == ticker) &
        (previous_calls_df['et_timestamp'] < et_timestamp_minus_30)
    ].sort_values('et_timestamp', ascending=False).head(1)

    # Skip if no previous calls for comparison
    if len(previous_calls) < 1:
        return None

    previous_calls_sorted = previous_calls.sort_values('et_timestamp', ascending=True)
    previous_qnas = previous_calls_sorted['qna_section'].tolist()
    previous_datetimes = previous_calls_sorted['et_timestamp'].astype(str).tolist()

    # Analyze with LLM
    result = analyze_qna_with_llm(current_qna, previous_qnas, ticker, str(et_timestamp), previous_datetimes, api_key)

    # Prepare output row
    output_row = {}
    for col in row_data.index:
        val = row_data[col]
        if isinstance(val, (list, dict)):
            output_row[col] = str(val)
        else:
            output_row[col] = val

    # Add analysis results
    output_row['task1_speaker_qualification'] = result['task1_speaker_qualification']
    output_row['task2_analyst_sentiment'] = result['task2_analyst_sentiment']

    return output_row

def append_to_csv(row, filename):
    with LOCK:
        df_row = pd.DataFrame([row])
        if not os.path.exists(filename):
            df_row.to_csv(filename, index=False, mode='w')
        else:
            df_row.to_csv(filename, index=False, mode='a', header=False)

def save_to_json(results, filename="earnings_analysis_modified.json"):
    """Save analysis results in JSON format for easier parsing."""
    analysis_data = []

    for result in results:
        if result and ('task1_speaker_qualification' in result or 'task2_analyst_sentiment' in result):
            analysis_data.append({
                'ticker': result.get('ticker', ''),
                'date': result.get('date', ''),
                'et_timestamp': str(result.get('et_timestamp', '')),
                'task1_speaker_qualification': result.get('task1_speaker_qualification', ''),
                'task2_analyst_sentiment': result.get('task2_analyst_sentiment', '')
            })

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, indent=2)

    logging.info(f"Saved analysis for {len(analysis_data)} records to {filename}")

def process_batch(batch_data, all_calls_subset, worker_id, api_key):
    # Reconstruct minimal DataFrame from subset data
    all_calls_df = pd.DataFrame(all_calls_subset)
    all_calls_df['et_timestamp'] = pd.to_datetime(all_calls_df['et_timestamp'])

    results = []
    for idx, (_, row) in enumerate(batch_data.iterrows()):
        try:
            logging.info(f"Worker {worker_id}: Processing {idx+1}/{len(batch_data)} - Ticker: {row['ticker']}, Date: {row['date']}")
            result = process_single_call(row, all_calls_df, api_key)
            if result is not None:
                append_to_csv(result, OUTPUT_FILE)
                results.append(result)
                logging.info(f"Worker {worker_id}: Completed {idx+1}/{len(batch_data)} - Logged to CSV")
            else:
                logging.debug(f"Worker {worker_id}: Skipped {idx+1}/{len(batch_data)} - No previous calls")
        except Exception as e:
            logging.error(f"Worker {worker_id}: Error processing row {idx+1}/{len(batch_data)} - {row['ticker']}: {e}")
    return results

def main():
    logging.info("Loading data...")
    df = pd.read_csv('extracted_earnings_calls.csv')

    # Check if close matrix exists, if not, proceed without filtering
    try:
        close_matrix = pd.read_csv('hourly_close_to_close_returns_matrix.csv')
        has_close_matrix = True
    except FileNotFoundError:
        logging.warning("hourly_close_to_close_returns_matrix.csv not found. Proceeding without ticker filtering.")
        has_close_matrix = False

    logging.info(f"Original data: {len(df)} rows")

    # Load already processed records if output file exists
    already_processed = set()
    if os.path.exists(OUTPUT_FILE):
        logging.info(f"Loading already processed records from {OUTPUT_FILE}...")
        try:
            processed_df = pd.read_csv(OUTPUT_FILE)
            # Create a unique identifier for each record using record_id and source_file
            already_processed = set(zip(processed_df['record_id'], processed_df['source_file']))
            logging.info(f"Found {len(already_processed)} already processed records")
        except Exception as e:
            logging.error(f"Error loading processed records: {e}")
            already_processed = set()

    # Process timestamps
    df['et_timestamp'] = df['et_timestamp'].str.replace(" ET", "", regex=False)
    df['et_timestamp'] = pd.to_datetime(df['et_timestamp'], errors='coerce')
    df = df.sort_values('et_timestamp')

    # Filter for records with Q&A sections
    df_filtered = df[~df["qna_section"].isna()].copy()
    logging.info(f"After filtering NaN qna_section: {len(df_filtered)} rows")

    # Filter out already processed records
    if already_processed:
        initial_count = len(df_filtered)
        df_filtered = df_filtered[~df_filtered.apply(lambda row: (row['record_id'], row['source_file']) in already_processed, axis=1)]
        logging.info(f"After filtering already processed records: {len(df_filtered)} rows (skipped {initial_count - len(df_filtered)} already processed)")

    # Filter tickers if close matrix is available
    if has_close_matrix:
        df_filtered = df_filtered[df_filtered["ticker"].isin(close_matrix.columns.tolist())]
        logging.info(f"After filtering tickers in close_matrix: {len(df_filtered)} rows")

    # Note: Not removing existing output file anymore since we're appending only new records
    if os.path.exists(OUTPUT_FILE):
        logging.info(f"Will append new results to existing file: {OUTPUT_FILE}")
    else:
        logging.info(f"Creating new output file: {OUTPUT_FILE}")

    # Print new task descriptions
    logging.info("\n=== ANALYSIS TASKS ===")
    logging.info("Task 1: Summarise, for each speaker, relative to the previous call, which topics")
    logging.info("        the speakers provided qualified (justified by numbers) answers to, and")
    logging.info("        the topics the speaker did less well in qualifying (with vaguer answer)")
    logging.info("\nTask 2: Summarise the topics analysts are curious about and the market")
    logging.info("        sentiment of analysts.")
    logging.info("\nEach task output will be a paragraph, limited to 500 tokens total.\n")

    # Set up parallel processing
    num_workers = 10
    batch_size = len(df_filtered) // num_workers + 1

    batches = []
    for i in range(num_workers):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df_filtered))
        if start_idx < len(df_filtered):
            batches.append((df_filtered.iloc[start_idx:end_idx], i))

    logging.info(f"\nStarting parallel processing with {num_workers} workers...")
    logging.info(f"Total batches: {len(batches)}")

    all_results = []

    # Pass API key to workers
    api_key = credentials['openai_api_key']

    # Create a minimal subset with only required columns to reduce memory usage
    all_calls_subset = df_filtered[['ticker', 'et_timestamp', 'qna_section']].to_dict('records')
    logging.info(f"Prepared minimal dataset subset for workers (reduced memory footprint)")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_batch, batch, all_calls_subset, worker_id, api_key): worker_id
            for batch, worker_id in batches
        }

        completed = 0
        for future in as_completed(futures):
            worker_id = futures[future]
            try:
                results = future.result()
                all_results.extend(results)
                completed += 1
                logging.info(f"\n=== Worker {worker_id} completed ({completed}/{len(batches)} batches done) ===\n")
            except Exception as e:
                logging.error(f"\n!!! Worker {worker_id} failed with error: {e} !!!\n")

    logging.info(f"\nProcessing complete! Results saved to {OUTPUT_FILE}")

    # Save to JSON for easier parsing
    save_to_json(all_results, "earnings_analysis_modified.json")

    # Print summary
    if os.path.exists(OUTPUT_FILE):
        final_df = pd.read_csv(OUTPUT_FILE)
        logging.info(f"Final output: {len(final_df)} rows")

        # Show sample output
        if len(final_df) > 0:
            logging.info("\n=== SAMPLE OUTPUT ===")
            sample = final_df.iloc[0]
            logging.info(f"Ticker: {sample.get('ticker', 'N/A')}")
            logging.info(f"Date: {sample.get('date', 'N/A')}")
            if 'task1_speaker_qualification' in sample:
                logging.info(f"\nTask 1 (Speaker Qualification):\n{sample['task1_speaker_qualification'][:300]}...")
            if 'task2_analyst_sentiment' in sample:
                logging.info(f"\nTask 2 (Analyst Sentiment):\n{sample['task2_analyst_sentiment'][:300]}...")

if __name__ == "__main__":
    main()
