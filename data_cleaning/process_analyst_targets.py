#!/usr/bin/env python3
"""
Process analyst targets and earnings calls data in parallel.
For each earnings call, compute:
1. Mean consensus price target before the call (monthly forward fill)
2. Mean consensus price target in 5-day window after the call
3. Percentage difference between these two values
4. Return from T+1 to T+20 (where T is nearest hourly boundary after call)
5. Return from T+1 to T+40
Outputs earnings calls CSV with new analyst target change columns and return columns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count, Manager
import os
import warnings

# Suppress FutureWarnings about timezone parsing
warnings.filterwarnings('ignore', category=FutureWarning)

# Global variables that will be shared across processes
analysts_global = None
returns_global = None
examples_counter = None
max_examples = 5

def init_worker(analysts_df, returns_df, counter):
    """Initialize global variables for each worker process"""
    global analysts_global, returns_global, examples_counter
    analysts_global = analysts_df
    returns_global = returns_df
    examples_counter = counter

def process_single_row(args):
    """Process a single earnings call row"""
    idx, row, ticker = args

    result = {
        'idx': idx,
        'analyst_target_pct_change': np.nan,
        'mean_consensus_before': np.nan,
        'mean_consensus_after': np.nan,
        'n_analysts_before': 0,
        'n_analysts_after': 0,
        'return_T1_to_T20': np.nan,
        'return_T1_to_T40': np.nan,
        'T_timestamp': pd.NaT,
        'example_log': None
    }

    call_time = pd.to_datetime(row['et_timestamp'], errors='coerce')

    if pd.isna(call_time):
        return result

    # Process analyst targets
    ticker_analysts = analysts_global[analysts_global['TICKER'] == ticker].copy()

    if len(ticker_analysts) > 0:
        # Get consensus before call (monthly fill forward - 60 days)
        before_call = ticker_analysts[ticker_analysts['activation_time'] < call_time].copy()

        if len(before_call) > 0:
            # Keep only most recent target per analyst within 60 days
            before_call['days_before'] = (call_time - before_call['activation_time']).dt.days
            before_call_monthly = before_call[before_call['days_before'] <= 60]

            if len(before_call_monthly) > 0:
                # Get most recent target per analyst
                before_consensus = before_call_monthly.groupby('ESTIMID').tail(1)

                if len(before_consensus) > 0:
                    mean_before = before_consensus['VALUE'].mean()

                    # Get consensus in 7-day window after call
                    after_call_start = call_time
                    after_call_end = call_time + timedelta(days=7)

                    after_call = ticker_analysts[
                        (ticker_analysts['activation_time'] >= after_call_start) &
                        (ticker_analysts['activation_time'] <= after_call_end)
                    ]

                    if len(after_call) > 0:
                        # Get most recent target per analyst in the 7-day window
                        after_consensus = after_call.groupby('ESTIMID').tail(1)
                        mean_after = after_consensus['VALUE'].mean()

                        # Compute percentage difference
                        if mean_before > 0:
                            pct_diff = ((mean_after - mean_before) / mean_before) * 100
                        else:
                            pct_diff = np.nan

                        # Store results
                        result['analyst_target_pct_change'] = pct_diff
                        result['mean_consensus_before'] = mean_before
                        result['mean_consensus_after'] = mean_after
                        result['n_analysts_before'] = len(before_consensus)
                        result['n_analysts_after'] = len(after_consensus)

    # Process returns
    if ticker in returns_global.columns:
        # Find nearest hourly boundary T that follows the call time
        T = call_time.ceil('h')
        T_original = T

        # Find T in the returns matrix index
        if T not in returns_global.index:
            future_times = returns_global.index[returns_global.index >= T]
            if len(future_times) > 0:
                T = future_times[0]

        if T in returns_global.index:
            T_pos = returns_global.index.get_loc(T)

            # Print example for first few matches (thread-safe check)
            current_count = examples_counter.value
            if current_count < max_examples:
                log_lines = []
                log_lines.append(f"\n  Example {current_count + 1}:")
                log_lines.append(f"    Ticker: {ticker}")
                log_lines.append(f"    Call time: {call_time}")
                log_lines.append(f"    T (hourly ceil): {T_original}")
                if T != T_original:
                    log_lines.append(f"    T (adjusted to next available): {T}")
                log_lines.append(f"    T position in matrix: {T_pos}")
                log_lines.append(f"    T+1 timestamp: {returns_global.index[T_pos+1] if T_pos+1 < len(returns_global) else 'N/A'}")
                log_lines.append(f"    T+20 timestamp: {returns_global.index[T_pos+20] if T_pos+20 < len(returns_global) else 'N/A'}")
                log_lines.append(f"    T+40 timestamp: {returns_global.index[T_pos+40] if T_pos+40 < len(returns_global) else 'N/A'}")
                result['example_log'] = '\n'.join(log_lines)
                examples_counter.value = current_count + 1

            # Calculate T+1 to T+20
            if T_pos + 20 < len(returns_global):
                returns_slice_20 = returns_global.iloc[T_pos+1:T_pos+21][ticker]
                cum_return_20 = (1 + returns_slice_20).prod() - 1
                result['return_T1_to_T20'] = cum_return_20

            # Calculate T+1 to T+40
            if T_pos + 40 < len(returns_global):
                returns_slice_40 = returns_global.iloc[T_pos+1:T_pos+41][ticker]
                cum_return_40 = (1 + returns_slice_40).prod() - 1
                result['return_T1_to_T40'] = cum_return_40

            result['T_timestamp'] = T

    return result

def process_chunk(chunk_data):
    """Process a chunk of earnings calls"""
    chunk_num, earnings_chunk = chunk_data

    print(f"\nChunk {chunk_num + 1}:")
    print(f"  Processing {len(earnings_chunk)} rows...")

    # Initialize new columns
    earnings_chunk['analyst_target_pct_change'] = np.nan
    earnings_chunk['mean_consensus_before'] = np.nan
    earnings_chunk['mean_consensus_after'] = np.nan
    earnings_chunk['n_analysts_before'] = 0
    earnings_chunk['n_analysts_after'] = 0
    earnings_chunk['return_T1_to_T20'] = np.nan
    earnings_chunk['return_T1_to_T40'] = np.nan
    earnings_chunk['T_timestamp'] = pd.NaT

    # Prepare arguments for parallel processing
    args_list = []
    for idx, row in earnings_chunk.iterrows():
        ticker = row['ticker']
        args_list.append((idx, row, ticker))

    # Process rows in parallel within this chunk
    num_workers = max(1, cpu_count() // 2)  # Use half of CPU cores per chunk worker
    with Pool(processes=num_workers, initializer=init_worker,
              initargs=(analysts_global, returns_global, examples_counter)) as pool:
        results = pool.map(process_single_row, args_list)

    # Apply results back to the chunk
    for result in results:
        if result['example_log']:
            print(result['example_log'])

        idx = result['idx']
        earnings_chunk.loc[idx, 'analyst_target_pct_change'] = result['analyst_target_pct_change']
        earnings_chunk.loc[idx, 'mean_consensus_before'] = result['mean_consensus_before']
        earnings_chunk.loc[idx, 'mean_consensus_after'] = result['mean_consensus_after']
        earnings_chunk.loc[idx, 'n_analysts_before'] = result['n_analysts_before']
        earnings_chunk.loc[idx, 'n_analysts_after'] = result['n_analysts_after']
        earnings_chunk.loc[idx, 'return_T1_to_T20'] = result['return_T1_to_T20']
        earnings_chunk.loc[idx, 'return_T1_to_T40'] = result['return_T1_to_T40']
        earnings_chunk.loc[idx, 'T_timestamp'] = result['T_timestamp']

    chunk_matches = earnings_chunk['analyst_target_pct_change'].notna().sum()
    return_matches = earnings_chunk['return_T1_to_T20'].notna().sum()

    print(f"  Analyst targets: {chunk_matches} matches")
    print(f"  Returns: {return_matches} matches")

    return chunk_num, earnings_chunk, chunk_matches, return_matches


if __name__ == '__main__':
    print("Loading analyst targets...")
    analysts = pd.read_csv('analystsTargets.csv')
    print(f"Loaded {len(analysts)} analyst target records")

    # Parse activation timestamp (ACTDATS + ACTTIMS)
    analysts['activation_time'] = pd.to_datetime(
        analysts['ACTDATS'] + ' ' + analysts['ACTTIMS'],
        format='%Y-%m-%d %H:%M:%S'
    )

    # Sort by ticker and time for efficient processing
    analysts = analysts.sort_values(['TICKER', 'activation_time'])

    print("\nLoading hourly close-to-close returns matrix...")
    returns = pd.read_csv('hourly_close_to_close_returns_matrix.csv', parse_dates=['datetime'], index_col='datetime')
    print(f"Loaded returns matrix: {returns.shape[0]:,} timestamps x {returns.shape[1]:,} tickers")
    print(f"Date range: {returns.index[0]} to {returns.index[-1]}")

    # Set global variables
    analysts_global = analysts
    returns_global = returns

    # Create shared counter for examples
    manager = Manager()
    examples_counter = manager.Value('i', 0)

    print("\nProcessing earnings calls in chunks...")
    chunk_size = 50000
    output_file = 'earnings_calls_with_analyst_changes.csv'
    total_matches = 0
    total_rows = 0
    first_chunk = True

    # Read and process chunks sequentially (but each chunk processes rows in parallel)
    for chunk_num, earnings_chunk in enumerate(pd.read_csv('extracted_earnings_calls.csv', chunksize=chunk_size)):
        chunk_num_result, processed_chunk, chunk_matches, return_matches = process_chunk((chunk_num, earnings_chunk))

        # Select only essential columns (exclude raw transcript)
        output_columns = [
            'ticker',
            'et_timestamp',
            'analyst_target_pct_change',
            'mean_consensus_before',
            'mean_consensus_after',
            'n_analysts_before',
            'n_analysts_after',
            'return_T1_to_T20',
            'return_T1_to_T40',
            'T_timestamp'
        ]
        processed_chunk_output = processed_chunk[output_columns]

        # Write to output file
        if first_chunk:
            processed_chunk_output.to_csv(output_file, index=False, mode='w')
            first_chunk = False
        else:
            processed_chunk_output.to_csv(output_file, index=False, mode='a', header=False)

        total_matches += chunk_matches
        total_rows += len(processed_chunk)

        print(f"  Cumulative analyst data: {total_matches}/{total_rows} ({100*total_matches/total_rows:.1f}%)")

    print(f"\n\n{'='*60}")
    print(f"Complete! Output saved to {output_file}")
    print(f"Total rows processed: {total_rows:,}")
    print(f"Rows with analyst target data: {total_matches:,} ({100*total_matches/total_rows:.1f}%)")
    print(f"{'='*60}")
