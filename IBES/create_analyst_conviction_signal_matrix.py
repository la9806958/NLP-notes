#!/usr/bin/env python3

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from joblib import Parallel, delayed
import multiprocessing
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data():
    """Load the necessary data files"""
    logger.info("Loading data files...")
    
    # Load broker info file
    broker_info = pd.read_csv('data/broker_info_extraction_4o_mini_final_enhanced.csv')
    
    # Load extracted tickers file  
    tickers = pd.read_csv('data/extracted_tickers.csv')
    
    # Load close prices matrix
    close_prices = pd.read_csv('data/close_prices_matrix.csv')
    
    logger.info(f"Loaded {len(broker_info)} broker records, {len(tickers)} ticker records, {len(close_prices)} price records")
    
    return broker_info, tickers, close_prices

def prepare_data(broker_info, tickers, close_prices):
    """Prepare and merge the data"""
    logger.info("Preparing data...")
    
    # Filter out invalid dates and convert to datetime
    valid_broker_info = broker_info[
        (broker_info['extracted_date'] != 'Not found') & 
        (broker_info['extracted_date'].notna()) &
        (broker_info['extracted_firm'] != 'Not found') &
        (broker_info['extracted_firm'].notna())
    ].copy()
    
    logger.info(f"Filtered broker info from {len(broker_info)} to {len(valid_broker_info)} valid records")
    
    # Convert to datetime
    valid_broker_info['date'] = pd.to_datetime(valid_broker_info['extracted_date'], errors='coerce')
    valid_broker_info = valid_broker_info.dropna(subset=['date'])
    
    # Extract firm from extracted_firm column
    valid_broker_info['reporting_firm'] = valid_broker_info['extracted_firm']
    
    # Merge broker info with tickers on filename
    merged = pd.merge(tickers, valid_broker_info[['filename', 'date', 'reporting_firm']], on='filename', how='inner')
    
    # Filter out rows with N/A price targets
    merged = merged[merged['price_target'] != 'N/A'].copy()
    merged['price_target'] = pd.to_numeric(merged['price_target'], errors='coerce')
    merged = merged.dropna(subset=['price_target'])
    
    # Clean up date column names - use the date from broker_info (date_y)
    merged['report_date'] = merged['date_y']
    
    logger.info(f"After merging and filtering: {len(merged)} records")
    
    # Prepare close prices matrix
    close_prices['DateTime'] = pd.to_datetime(close_prices['DateTime'])
    close_prices = close_prices.set_index('DateTime')
    
    return merged, close_prices

def calculate_tpr_batch(batch_data, close_prices):
    """Calculate TPR for a batch of data - parallelized function"""
    results = []
    
    for idx, row in batch_data.iterrows():
        ticker = row['ticker']
        target_price = row['price_target']
        report_date = row['report_date']
        reporting_firm = row['reporting_firm']
        
        # Find the previous trading day close price
        prev_close = None
        for i in range(1, 6):
            check_date = report_date - timedelta(days=i)
            if ticker in close_prices.columns and check_date in close_prices.index:
                prev_close = close_prices.loc[check_date, ticker]
                if pd.notna(prev_close) and prev_close > 0:
                    break
        
        if prev_close is not None and prev_close > 0:
            tpr = target_price / prev_close
            results.append({
                'date': report_date,
                'ticker': ticker,
                'reporting_firm': reporting_firm,
                'target_price': target_price,
                'prev_close': prev_close,
                'tpr': tpr
            })
    
    return results

def calculate_tpr_parallel(merged_data, close_prices, n_jobs=-1, batch_size=1000):
    """Calculate Target Price Ratio (TPR) for each analyst-firm-date with parallelization"""
    logger.info("Calculating TPR with parallelization...")
    
    # Split data into batches
    batches = [merged_data.iloc[i:i+batch_size] for i in range(0, len(merged_data), batch_size)]
    logger.info(f"Processing {len(batches)} batches with {n_jobs} workers")
    
    # Process batches in parallel
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count() if n_jobs == -1 else n_jobs) as executor:
        futures = [executor.submit(calculate_tpr_batch, batch, close_prices) for batch in batches]
        
        all_results = []
        for future in as_completed(futures):
            batch_results = future.result()
            all_results.extend(batch_results)
    
    tpr_df = pd.DataFrame(all_results)
    logger.info(f"Calculated TPR for {len(tpr_df)} records")
    
    return tpr_df

def calculate_ranked_tpr_for_firm(firm_data, firm_name):
    """Calculate RankedTPR for a single firm - parallelized function"""
    logger.info(f"Processing firm: {firm_name} with {len(firm_data)} records")
    
    # Sort by date to ensure proper rolling window calculation
    firm_data = firm_data.sort_values('date').reset_index(drop=True)
    
    results = []
    
    for idx, row in firm_data.iterrows():
        current_date = row['date']
        current_tpr = row['tpr']
        
        # Get 365-day historical window
        start_date = current_date - timedelta(days=365)
        historical_data = firm_data[
            (firm_data['date'] >= start_date) & 
            (firm_data['date'] <= current_date)
        ]
        
        if len(historical_data) >= 2:  # Need at least 2 data points for percentile
            # Calculate percentile rank
            tpr_values = historical_data['tpr'].values
            percentile_rank = (tpr_values < current_tpr).sum() / len(tpr_values)
            
            results.append({
                'date': current_date,
                'ticker': row['ticker'],
                'reporting_firm': firm_name,
                'target_price': row['target_price'],
                'prev_close': row['prev_close'],
                'tpr': current_tpr,
                'ranked_tpr': percentile_rank,
                'historical_count': len(historical_data)
            })
    
    return results

def calculate_ranked_tpr_parallel(tpr_df, n_jobs=-1):
    """Calculate RankedTPR using 365-day percentile rank with parallelization"""
    logger.info("Calculating RankedTPR with parallelization...")
    
    firms = tpr_df['reporting_firm'].unique()
    logger.info(f"Processing {len(firms)} firms with {n_jobs} workers")
    
    # Prepare firm data
    firm_data_list = [(tpr_df[tpr_df['reporting_firm'] == firm].copy(), firm) for firm in firms]
    
    # Process firms in parallel
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count() if n_jobs == -1 else n_jobs) as executor:
        futures = [executor.submit(calculate_ranked_tpr_for_firm, firm_data, firm_name) 
                   for firm_data, firm_name in firm_data_list]
        
        all_results = []
        for future in as_completed(futures):
            firm_results = future.result()
            all_results.extend(firm_results)
    
    ranked_tpr_df = pd.DataFrame(all_results)
    logger.info(f"Calculated RankedTPR for {len(ranked_tpr_df)} records")
    
    return ranked_tpr_df

def process_analyst_date_combination(args):
    """Process a single analyst-date combination for conviction weights"""
    firm_data_subset, firm, date = args
    
    date_data = firm_data_subset[firm_data_subset['date'] == date].copy()
    
    if len(date_data) >= 5:  # Need at least 5 records to create meaningful quintiles
        # Create quintiles based on ranked_tpr for this analyst on this date
        date_data['quintile_rank'] = pd.qcut(date_data['ranked_tpr'], 
                                           q=5, labels=[1, 2, 3, 4, 5])
        
        # Map quintile ranks to conviction weights
        conviction_weight_map = {1: -2, 2: -1, 3: 0, 4: 1, 5: 2}
        date_data['conviction_weight'] = date_data['quintile_rank'].map(conviction_weight_map)
        
        results = []
        for _, row in date_data.iterrows():
            results.append({
                'date': row['date'],
                'ticker': row['ticker'],
                'reporting_firm': row['reporting_firm'],
                'ranked_tpr': row['ranked_tpr'],
                'quintile_rank': row['quintile_rank'],
                'conviction_weight': row['conviction_weight']
            })
        return results
    
    return []

def assign_conviction_weights_parallel(ranked_tpr_df, n_jobs=-1):
    """Assign conviction weights based on analyst quintile ranks with parallelization"""
    logger.info("Assigning conviction weights with parallelization...")
    
    # Prepare all analyst-date combinations
    combinations = []
    for firm in ranked_tpr_df['reporting_firm'].unique():
        firm_data = ranked_tpr_df[ranked_tpr_df['reporting_firm'] == firm].copy()
        for date in firm_data['date'].unique():
            combinations.append((firm_data, firm, date))
    
    logger.info(f"Processing {len(combinations)} analyst-date combinations")
    
    # Process combinations in parallel
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count() if n_jobs == -1 else n_jobs) as executor:
        futures = [executor.submit(process_analyst_date_combination, combo) for combo in combinations]
        
        all_results = []
        for future in as_completed(futures):
            combo_results = future.result()
            all_results.extend(combo_results)
    
    conviction_df = pd.DataFrame(all_results)
    logger.info(f"Generated conviction weights for {len(conviction_df)} records")
    
    return conviction_df

def create_signal_matrix(conviction_df, close_prices):
    """Create signal matrix with conviction weights"""
    logger.info("Creating signal matrix...")
    
    # Get all unique dates and tickers
    all_dates = sorted(conviction_df['date'].unique())
    all_tickers = sorted(close_prices.columns)  # All tickers from close prices
    
    # Create empty matrix
    signal_matrix = pd.DataFrame(index=all_dates, columns=all_tickers)
    signal_matrix = signal_matrix.fillna(0.0)
    
    # Aggregate conviction weights by date and ticker (average across analysts)
    aggregated_weights = conviction_df.groupby(['date', 'ticker'])['conviction_weight'].mean().reset_index()
    
    # Fill in the signal values
    for _, row in aggregated_weights.iterrows():
        date = row['date']
        ticker = row['ticker']
        signal_value = row['conviction_weight']
        
        if ticker in signal_matrix.columns and date in signal_matrix.index:
            signal_matrix.loc[date, ticker] = signal_value
    
    # Reset index to make date a column
    signal_matrix.reset_index(inplace=True)
    signal_matrix = signal_matrix.rename(columns={'index': 'DateTime'})
    
    logger.info(f"Created signal matrix with shape: {signal_matrix.shape}")
    
    return signal_matrix

def main():
    """Main function to execute the parallelized Task 4"""
    logger.info("Starting Parallelized Task 4: Creating Analyst Conviction Signal Matrix")
    
    # Load data
    broker_info, tickers, close_prices = load_data()
    
    # Prepare data
    merged_data, close_prices_matrix = prepare_data(broker_info, tickers, close_prices)
    
    # Calculate TPR with parallelization
    tpr_df = calculate_tpr_parallel(merged_data, close_prices_matrix)
    
    # Calculate RankedTPR with parallelization
    ranked_tpr_df = calculate_ranked_tpr_parallel(tpr_df)
    
    # Assign conviction weights with parallelization
    conviction_df = assign_conviction_weights_parallel(ranked_tpr_df)
    
    # Create signal matrix
    signal_matrix = create_signal_matrix(conviction_df, close_prices_matrix)
    
    # Save results
    logger.info("Saving results...")
    tpr_df.to_csv('data/tpr_results.csv', index=False)
    ranked_tpr_df.to_csv('data/ranked_tpr_results.csv', index=False)
    conviction_df.to_csv('data/conviction_weights.csv', index=False)
    signal_matrix.to_csv('data/analyst_conviction_signal_matrix.csv', index=False)
    
    logger.info("Parallelized Task 4 completed successfully!")
    
    # Print summary statistics
    logger.info(f"Summary:")
    logger.info(f"- TPR records: {len(tpr_df)}")
    logger.info(f"- RankedTPR records: {len(ranked_tpr_df)}")
    logger.info(f"- Conviction weight records: {len(conviction_df)}")
    logger.info(f"- Signal matrix shape: {signal_matrix.shape}")
    logger.info(f"- Date range: {signal_matrix['DateTime'].min()} to {signal_matrix['DateTime'].max()}")
    logger.info(f"- Non-zero signals: {(signal_matrix.iloc[:, 1:] != 0).sum().sum()}")
    
    # Print conviction weight distribution
    if len(conviction_df) > 0:
        weight_dist = conviction_df['conviction_weight'].value_counts().sort_index()
        logger.info(f"- Conviction weight distribution: {weight_dist.to_dict()}")

if __name__ == "__main__":
    main()