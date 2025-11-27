#!/usr/bin/env python3
"""
Data loading utilities for Alpha Agent Factor Pipeline

This module handles:
- Loading API credentials
- Parallel processing of CSV files for volume calculation
- Loading real market data from CSV files
- Managing top liquid tickers by volume
"""

import os
import json
import glob
import logging
import multiprocessing as mp
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np

from config import TWO_YEARS_AGO

logger = logging.getLogger(__name__)


def keep_us_regular_session(df: pd.DataFrame,
                            start: str = "09:30",
                            end: str = "16:00",
                            tz: str = "America/New_York") -> pd.DataFrame:
    """
    Filter intraday data to the regular U.S. equity session (09:30–16:00 ET).

    Args:
        df: DataFrame with a DatetimeIndex (naive or tz-aware).
        start: session start time (HH:MM).
        end: session end time (HH:MM).
        tz: timezone of the market data (default = U.S. Eastern).

    Returns:
        DataFrame containing only timestamps within the regular session.
    """
    # Ensure index is timezone-aware
    if df.index.tz is None:
        df = df.tz_localize(tz)
    else:
        df = df.tz_convert(tz)

    # Keep rows within the session window
    df = df.between_time(start_time=start, end_time=end)
    return df


def load_credentials(credentials_path: str = "credentials.json") -> Dict[str, str]:
    """Load API credentials from JSON file.

    Args:
        credentials_path: Path to credentials JSON file

    Returns:
        Dictionary of credentials
    """
    try:
        with open(credentials_path, 'r') as f:
            credentials = json.load(f)
        return credentials
    except FileNotFoundError:
        logger.warning(f"{credentials_path} not found. Using environment variables.")
        return {}
    except json.JSONDecodeError:
        logger.warning(f"Invalid JSON in {credentials_path}. Using environment variables.")
        return {}


def _process_csv_for_volume(csv_file: str) -> Tuple[Optional[str], Optional[float]]:
    """Helper function to process a single CSV file for volume calculation.

    Args:
        csv_file: Path to CSV file

    Returns:
        Tuple of (ticker, avg_volume) or (None, None) if processing fails
    """
    try:
        # Extract ticker from filename
        ticker = os.path.splitext(os.path.basename(csv_file))[0]

        # Load and calculate average volume
        df = pd.read_csv(csv_file)
        # Standardize column names to lowercase to handle both 'Volume' and 'volume'
        df.columns = df.columns.str.lower()

        if 'volume' not in df.columns:
            return ticker, None  # Will log warning in main process

        # Handle timestamp/datetime column and filter to last two years
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'])
        elif 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        else:
            # Skip files without datetime information
            return ticker, None

        # Filter to last two years
        df = df[df['datetime'] >= TWO_YEARS_AGO]

        if len(df) == 0:
            return ticker, None  # No data from last two years

        avg_volume = df['volume'].mean()
        return ticker, avg_volume
    except Exception:
        return None, None


def _process_csv_for_data(csv_file_info: Tuple[str, int]) -> Tuple[Optional[str], Optional[Tuple[str, float]]]:
    """Helper function to process a single CSV file for data loading.

    Args:
        csv_file_info: Tuple of (csv_file_path, max_samples_to_scan)

    Returns:
        Tuple of (ticker, (csv_file, avg_volume)) or (ticker, None) if validation fails, or (None, None) if processing fails
    """
    csv_file, max_samples = csv_file_info
    try:
        ticker = os.path.splitext(os.path.basename(csv_file))[0]
        df = pd.read_csv(csv_file)

        # Standardize column names to lowercase
        df.columns = df.columns.str.lower()
        required_cols = ['open', 'high', 'low', 'close', 'volume']

        # Check for missing columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            # Return ticker with None to indicate validation failure (will be logged)
            return ticker, None

        # Handle timestamp/datetime column
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], errors='coerce')
        elif 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        else:
            # Skip files without datetime information
            return ticker, None

        # Remove rows with invalid datetime
        df = df.dropna(subset=['datetime'])

        # Filter to last two years BEFORE setting index
        df = df[df['datetime'] >= TWO_YEARS_AGO]

        if len(df) == 0:
            return ticker, None  # No data from last two years

        df.set_index('datetime', inplace=True)
        df = df[required_cols].sort_index()

        # Filter out invalid data
        df = df[(df > 0).all(axis=1)]  # Remove rows with non-positive values
        df = df.dropna()

        if len(df) < 1000:  # Need sufficient data
            return ticker, None

        # Return filepath and avg_volume only (not the dataframe!)
        # Loading dataframes in parallel causes BrokenPipeError
        avg_volume = df['volume'].mean()
        return ticker, (csv_file, avg_volume)

    except Exception as e:
        # Log exception details for debugging
        # Note: logging from worker processes may not work well with multiprocessing
        return None, None


def load_real_market_data(data_path: str = "/home/lichenhui/data/1min",
                          n_tickers: int = 5,
                          n_cores: int = 20,
                          ticker_cache_path: str = "ticker_mapping.json") -> Dict:
    """Load real market data from CSV files in the data directory.

    Args:
        data_path: Path to directory containing ticker CSV files
        n_tickers: Number of top liquid tickers to load
        n_cores: Number of CPU cores to use for parallel processing
        ticker_cache_path: Path to save/load ticker mapping cache

    Returns:
        Dictionary containing:
            - 'tickers': List of ticker symbols
            - 'ticker_files': Dict mapping tickers to file paths
            - 'ticker_data': Dict mapping tickers to DataFrames
    """
    csv_files = glob.glob(os.path.join(data_path, "*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in {data_path}. Cannot load market data.")

    # Check if ticker mapping cache exists and is still valid
    ticker_volumes = {}
    ticker_files = {}
    use_cache = False

    if os.path.exists(ticker_cache_path):
        try:
            with open(ticker_cache_path, 'r') as f:
                cache_data = json.load(f)

            # Validate cache - check if files still exist
            cached_files = set(cache_data['ticker_files'].values())
            current_files = set(csv_files)

            if cached_files.issubset(current_files):
                ticker_volumes = cache_data['ticker_volumes']
                ticker_files = cache_data['ticker_files']
                use_cache = True
                logger.info(f"Loaded ticker mapping from cache: {ticker_cache_path}")
                logger.info(f"Cache contains {len(ticker_volumes)} tickers")
        except Exception as e:
            logger.warning(f"Failed to load ticker cache: {e}. Will recompute.")
            use_cache = False

    if not use_cache:
        # Load and rank tickers by average volume using parallel processing
        ticker_data = {}

        # Process enough files to get the requested number of tickers
        # Assuming ~80% success rate, process 1.5x the requested amount for safety
        files_needed = min(len(csv_files), int(n_tickers * 1.5))
        files_to_process = csv_files[:files_needed]

        # Use limited CPU cores for parallel processing to avoid overloading system
        n_cores = min(n_cores, mp.cpu_count())
        logger.info(f"Processing {len(files_to_process)} CSV files using {n_cores} CPU cores to get {n_tickers} tickers")

        # Create input data for parallel processing
        csv_file_info = [(csv_file, 20) for csv_file in files_to_process]  # 20 is max_samples parameter

        # Process CSV files IN PARALLEL for 2nd stage ticker retrieval
        logger.info("Processing files in parallel for ticker ranking...")
        try:
            with mp.Pool(processes=n_cores) as pool:
                results = pool.map(_process_csv_for_data, csv_file_info)
            logger.info(f"Parallel processing completed with {n_cores} workers")
        except Exception as e:
            logger.warning(f"Parallel processing failed ({e}), falling back to serial processing...")
            results = [_process_csv_for_data(info) for info in csv_file_info]

        # Collect results and log warnings
        ticker_files_new = {}  # Store file paths instead of dataframes
        skipped_tickers = []
        for ticker, result in results:
            if ticker is not None and result is not None:
                csv_file, avg_volume = result
                ticker_volumes[ticker] = avg_volume
                ticker_files_new[ticker] = csv_file
            elif ticker is not None:
                skipped_tickers.append(ticker)

        # Log skipped tickers with more detailed information
        if skipped_tickers:
            logger.info(f"Skipped {len(skipped_tickers)} tickers due to missing required columns or insufficient data")
            logger.debug(f"Skipped tickers: {', '.join(skipped_tickers[:20])}" +
                        (f" ... and {len(skipped_tickers) - 20} more" if len(skipped_tickers) > 20 else ""))

        ticker_files = ticker_files_new

        if not ticker_volumes:
            raise ValueError("No valid ticker data found. Cannot load market data.")

        # Save ticker mapping cache for future use
        try:
            cache_data = {
                'ticker_volumes': ticker_volumes,
                'ticker_files': ticker_files
            }
            with open(ticker_cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
            logger.info(f"Saved ticker mapping cache to: {ticker_cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save ticker cache: {e}")

    # Select top n_tickers by volume
    top_tickers = sorted(ticker_volumes.items(), key=lambda x: x[1], reverse=True)[:n_tickers]

    logger.info(f"Loaded {len(top_tickers)} tickers: {[ticker for ticker, _ in top_tickers]}")

    # Now load the actual data SERIALLY for the top tickers only
    ticker_data = {}
    logger.info(f"Loading data for top {len(top_tickers)} tickers (serial)...")
    for ticker, _ in top_tickers:
        try:
            csv_file = ticker_files[ticker]
            df = pd.read_csv(csv_file)
            df.columns = df.columns.str.lower()

            # Process same as in _process_csv_for_data
            if 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'])
            elif 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
            else:
                continue

            df = df[df['datetime'] >= TWO_YEARS_AGO]
            df.set_index('datetime', inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volume']].sort_index()
            df = df[(df > 0).all(axis=1)].dropna()

            ticker_data[ticker] = df
        except Exception as e:
            logger.error(f"Error loading {ticker}: {e}")

    # Return ticker metadata and file paths for one-by-one processing
    # This avoids loading all tickers into memory at once
    logger.info(f"Prepared {len(top_tickers)} tickers for one-by-one processing")

    return {
        'tickers': [ticker for ticker, _ in top_tickers],
        'ticker_files': ticker_files,
        'ticker_data': ticker_data  # Keep loaded data for reference
    }
