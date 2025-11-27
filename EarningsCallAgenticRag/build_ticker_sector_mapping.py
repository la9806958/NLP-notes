import pandas as pd
import yfinance as yf
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os
from datetime import datetime

def get_sector_for_ticker(ticker):
    """Get GICS sector for a single ticker using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        sector = info.get('sector', 'Unknown')
        industry = info.get('industry', 'Unknown')
        return ticker, sector, industry
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return ticker, 'Unknown', 'Unknown'

def main():
    print(f"Starting ticker sector mapping process at {datetime.now()}")

    # Read the CSV file
    csv_file = 'earnings_returns_filtered_results.csv'
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found in current directory")
        return None

    print(f"Reading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} rows of data")

    # Get unique tickers
    unique_tickers = df['ticker'].unique()
    print(f"Found {len(unique_tickers)} unique tickers")

    # Check if existing mapping exists to avoid re-processing
    existing_mapping = {}
    if os.path.exists('ticker_sector_mapping.json'):
        print("Found existing mapping file, loading...")
        with open('ticker_sector_mapping.json', 'r') as f:
            existing_mapping = json.load(f)
        print(f"Loaded {len(existing_mapping)} existing mappings")

    # Filter out tickers that already have mappings
    tickers_to_process = [t for t in unique_tickers if t not in existing_mapping]
    print(f"Need to process {len(tickers_to_process)} new tickers")

    # Create ticker to sector mapping starting with existing data
    ticker_sector_mapping = existing_mapping.copy()
    ticker_industry_mapping = {}

    if tickers_to_process:
        print("Fetching sector and industry data from Yahoo Finance...")
        # Use threading to speed up API calls
        with ThreadPoolExecutor(max_workers=5) as executor:  # Reduced workers to be more respectful
            # Submit all tasks
            future_to_ticker = {executor.submit(get_sector_for_ticker, ticker): ticker
                               for ticker in tickers_to_process}

            # Collect results
            for i, future in enumerate(as_completed(future_to_ticker)):
                ticker, sector, industry = future.result()
                ticker_sector_mapping[ticker] = sector
                ticker_industry_mapping[ticker] = industry

                if (i + 1) % 25 == 0:  # More frequent updates
                    print(f"Processed {i + 1}/{len(tickers_to_process)} tickers")

                # Small delay to be respectful to the API
                time.sleep(0.2)  # Slightly longer delay

    # Print summary of sectors found
    sector_counts = {}
    for sector in ticker_sector_mapping.values():
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    print(f"\nSector distribution ({len(ticker_sector_mapping)} total tickers):")
    for sector, count in sorted(sector_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{sector}: {count} tickers")

    # Save the mapping to JSON
    print(f"\nSaving mappings...")
    with open('ticker_sector_mapping.json', 'w') as f:
        json.dump(ticker_sector_mapping, f, indent=2)

    # Create a CSV with ticker-sector mapping
    mapping_data = []
    for ticker in ticker_sector_mapping:
        sector = ticker_sector_mapping[ticker]
        industry = ticker_industry_mapping.get(ticker, 'Unknown')
        mapping_data.append({'ticker': ticker, 'gics_sector': sector, 'industry': industry})

    mapping_df = pd.DataFrame(mapping_data)
    mapping_df = mapping_df.sort_values('ticker')
    mapping_df.to_csv('ticker_sector_mapping.csv', index=False)

    # Create a detailed summary
    summary_stats = {
        'total_tickers': len(ticker_sector_mapping),
        'total_sectors': len(sector_counts),
        'unknown_sectors': sector_counts.get('Unknown', 0),
        'processed_date': datetime.now().isoformat(),
        'sector_breakdown': sector_counts
    }

    with open('ticker_sector_summary.json', 'w') as f:
        json.dump(summary_stats, f, indent=2)

    print(f"\nFiles created:")
    print("- ticker_sector_mapping.json (ticker -> sector mapping)")
    print("- ticker_sector_mapping.csv (CSV format with industry)")
    print("- ticker_sector_summary.json (summary statistics)")
    print(f"\nCompleted at {datetime.now()}")

    return ticker_sector_mapping

if __name__ == "__main__":
    mapping = main()