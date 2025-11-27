import pandas as pd
import yfinance as yf
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def get_sector_for_ticker(ticker):
    """Get GICS sector for a single ticker using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        sector = info.get('sector', 'Unknown')
        return ticker, sector
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return ticker, 'Unknown'

def main():
    # Read the CSV file
    df = pd.read_csv('earnings_returns_filtered_results.csv')

    # Get unique tickers
    unique_tickers = df['ticker'].unique()
    print(f"Found {len(unique_tickers)} unique tickers")

    # Create ticker to sector mapping
    ticker_sector_mapping = {}

    # Use threading to speed up API calls
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all tasks
        future_to_ticker = {executor.submit(get_sector_for_ticker, ticker): ticker
                           for ticker in unique_tickers}

        # Collect results
        for i, future in enumerate(as_completed(future_to_ticker)):
            ticker, sector = future.result()
            ticker_sector_mapping[ticker] = sector

            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(unique_tickers)} tickers")

            # Small delay to be respectful to the API
            time.sleep(0.1)

    # Print summary of sectors found
    sector_counts = {}
    for sector in ticker_sector_mapping.values():
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    print("\nSector distribution:")
    for sector, count in sorted(sector_counts.items()):
        print(f"{sector}: {count} tickers")

    # Save the mapping to JSON
    with open('ticker_sector_mapping.json', 'w') as f:
        json.dump(ticker_sector_mapping, f, indent=2)

    # Create a CSV with ticker-sector mapping
    mapping_df = pd.DataFrame(list(ticker_sector_mapping.items()),
                             columns=['ticker', 'gics_sector'])
    mapping_df.to_csv('ticker_sector_mapping.csv', index=False)

    print(f"\nMapping saved to:")
    print("- ticker_sector_mapping.json")
    print("- ticker_sector_mapping.csv")

    return ticker_sector_mapping

if __name__ == "__main__":
    mapping = main()