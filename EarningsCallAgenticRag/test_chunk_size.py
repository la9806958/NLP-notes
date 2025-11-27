#!/usr/bin/env python3
"""
Test script to show the impact of the new chunk size
"""
import pandas as pd
import math
import sys
import os

# Import the constants
sys.path.append('.')
from orchestrator_parallel_facts import CHUNK_SIZE, load_processed_items

def test_chunk_impact():
    """Test the impact of new chunk size"""
    print("🧪 Testing Impact of New Chunk Size")
    print("=" * 50)

    try:
        # Load data
        df = pd.read_csv("EarningsFilteredResults2.csv")
        print(f"📊 Total items in dataset: {len(df)}")

        # Load processed items
        processed_items = load_processed_items()
        print(f"📊 Already processed items: {len(processed_items)}")

        # Simulate filtering
        if processed_items:
            df['processed_key'] = list(zip(df['ticker'], df['future_3bday_cum_return']))
            unprocessed_mask = ~df['processed_key'].isin(processed_items)
            df = df[unprocessed_mask].drop('processed_key', axis=1)

        # Calculate unique tickers
        unique_tickers = df['ticker'].unique()
        total_tickers = len(unique_tickers)

        print(f"📋 Remaining unique tickers to process: {total_tickers}")
        print(f"📋 Remaining rows to process: {len(df)}")

        # Calculate chunks with new size
        total_chunks = math.ceil(total_tickers / CHUNK_SIZE)

        print(f"\n📦 Chunk Analysis:")
        print(f"  • Chunk size: {CHUNK_SIZE} tickers")
        print(f"  • Total chunks needed: {total_chunks}")
        print(f"  • Average rows per chunk: {len(df) / total_chunks:.1f}")

        # Show first few chunks
        print(f"\n📋 First 5 chunks breakdown:")
        for chunk_idx in range(min(5, total_chunks)):
            start_ticker_idx = chunk_idx * CHUNK_SIZE
            end_ticker_idx = min((chunk_idx + 1) * CHUNK_SIZE, total_tickers)
            chunk_tickers = unique_tickers[start_ticker_idx:end_ticker_idx]
            chunk_df = df[df['ticker'].isin(chunk_tickers)]

            print(f"  Chunk {chunk_idx + 1}: {len(chunk_tickers)} tickers, {len(chunk_df)} rows")
            print(f"    Tickers: {', '.join(chunk_tickers[:3])}{'...' if len(chunk_tickers) > 3 else ''}")

        if total_chunks > 5:
            print(f"  ... and {total_chunks - 5} more chunks")

        print(f"\n⏱️  Estimated Processing:")
        print(f"  • With resume functionality: {total_chunks} chunks to process")
        print(f"  • Neo4j will be cleared {total_chunks} times (after each chunk)")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_chunk_impact()