#!/usr/bin/env python3
"""
Test script to verify the resume functionality of orchestrator_parallel_facts.py
"""
import pandas as pd
import sys
import os

# Import the load_processed_items function
sys.path.append('.')
from orchestrator_parallel_facts import load_processed_items

def test_load_processed_items():
    """Test the load_processed_items function"""
    print("🧪 Testing load_processed_items function...")

    # Test loading from existing FinalResults.csv
    processed_items = load_processed_items()

    print(f"📊 Loaded {len(processed_items)} processed items")

    if len(processed_items) > 0:
        # Show first 10 items
        sample_items = list(processed_items)[:10]
        print(f"📝 First 10 processed items:")
        for i, item in enumerate(sample_items, 1):
            ticker, return_val = item
            print(f"  {i}. {ticker}: {return_val}")

    return processed_items

def test_filtering_logic():
    """Test the filtering logic against actual data"""
    print("\n🧪 Testing filtering logic...")

    # Load the data files
    try:
        df = pd.read_csv("EarningsFilteredResults2.csv")
        print(f"📊 Loaded {len(df)} rows from EarningsFilteredResults2.csv")

        # Load processed items
        processed_items = load_processed_items()

        if len(processed_items) == 0:
            print("⚠️  No processed items found, cannot test filtering")
            return

        # Simulate the filtering logic
        initial_count = len(df)
        df['processed_key'] = list(zip(df['ticker'], df['future_3bday_cum_return']))
        unprocessed_mask = ~df['processed_key'].isin(processed_items)
        filtered_df = df[unprocessed_mask].drop('processed_key', axis=1)

        filtered_count = initial_count - len(filtered_df)
        print(f"✅ Would filter out {filtered_count} already processed items")
        print(f"📋 Would process {len(filtered_df)} remaining items")

        # Show some examples of what would be filtered
        if filtered_count > 0:
            processed_df = df[~unprocessed_mask]
            print(f"📝 Sample items that would be skipped:")
            for i, (_, row) in enumerate(processed_df.head(5).iterrows(), 1):
                print(f"  {i}. {row['ticker']}: {row['future_3bday_cum_return']}")

    except Exception as e:
        print(f"❌ Error testing filtering logic: {e}")

if __name__ == "__main__":
    print("🚀 Testing Resume Functionality")
    print("=" * 50)

    # Test 1: Load processed items
    processed_items = test_load_processed_items()

    # Test 2: Test filtering logic
    test_filtering_logic()

    print("\n" + "=" * 50)
    print("✅ Resume functionality testing complete!")