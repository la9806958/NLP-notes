#!/usr/bin/env python3
"""
Hybrid Ticker Extractor
1. First tries to extract ticker from original_title (in brackets)
2. If no ticker in title, falls back to transcript extraction with enhanced filtering
3. Only keeps ET timezone entries
"""

import pandas as pd
import re
import argparse
from typing import Optional

def extract_ticker_from_title(title: str) -> Optional[str]:
    """Extract ticker symbol from title enclosed in parentheses."""
    if not title or pd.isna(title):
        return None
    
    title_str = str(title)
    
    # Look for ticker in parentheses format: "Company Name (TICKER)"
    # Pattern looks for parentheses containing 1-5 uppercase letters
    pattern = r'\(([A-Z]{1,5})\)'
    matches = re.findall(pattern, title_str)
    
    if matches:
        # Return the first ticker found in parentheses
        ticker = matches[0].upper().strip()
        # Additional validation for title tickers
        if len(ticker) >= 1 and len(ticker) <= 5 and ticker.isalpha():
            return ticker
    
    return None

def extract_ticker_from_transcript(transcript: str) -> Optional[str]:
    """Extract ticker symbol from transcript text with enhanced filtering."""
    if not transcript or pd.isna(transcript):
        return None
            
    # Focus on the first 1000 characters where ticker is most likely mentioned
    text = str(transcript).strip()[:1000]
    
    # Enhanced false positives list - including common misidentifications
    false_positives = {
        # Stock exchanges and trading platforms
        'NYSE', 'NASDAQ', 'NASD', 'OTC', 'OTCBB', 'TSX', 'LSE', 'AMEX',
        
        # Time zones and geographical codes
        'ET', 'EST', 'EDT', 'PT', 'PST', 'PDT', 'CT', 'CST', 'CDT', 
        'MT', 'MST', 'MDT', 'GMT', 'UTC', 'BST', 'CET', 'JST',
        
        # Corporate titles and roles
        'CEO', 'CFO', 'CTO', 'COO', 'CRO', 'CMO', 'CIO', 'CLO',
        
        # Financial terms and periods
        'GAAP', 'EBITDA', 'CAPEX', 'OPEX', 'ROI', 'ROE', 'EPS',
        'Q1', 'Q2', 'Q3', 'Q4', 'FY', 'YOY', 'QOQ', 'YTD',
        
        # Countries and regions
        'USA', 'USD', 'US', 'UK', 'EU', 'CA', 'AU', 'CN', 'JP',
        
        # Business entity types
        'INC', 'LLC', 'LTD', 'CORP', 'CO', 'LP', 'PC', 'PLC',
        
        # Common words and prepositions
        'THE', 'AND', 'FOR', 'WITH', 'FROM', 'INTO', 'OVER', 'UNDER',
        'ABOUT', 'ABOVE', 'BELOW', 'AFTER', 'BEFORE', 'SINCE', 'UNTIL',
        
        # Web and tech terms
        'HTTPS', 'HTTP', 'WWW', 'COM', 'NET', 'ORG', 'GOV', 'EDU',
        
        # Meeting and call terms
        'CALL', 'CONF', 'MEET', 'WEBCAST', 'LIVE', 'EVENT',
        
        # Business terms
        'GROUP', 'TEAM', 'STAFF', 'BOARD', 'PANEL',
        
        # Time and date terms  
        'YEAR', 'MONTH', 'WEEK', 'DAY', 'TODAY', 'TOMORROW',
        
        # Numerical terms
        'MILLION', 'BILLION', 'THOUSAND', 'PERCENT',
        
        # Ordinal numbers
        'FIRST', 'SECOND', 'THIRD', 'FOURTH', 'FIFTH', 'SIXTH',
        
        # Business metrics
        'GROWTH', 'REVENUE', 'PROFIT', 'MARGIN', 'SALES', 'LOSS',
        
        # Common greetings and phrases
        'GOOD', 'THANK', 'THANKS', 'HELLO', 'WELCOME', 'PLEASE',
        
        # Conference call artifacts
        'CONFERENCE', 'MEETING', 'SESSION', 'PRESENTATION', 'UPDATE',
        
        # Suffixes that might get extracted
        'ATION', 'MENT', 'TION', 'SION', 'NESS', 'ABLE', 'IBLE'
    }
    
    # Improved ticker symbol patterns - focusing on parentheses format first
    ticker_patterns = [
        # Primary pattern: Company Name (TICKER) - most reliable  
        r'\b[A-Za-z][^()]*\(([A-Z]{1,5})\)',
        
        # Secondary patterns: TICKER at start of transcript with context
        r'^([A-Z]{2,5})\s+(?:Q[1-4]\s+\d{4}|earnings|call|results|reports|announces)',
        
        # Trading symbol patterns with context
        r'(?:NYSE|NASDAQ):\s*([A-Z]{2,5})\b',
        r'(?:Symbol|Ticker):\s*([A-Z]{2,5})\b',
        
        # Company mentions with ticker and business context
        r'\b([A-Z]{2,5})\s+(?:Incorporated|Inc\.?|Corporation|Corp\.?|Company|Co\.?|Ltd\.?)(?:\s+\([^)]*\))?',
        
        # Pattern for earnings call format: "TICKER Q1 2023"
        r'\b([A-Z]{2,5})\s+Q[1-4]\s+\d{4}',
        
        # Fallback: isolated ticker-like sequences with strong business context
        r'\b([A-Z]{2,5})\s+(?:Reports?|Announces?|Delivers?|Releases?|Beats?|Misses?)',
    ]
    
    # Try each pattern in order of reliability
    for pattern in ticker_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            for match in matches:
                ticker = match.upper().strip()
                # Validate ticker format and exclude false positives
                if (len(ticker) >= 2 and len(ticker) <= 5 and 
                    ticker.isalpha() and 
                    ticker not in false_positives):
                    return ticker
    
    return None

def has_et_timezone(timestamp: str) -> bool:
    """Check if timestamp contains ET timezone."""
    if not timestamp or pd.isna(timestamp):
        return False
    
    timestamp_str = str(timestamp).upper()
    
    # Check for various ET formats
    et_patterns = [
        r'\bET\b',    # Exact match for ET
        r'\bEST\b',   # Eastern Standard Time
        r'\bEDT\b'    # Eastern Daylight Time
    ]
    
    for pattern in et_patterns:
        if re.search(pattern, timestamp_str):
            return True
    
    return False

def extract_hybrid_ticker(row) -> tuple:
    """
    Extract ticker using hybrid approach:
    1. Try title first
    2. Fall back to transcript if title has no ticker
    
    Returns: (ticker, source) where source is 'title' or 'transcript'
    """
    # Try title first
    title_ticker = extract_ticker_from_title(row['original_title'])
    if title_ticker:
        return title_ticker, 'title'
    
    # Fall back to transcript
    transcript_ticker = extract_ticker_from_transcript(row['cleaned_transcript'])
    if transcript_ticker:
        return transcript_ticker, 'transcript'
    
    return None, 'none'

def process_with_hybrid_ticker(input_file: str, output_file: str) -> dict:
    """
    Process earnings data using hybrid ticker extraction.
    Only filters for ET timezone entries.
    
    Returns:
        dict: Statistics about the processing
    """
    print(f"Processing data from {input_file} using hybrid ticker extraction...")
    
    # Read the data in chunks to handle large files
    chunk_size = 10000
    processed_chunks = []
    
    total_rows = 0
    no_ticker_dropped = 0
    non_et_dropped = 0
    kept_rows = 0
    title_source_count = 0
    transcript_source_count = 0
    
    for chunk_num, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size)):
        print(f"Processing chunk {chunk_num + 1} ({len(chunk)} rows)...")
        total_rows += len(chunk)
        
        # Apply hybrid ticker extraction
        ticker_results = chunk.apply(extract_hybrid_ticker, axis=1)
        chunk['hybrid_ticker'] = [result[0] for result in ticker_results]
        chunk['ticker_source'] = [result[1] for result in ticker_results]
        
        # Count sources for this chunk
        chunk_title_count = sum(1 for source in chunk['ticker_source'] if source == 'title')
        chunk_transcript_count = sum(1 for source in chunk['ticker_source'] if source == 'transcript')
        title_source_count += chunk_title_count
        transcript_source_count += chunk_transcript_count
        
        # Filter 1: Keep only rows where we can extract a ticker
        before_ticker_filter = len(chunk)
        has_ticker = chunk['hybrid_ticker'].notna()
        chunk_filtered = chunk[has_ticker].copy()
        no_ticker_dropped += (before_ticker_filter - len(chunk_filtered))
        
        print(f"  - After hybrid ticker filter: {len(chunk_filtered)} rows (dropped {before_ticker_filter - len(chunk_filtered)})")
        print(f"    * Title source: {chunk_title_count}, Transcript source: {chunk_transcript_count}")
        
        # Filter 2: Keep only rows with ET timezone
        before_et_filter = len(chunk_filtered)
        et_mask = chunk_filtered['timestamp'].apply(has_et_timezone)
        chunk_final = chunk_filtered[et_mask].copy()
        non_et_dropped += (before_et_filter - len(chunk_final))
        
        print(f"  - After ET timezone filter: {len(chunk_final)} rows (dropped {before_et_filter - len(chunk_final)})")
        
        # Replace the original ticker with hybrid ticker
        if len(chunk_final) > 0:
            chunk_final['ticker'] = chunk_final['hybrid_ticker']
            
            # Drop the helper columns
            chunk_final = chunk_final.drop(['hybrid_ticker', 'ticker_source'], axis=1)
            
            kept_rows += len(chunk_final)
            processed_chunks.append(chunk_final)
    
    # Combine all chunks
    if processed_chunks:
        print("\nCombining processed chunks...")
        final_df = pd.concat(processed_chunks, ignore_index=True)
        
        # Save processed data
        print(f"Saving processed data to {output_file}...")
        final_df.to_csv(output_file, index=False)
        
        # Generate statistics
        stats = {
            'original_rows': total_rows,
            'no_ticker_dropped': no_ticker_dropped,
            'non_et_timezone_dropped': non_et_dropped,
            'final_rows': kept_rows,
            'total_dropped': total_rows - kept_rows,
            'retention_rate': (kept_rows / total_rows) * 100 if total_rows > 0 else 0,
            'title_source_count': title_source_count,
            'transcript_source_count': transcript_source_count
        }
        
        print("\n" + "="*60)
        print("HYBRID TICKER PROCESSING STATISTICS")
        print("="*60)
        print(f"Original rows:                {stats['original_rows']:,}")
        print(f"Dropped (no ticker found):    {stats['no_ticker_dropped']:,}")
        print(f"Dropped (non-ET timezone):    {stats['non_et_timezone_dropped']:,}")
        print(f"Total dropped:                {stats['total_dropped']:,}")
        print(f"Final rows kept:              {stats['final_rows']:,}")
        print(f"Retention rate:               {stats['retention_rate']:.1f}%")
        print()
        print("TICKER SOURCE BREAKDOWN:")
        print(f"From title (parentheses):     {stats['title_source_count']:,}")
        print(f"From transcript (fallback):   {stats['transcript_source_count']:,}")
        if stats['final_rows'] > 0:
            title_pct = (stats['title_source_count'] / stats['final_rows']) * 100
            transcript_pct = (stats['transcript_source_count'] / stats['final_rows']) * 100
            print(f"Title percentage:             {title_pct:.1f}%")
            print(f"Transcript percentage:        {transcript_pct:.1f}%")
        print("="*60)
        
        # Show sample of processed data
        print("\nSample of processed data (hybrid approach):")
        sample_cols = ['ticker', 'timestamp', 'original_title']
        if all(col in final_df.columns for col in sample_cols):
            sample_df = final_df[sample_cols].head(10)
            for idx, row in sample_df.iterrows():
                title_preview = str(row['original_title'])[:80] + "..." if len(str(row['original_title'])) > 80 else str(row['original_title'])
                print(f"  {row['ticker']:>6} | {row['timestamp']} | {title_preview}")
        
        # Show ticker distribution
        print(f"\nTicker distribution (top 15):")
        ticker_counts = final_df['ticker'].value_counts().head(15)
        for ticker, count in ticker_counts.items():
            print(f"  {ticker}: {count} occurrences")
        
        return stats
    else:
        print("No data remaining after filtering!")
        return {
            'original_rows': total_rows,
            'no_ticker_dropped': no_ticker_dropped, 
            'non_et_timezone_dropped': non_et_dropped,
            'final_rows': 0,
            'total_dropped': total_rows,
            'retention_rate': 0,
            'title_source_count': 0,
            'transcript_source_count': 0
        }

def main():
    parser = argparse.ArgumentParser(
        description='Process earnings call data using hybrid ticker extraction (title first, transcript fallback)'
    )
    parser.add_argument('--input', '-i', required=True,
                       help='Input CSV file path')
    parser.add_argument('--output', '-o', required=True,
                       help='Output CSV file path')
    
    args = parser.parse_args()
    
    print("Starting hybrid ticker processing...")
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    
    # Process the data
    stats = process_with_hybrid_ticker(args.input, args.output)
    
    print(f"\nHybrid processing completed successfully!")
    print(f"Processed data saved to: {args.output}")

if __name__ == "__main__":
    main()