import pandas as pd
import pytz
from datetime import datetime
import re

def convert_timezone_to_et(timestamp_str):
    """
    Convert various timezone formats to Eastern Time (ET)
    """
    if pd.isna(timestamp_str) or timestamp_str == 'N/A' or not timestamp_str.strip():
        return None
    
    try:
        # Clean up the timestamp string
        timestamp_str = timestamp_str.strip()
        
        # Handle different timezone patterns
        timezone_mapping = {
            'GMT': 'UTC',
            'UTC': 'UTC', 
            'EDT': 'US/Eastern',
            'EST': 'US/Eastern',
            'ET': 'US/Eastern',
            'JST': 'Asia/Tokyo',
            'N/A': None
        }
        
        # Extract timezone from the end of the string
        tz_pattern = r'\b(GMT|UTC|EDT|EST|ET|JST|N/A)\b$'
        tz_match = re.search(tz_pattern, timestamp_str)
        
        if not tz_match:
            # If no timezone found, skip this entry
            return None
            
        timezone_abbr = tz_match.group(1)
        
        # Skip N/A timezones
        if timezone_abbr == 'N/A':
            return None
            
        # Get the timezone
        source_tz_name = timezone_mapping.get(timezone_abbr)
        if not source_tz_name:
            return None
        
        # Extract datetime part (everything before timezone)
        datetime_part = timestamp_str[:tz_match.start()].strip()
        
        # Parse the datetime
        dt = datetime.strptime(datetime_part, '%Y-%m-%d %H:%M:%S')
        
        # Set source timezone
        source_tz = pytz.timezone(source_tz_name)
        
        # If it's already ET/EST/EDT, just localize
        if source_tz_name == 'US/Eastern':
            localized_dt = source_tz.localize(dt)
        else:
            # Localize to source timezone then convert to ET
            localized_dt = source_tz.localize(dt)
            et_tz = pytz.timezone('US/Eastern')
            localized_dt = localized_dt.astimezone(et_tz)
        
        # Return as string in ET
        return localized_dt.strftime('%Y-%m-%d %H:%M:%S %Z')
        
    except Exception as e:
        print(f"Error processing timestamp '{timestamp_str}': {e}")
        return None

def main():
    # Read the timestamps CSV
    print("Reading timestamps CSV...")
    timestamps_df = pd.read_csv('data/timestamps_from_csv.csv')
    print(f"Original timestamps data: {len(timestamps_df)} rows")
    
    # Read the tickers CSV  
    print("Reading extracted tickers CSV...")
    tickers_df = pd.read_csv('data/extracted_tickers.csv')
    print(f"Tickers data: {len(tickers_df)} rows")
    
    # Convert timestamps to ET and filter out malformed/N/A entries
    print("Converting timestamps to ET...")
    timestamps_df['report_timestamp_et'] = timestamps_df['report_timestamp'].apply(convert_timezone_to_et)
    
    # Remove rows with failed timestamp conversions
    before_filter = len(timestamps_df)
    timestamps_df_clean = timestamps_df.dropna(subset=['report_timestamp_et'])
    after_filter = len(timestamps_df_clean)
    print(f"Filtered timestamps: {before_filter} -> {after_filter} rows ({before_filter - after_filter} removed)")
    
    # Join the datasets on full_file_path
    print("Joining datasets on full_file_path...")
    joined_df = timestamps_df_clean.merge(tickers_df, on='full_file_path', how='inner', suffixes=('_timestamps', '_tickers'))
    
    print(f"Joined data: {len(joined_df)} rows")
    
    # Reorder columns for better readability
    column_order = [
        'date_timestamps', 'filename_timestamps', 'full_file_path', 
        'report_timestamp', 'report_timestamp_et',
        'ticker', 'price_target', 'eps_target'
    ]
    
    # Only keep columns that exist
    existing_columns = [col for col in column_order if col in joined_df.columns]
    joined_df_ordered = joined_df[existing_columns]
    
    # Save the result
    output_path = 'data/timestamps_tickers_joined.csv'
    joined_df_ordered.to_csv(output_path, index=False)
    print(f"Saved joined dataset to: {output_path}")
    
    # Display some statistics
    print(f"\nDataset Statistics:")
    print(f"- Total joined rows: {len(joined_df_ordered)}")
    print(f"- Unique files: {joined_df_ordered['full_file_path'].nunique()}")
    print(f"- Unique tickers: {joined_df_ordered['ticker'].nunique()}")
    print(f"- Files with multiple tickers: {(joined_df_ordered.groupby('full_file_path').size() > 1).sum()}")
    
    # Show sample of the result
    print(f"\nSample of joined data:")
    print(joined_df_ordered.head(10).to_string())
    
    return joined_df_ordered

if __name__ == "__main__":
    result = main()