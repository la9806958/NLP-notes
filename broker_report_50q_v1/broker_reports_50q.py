#!/usr/bin/env python3
"""
V1 of Broker Report Questions
Portfolio Allocation Agent with 50 Questions
Modified version that first runs ticker extraction, then processes portfolio allocation decisions.

This agent first extracts US tickers from broker reports using LLM, then processes filtered 
reports and earnings calls data to make portfolio allocation decisions using GPT-4o-mini.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from openai import OpenAI
import os
import json
from typing import Dict, List, Optional, Tuple
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import re
import subprocess
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PortfolioAgent50Q:
    def __init__(self, credentials_path: str = None, api_key: str = None, us_tickers_path: str = None):
        """Initialize the portfolio agent with OpenAI API key from credentials file or direct key."""
        if credentials_path:
            with open(credentials_path, 'r') as f:
                credentials = json.load(f)
            api_key = credentials.get('openai_api_key')
            if not api_key:
                raise ValueError("openai_api_key not found in credentials.json")
        elif not api_key:
            # Try to load from base folder credentials.json as default
            try:
                with open('/home/lichenhui/credentials.json', 'r') as f:
                    credentials = json.load(f)
                api_key = credentials.get('openai_api_key')
                if not api_key:
                    raise ValueError("openai_api_key not found in credentials.json")
            except FileNotFoundError:
                raise ValueError("Either credentials_path or api_key must be provided, or credentials.json must exist in base folder")
            
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"
        
        # Load US tickers list from open_to_open_returns.csv
        if not us_tickers_path:
            us_tickers_path = 'data/open_to_open_returns.csv'
        
        self.us_tickers = self.load_us_tickers(us_tickers_path)
        self.extracted_ticker_data = None
        
    def load_csv_ticker_data(self, csv_path: str) -> pd.DataFrame:
        """Load ticker data from CSV file."""
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} ticker entries from CSV")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV ticker data: {e}")
            return pd.DataFrame()
    
    def get_extracted_tickers_for_report(self, file_path: str, date: str) -> List[str]:
        """Get extracted tickers for a specific report."""
        if self.extracted_ticker_data is None:
            logger.debug(f"No extracted ticker data loaded for {file_path}")
            return []
        
        if isinstance(self.extracted_ticker_data, pd.DataFrame):
            # CSV data - filter by file path only (ignore date from extracted_tickers.csv)
            matching_rows = self.extracted_ticker_data[
                self.extracted_ticker_data['full_file_path'] == file_path
            ]
            tickers = matching_rows['ticker'].unique().tolist()
            logger.debug(f"Found {len(tickers)} extracted tickers for {file_path}: {tickers}")
            return tickers
        else:
            # JSON data (legacy support)
            for report_data in self.extracted_ticker_data:
                if report_data.get('file_path') == file_path:
                    ticker_data = report_data.get('ticker_data', [])
                    return [ticker_info[0] for ticker_info in ticker_data if len(ticker_info) > 0]
        
        logger.debug(f"No extracted tickers found for {file_path}")
        return []
        
    def load_us_tickers(self, us_tickers_path: str) -> set:
        """Load US tickers from open_to_open_returns.csv columns."""
        try:
            # Read just the header to get column names
            returns_df = pd.read_csv(us_tickers_path, nrows=0)
            # Get all columns except DateTime
            us_tickers = set(col for col in returns_df.columns if col != 'DateTime')
            logger.info(f"Loaded {len(us_tickers)} US tickers for filtering")
            return us_tickers
        except Exception as e:
            logger.error(f"Error loading US tickers: {e}")
            return set()
    
    def load_datasets(self, broker_reports_path: str, earnings_calls_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and preprocess the datasets."""
        try:
            # Load broker reports
            broker_df = pd.read_csv(broker_reports_path)
            broker_df['selected_date'] = pd.to_datetime(broker_df['selected_date']).dt.tz_localize(None)
            logger.info(f"Loaded {len(broker_df)} broker reports")
            
            # Load earnings calls
            earnings_df = pd.read_csv(earnings_calls_path)
            earnings_df['date'] = pd.to_datetime(earnings_df['date']).dt.tz_localize(None)
            logger.info(f"Loaded {len(earnings_df)} earnings calls")
            
            return broker_df, earnings_df
            
        except Exception as e:
            logger.error(f"Error loading datasets: {e}")
            raise
    
    def read_broker_report(self, file_path: str) -> str:
        """Read broker report content from file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content[:10000]  # Limit to first 10k characters
        except Exception as e:
            logger.warning(f"Could not read broker report {file_path}: {e}")
            return f"[Could not read broker report: {file_path}]"
    
    def match_data_by_date(self, broker_df: pd.DataFrame, earnings_df: pd.DataFrame, 
                          target_date: datetime, window_days: int = 1) -> Tuple[List[Dict], List[Dict]]:
        """Match broker reports and earnings calls within a date window."""
        start_date = target_date
        end_date = target_date + timedelta(days=window_days)
        
        # Filter broker reports (exclude end_date)
        broker_matches = broker_df[
            (broker_df['selected_date'] >= start_date) & 
            (broker_df['selected_date'] < end_date)
        ].copy()
        
        # Filter earnings calls - use inclusive date range to capture more matches
        earnings_matches = earnings_df[
            (earnings_df['date'] >= start_date) & 
            (earnings_df['date'] < end_date)
        ].copy()
        
        logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        logger.info(f"Broker matches before filtering: {len(broker_matches)}")
        logger.info(f"Earnings matches before filtering: {len(earnings_matches)}")
        
        # Debug: Check if extracted ticker data is loaded
        if self.extracted_ticker_data is not None:
            logger.info(f"Extracted ticker data loaded with {len(self.extracted_ticker_data)} entries")
            # Sample some file paths from broker matches and extracted data for comparison
            if len(broker_matches) > 0:
                sample_broker_path = broker_matches.iloc[0]['file_path']
                logger.info(f"Sample broker file path: {sample_broker_path}")
                sample_tickers = self.get_extracted_tickers_for_report(sample_broker_path, target_date.strftime('%Y-%m-%d'))
                logger.info(f"Sample broker extracted tickers: {sample_tickers}")
        else:
            logger.warning("Extracted ticker data is None!")
        
        broker_data = []
        for _, row in broker_matches.iterrows():
            content = self.read_broker_report(row['file_path'])
            report_date = row['selected_date'].strftime('%Y-%m-%d')
            
            # Check if this report has extracted tickers (from ticker extractor)
            extracted_tickers = self.get_extracted_tickers_for_report(row['file_path'], report_date)
            
            if extracted_tickers:
                broker_data.append({
                    'date': report_date,
                    'file_path': row['file_path'],
                    'content': content,
                    'extracted_tickers': extracted_tickers
                })
                logger.info(f"Included broker report with tickers {extracted_tickers}: {row['file_path']}")
            else:
                logger.debug(f"Filtered out broker report (no extracted US tickers): {row['file_path']}")
        
        earnings_data = []
        for _, row in earnings_matches.iterrows():
            earnings_data.append({
                'ticker': row['ticker'],
                'quarter': row['quarter'],
                'date': row['date'].strftime('%Y-%m-%d'),
                'transcript': row['transcript'] # Limit transcript length
            })
        
        logger.info(f"Found {len(broker_data)} broker reports and {len(earnings_data)} earnings calls for date {target_date.strftime('%Y-%m-%d')}")
        
        # Log sample data for debugging
        if earnings_data:
            logger.info(f"Sample earnings call: {earnings_data[0]['ticker']} on {earnings_data[0]['date']}")
        if broker_data:
            logger.info(f"Sample broker report: {broker_data[0]['date']}")
            
        return broker_data, earnings_data
    
    def create_allocation_prompt(self, broker_reports: List[Dict], earnings_calls: List[Dict], batch_num: int = 1) -> str:
        """Create the prompt for GPT-4o-mini allocation decision with 50 questions."""
        # Collect all relevant tickers from both data sources
        all_tickers = set()
        
        # Get tickers from broker reports (extracted tickers)
        for report in broker_reports:
            extracted_tickers = report.get('extracted_tickers', [])
            all_tickers.update(extracted_tickers)
        
        # Get tickers from earnings calls
        limited_earnings = earnings_calls[:3]  # Limit to 3 earnings calls per prompt
        for earnings in limited_earnings:
            all_tickers.add(earnings['ticker'])
        
        # Sort tickers for consistent ordering
        sorted_tickers = sorted(list(all_tickers))
        
        # Log the relevant tickers for debugging
        logger.info(f"Relevant tickers for batch {batch_num}: {sorted_tickers}")
        
        prompt = f"""You are a portfolio manager and you have access to the below events as of today.
{'(Batch ' + str(batch_num) + ')' if batch_num > 1 else ''}

RELEVANT TICKERS FOR ANALYSIS:
{', '.join(sorted_tickers)}

EARNINGS TRANSCRIPTS:
"""
        
        for i, earnings in enumerate(limited_earnings, 1):
            prompt += f"\n--- Earnings Call {i}: {earnings['ticker']} ({earnings['quarter']}) ---\n"
            prompt += f"Date: {earnings['date']}\n"
            prompt += f"Transcript: {earnings['transcript'][:8000]}...\n"  # Limit transcript length
        
        prompt += "\n\nBROKER REPORT TRANSCRIPTS:\n"
        
        # Limit broker reports per batch (already limited to 300 by caller)
        for i, report in enumerate(broker_reports, 1):
            prompt += f"\n--- Broker Report {i} ---\n"
            prompt += f"Date: {report['date']}\n"
            prompt += f"Extracted Tickers: {', '.join(report.get('extracted_tickers', []))}\n"
            prompt += f"Content: {report['content']}\n"
        
        prompt += f"""

ANALYSIS TARGET:
Focus your analysis on these specific tickers: {', '.join(sorted_tickers)}

1. For each US equity ticker that was extracted from the broker reports above, you must provide answers to the 50 questions below.

2. You must rate a ticker relative to other tickers discussed, on a scale of 0-10, whereby 0-5 indicates underperformance to peers and 6-10 indicates overperformance.

2. Focus only on the extracted tickers listed in each broker report.

3. You must only analyze US equities (common stocks listed on NYSE or NASDAQ; no ADRs or ETFs).

The 50 questions you must answer for each ticker (rate each on a scale of 0-10):

1. How strong is the company's revenue growth trajectory?
2. What is the quality of the company's earnings?
3. How sustainable are the company's profit margins?
4. What is the company's competitive position in its industry?
5. How effective is the company's management team?
6. What is the strength of the company's balance sheet?
7. How efficient is the company's capital allocation?
8. What is the company's cash flow generation capability?
9. How diversified are the company's revenue streams?
10. What is the company's exposure to economic cycles?
11. How strong is the company's brand recognition?
12. What is the quality of the company's customer base?
13. How innovative is the company's product/service offering?
14. What is the company's market share position?
15. How scalable is the company's business model?
16. What is the company's pricing power?
17. How strong are the company's regulatory barriers?
18. What is the company's technological advantage?
19. How efficient are the company's operations?
20. What is the company's supply chain resilience?
21. How strong is the company's corporate governance?
22. What is the quality of the company's disclosure practices?
23. How aligned are management incentives with shareholders?
24. What is the company's ESG (Environmental, Social, Governance) performance?
25. How strong is the company's risk management framework?
26. What is the company's ability to adapt to change?
27. How strong is the company's international presence?
28. What is the company's exposure to currency fluctuations?
29. How dependent is the company on key suppliers?
30. What is the company's employee retention and satisfaction?
31. How strong is the company's research and development capabilities?
32. What is the company's capital expenditure efficiency?
33. How strong is the company's acquisition integration track record?
34. What is the company's dividend sustainability?
35. How strong is the company's share buyback program effectiveness?
36. What is the company's debt management quality?
37. How strong is the company's liquidity position?
38. What is the company's working capital management efficiency?
39. How strong is the company's inventory management?
40. What is the company's accounts receivable quality?
41. How strong is the company's cost control measures?
42. What is the company's operating leverage potential?
43. How strong is the company's seasonal performance stability?
44. What is the company's geographic revenue concentration risk?
45. How strong is the company's customer concentration risk management?
46. What is the company's product lifecycle management?
47. How strong is the company's intellectual property portfolio?
48. What is the company's regulatory compliance track record?
49. How strong is the company's crisis management capabilities?
50. What is the overall investment attractiveness for short-term (< 1 month) thesis?

Please provide your response in the following JSON format:
"""
        
        # Add the JSON example separately to avoid f-string formatting conflicts
        prompt += """{
    "Response": [
        {
            "ticker": "TICKER_SYMBOL",
            "scores": [9, 8, 7, 6, 8, 9, 7, 8, 6, 9, 8, 7, 9, 8, 6, 7, 8, 9, 8, 7, 6, 9, 8, 7, 8, 9, 7, 6, 8, 9, 7, 8, 6, 9, 8, 7, 9, 8, 6, 7, 8, 9, 8, 7, 6, 9, 8, 7, 8, 9]
        }
    ]
}

"""
        
        return prompt
    
    def get_allocation_decision(self, broker_reports: List[Dict], earnings_calls: List[Dict], batch_num: int = 1) -> Dict:
        """Get allocation decision from GPT-4o-mini."""
        if not broker_reports and not earnings_calls:
            return {
                "Response": []
            }
        
        prompt = self.create_allocation_prompt(broker_reports, earnings_calls, batch_num)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert portfolio manager. Provide investment analysis in the requested JSON format with 50 numerical scores (0-10) for each ticker."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=8000
            )
            
            # Parse the JSON response
            response_text = response.choices[0].message.content.strip()
            
            # Log the raw response for debugging
            logger.info(f"Raw LLM response length: {len(response_text)}")
            logger.debug(f"Raw LLM response: {response_text[:1000]}...")  # Log first 1000 chars
            
            # Try to extract JSON from the response
            # First, check if response is wrapped in markdown code blocks
            if "```json" in response_text:
                # Extract JSON from markdown code blocks
                start_marker = "```json"
                end_marker = "```"
                start_idx = response_text.find(start_marker) + len(start_marker)
                end_idx = response_text.find(end_marker, start_idx)
                if start_idx != -1 and end_idx != -1:
                    json_str = response_text[start_idx:end_idx].strip()
                    logger.info("Extracted JSON from markdown code blocks")
                else:
                    logger.warning("Found ```json marker but couldn't find closing ```, response appears incomplete")
                    return {"error": "Incomplete response - missing closing markdown marker", "raw_response": response_text}
            else:
                # Fallback to brace extraction
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx != -1 and end_idx != -1:
                    json_str = response_text[start_idx:end_idx]
                    logger.info("Extracted JSON from brace detection")
                else:
                    logger.error("Could not find valid JSON in response")
                    return {"error": "Invalid response format", "raw_response": response_text}
            
            # Validate that the JSON appears complete
            if not self.is_complete_json(json_str):
                logger.warning("Detected incomplete JSON response, discarding")
                return {"error": "Incomplete JSON response detected", "raw_response": response_text}
            
            logger.debug(f"Extracted JSON string length: {len(json_str)}")
            logger.debug(f"Extracted JSON: {json_str[:500]}...")  # Log first 500 chars of JSON
            
            try:
                allocation_decision = json.loads(json_str)
                logger.info("Successfully parsed JSON response")
                return allocation_decision
            except json.JSONDecodeError as e:
                logger.warning(f"Initial JSON parsing failed: {e}")
                logger.warning(f"Attempting to fix malformed JSON...")
                
                # Try to fix common JSON issues
                fixed_json = self.fix_malformed_json(json_str, e)
                if fixed_json:
                    try:
                        allocation_decision = json.loads(fixed_json)
                        logger.info("Successfully parsed JSON after fixing")
                        return allocation_decision
                    except json.JSONDecodeError as e2:
                        logger.error(f"JSON still invalid after fixing: {e2}")
                
                logger.error(f"JSON parsing error: {e}")
                logger.error(f"Problematic JSON around character {e.pos}: {json_str[max(0, e.pos-50):e.pos+50]}")
                return {"error": f"JSON parsing error: {str(e)}", "raw_response": response_text}
                
        except Exception as e:
            logger.error(f"Error getting allocation decision: {e}")
            return {"error": str(e)}

    def fix_malformed_json(self, json_str: str, error: json.JSONDecodeError) -> Optional[str]:
        """Attempt to fix common JSON malformation issues."""
        try:
            logger.info(f"Attempting to fix JSON error at position {error.pos}")
            
            # Make a copy to work with
            fixed = json_str
            
            # Common fix 1: Remove trailing commas in arrays/objects
            fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
            
            # Common fix 2: Fix incomplete arrays (missing closing bracket)
            if error.msg and "Expecting ',' delimiter" in error.msg:
                # Look for incomplete arrays near the error position
                pos = error.pos
                
                # Try to find the incomplete array and close it
                before_error = fixed[:pos]
                after_error = fixed[pos:]
                
                # Count opening and closing brackets
                open_brackets = before_error.count('[') - before_error.count(']')
                open_braces = before_error.count('{') - before_error.count('}')
                
                # If we have unclosed arrays, try to close them
                if open_brackets > 0:
                    # Find the last complete number in the array
                    # Look for pattern like: "6, 5, 6, 7, 6, 6, 6, 5, 6, 7, 6, 6, 7"
                    match = re.search(r',\s*(\d+)(?!\s*[,}\]])(?=\s*$|[^0-9,\s])', before_error)
                    if match:
                        # Add closing bracket after the last number
                        insert_pos = match.end()
                        fixed = before_error[:insert_pos] + ']' * open_brackets + after_error
                        logger.info(f"Added {open_brackets} closing brackets")
                
                # Similarly for unclosed objects
                if open_braces > 0:
                    fixed = fixed + '}' * open_braces
                    logger.info(f"Added {open_braces} closing braces")
            
            # Common fix 3: Ensure proper array structure
            # Fix incomplete ticker entries
            fixed = re.sub(r'"ticker":\s*"([^"]+)"[^}]*$', r'"ticker": "\1", "scores": []}', fixed, flags=re.MULTILINE)
            
            # Common fix 4: Fix numbers outside of proper JSON structure
            # Remove any trailing numbers/commas that aren't in proper JSON structure
            fixed = re.sub(r'}\s*,\s*[\d,\s]+$', '}', fixed)
            
            # Common fix 5: Ensure Response array is properly closed
            if '"Response":' in fixed and not re.search(r'"Response":\s*\[[^\]]*\]', fixed):
                # Find the Response array start
                response_match = re.search(r'"Response":\s*\[', fixed)
                if response_match:
                    # Find everything after the Response array start
                    after_response = fixed[response_match.end():]
                    # Count braces and brackets to ensure proper closing
                    open_braces = 1  # for the main object
                    open_brackets = 1  # for the Response array
                    
                    # Add proper closing if needed
                    if not after_response.strip().endswith(']}'):
                        if after_response.strip().endswith('}'):
                            fixed = fixed + ']'
                        elif after_response.strip().endswith(']'):
                            fixed = fixed + '}'
                        else:
                            fixed = fixed + ']}'
            
            logger.debug(f"Fixed JSON preview: {fixed[:500]}...")
            return fixed
            
        except Exception as fix_error:
            logger.error(f"Error while trying to fix JSON: {fix_error}")
            return None

    def is_complete_json(self, json_str: str) -> bool:
        """Check if the JSON response appears to be complete."""
        try:
            # Basic structural checks
            if not json_str.strip():
                return False
            
            # Check for incomplete array patterns that indicate truncation
            # Pattern: ends with incomplete number sequences like "7, 8, 6, 7, 7, 8, 7, 6, 6, 8"
            if re.search(r'[,\s]\d+(?:\s*,\s*\d+)*\s*$', json_str):
                logger.warning("Detected incomplete number sequence at end of response")
                return False
            
            # Check for incomplete ticker entries (missing closing bracket/brace)
            if re.search(r'"scores"\s*:\s*\[\s*[\d,\s]*$', json_str):
                logger.warning("Detected incomplete scores array")
                return False
            
            # Check that Response array appears to be properly closed
            if '"Response":' in json_str:
                # Find the Response array and check if it's properly structured
                response_match = re.search(r'"Response"\s*:\s*\[', json_str)
                if response_match:
                    after_response = json_str[response_match.end():]
                    # Should end with proper JSON structure, not hanging numbers
                    if not re.search(r'\]\s*}?\s*$', after_response):
                        logger.warning("Response array does not appear to be properly closed")
                        return False
            
            # Check balanced braces and brackets
            open_braces = json_str.count('{')
            close_braces = json_str.count('}')
            open_brackets = json_str.count('[')
            close_brackets = json_str.count(']')
            
            if open_braces != close_braces or open_brackets != close_brackets:
                logger.warning(f"Unbalanced braces/brackets: {open_braces}/{close_braces} braces, {open_brackets}/{close_brackets} brackets")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking JSON completeness: {e}")
            return False
    
    def process_date(self, target_date: datetime, broker_df: pd.DataFrame, 
                    earnings_df: pd.DataFrame) -> Dict:
        """Process a single date and return allocation decision."""
        broker_reports, earnings_calls = self.match_data_by_date(
            broker_df, earnings_df, target_date
        )
        
        # Batch both broker reports and earnings calls based on the maximum needed
        max_reports_per_batch = 15
        max_earnings_per_batch = 3
        all_responses = []
        
        # Calculate number of batches needed for each data source
        broker_batches_needed = (len(broker_reports) + max_reports_per_batch - 1) // max_reports_per_batch if len(broker_reports) > 0 else 0
        earnings_batches_needed = (len(earnings_calls) + max_earnings_per_batch - 1) // max_earnings_per_batch if len(earnings_calls) > 0 else 0
        
        # Use the maximum number of batches to ensure both data sources are distributed
        total_batches = max(broker_batches_needed, earnings_batches_needed)
        
        if total_batches > 1:
            logger.info(f"Processing {len(broker_reports)} broker reports and {len(earnings_calls)} earnings calls across {total_batches} batches")
            logger.info(f"Broker reports per batch: {max_reports_per_batch}, Earnings calls per batch: {max_earnings_per_batch}")
            
            for batch_num in range(1, total_batches + 1):
                # Get broker reports for this batch
                broker_start_idx = (batch_num - 1) * max_reports_per_batch
                broker_end_idx = min(broker_start_idx + max_reports_per_batch, len(broker_reports))
                batch_reports = broker_reports[broker_start_idx:broker_end_idx]
                
                # Get earnings calls for this batch
                earnings_start_idx = (batch_num - 1) * max_earnings_per_batch
                earnings_end_idx = min(earnings_start_idx + max_earnings_per_batch, len(earnings_calls))
                batch_earnings = earnings_calls[earnings_start_idx:earnings_end_idx]
                
                logger.info(f"Processing batch {batch_num}/{total_batches}: {len(batch_reports)} broker reports, {len(batch_earnings)} earnings calls")
                
                # Skip batch if both data sources are empty
                if len(batch_reports) == 0 and len(batch_earnings) == 0:
                    logger.info(f"Skipping empty batch {batch_num}")
                    continue
                
                allocation_decision = self.get_allocation_decision(
                    batch_reports, batch_earnings, batch_num
                )
                
                if 'Response' in allocation_decision:
                    all_responses.extend(allocation_decision['Response'])
            
            # Combine results from all batches
            final_allocation_decision = {
                "Response": all_responses
            }
        else:
            # Single batch processing
            final_allocation_decision = self.get_allocation_decision(broker_reports, earnings_calls)
        
        result = {
            "date": target_date.strftime('%Y-%m-%d'),
            "broker_reports_count": len(broker_reports),
            "earnings_calls_count": len(earnings_calls),
            "allocation_decision": final_allocation_decision
        }
        
        return result
    
    def append_result_to_file(self, result: Dict, output_path: str):
        """Append a single result to the output file."""
        try:
            # Try to read existing data
            try:
                with open(output_path, 'r') as f:
                    existing_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                existing_data = []
            
            # Append new result
            existing_data.append(result)
            
            # Write back to file
            with open(output_path, 'w') as f:
                json.dump(existing_data, f, indent=2, default=str)
                
            logger.info(f"Appended result for {result['date']} to {output_path}")
            
        except Exception as e:
            logger.error(f"Error appending result to file: {e}")
    
    def run_agent(self, broker_reports_path: str, earnings_calls_path: str, 
                 start_date: str, end_date: str, output_path: str = None, parallel: bool = True, us_tickers_path: str = None):
        """Run the agent for a date range with optional parallel processing."""
        # Load existing extracted ticker data if available
        extracted_ticker_path = 'extracted_tickers.csv'
        if os.path.exists(extracted_ticker_path):
            logger.info(f"Loading existing extracted ticker data from {extracted_ticker_path}")
            self.extracted_ticker_data = self.load_csv_ticker_data(extracted_ticker_path)
        else:
            logger.warning(f"No extracted ticker data found at {extracted_ticker_path}")
            self.extracted_ticker_data = None
        
        # Load datasets and proceed with portfolio analysis
        logger.info("Loading datasets and running portfolio analysis...")
        broker_df, earnings_df = self.load_datasets(broker_reports_path, earnings_calls_path)
        
        # Parse date range
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Generate list of dates to process
        dates_to_process = []
        current_date = start_dt
        while current_date <= end_dt:
            dates_to_process.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)
        
        logger.info(f"Processing {len(dates_to_process)} dates from {start_date} to {end_date}")
        
        if parallel and len(dates_to_process) > 1:
            # Parallel processing
            num_cores = mp.cpu_count()
            logger.info(f"Using parallel processing with {num_cores} cores")
            
            # Convert dataframes to dictionaries for serialization
            broker_df_dict = broker_df.to_dict('records')
            earnings_df_dict = earnings_df.to_dict('records')
            
            # Prepare arguments for each process
            us_tickers_file = us_tickers_path or 'data/open_to_open_returns.csv'
            process_args = [
                (date_str, broker_df_dict, earnings_df_dict, self.api_key, us_tickers_file)
                for date_str in dates_to_process
            ]
            
            results = []
            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                # Submit all tasks
                future_to_date = {
                    executor.submit(process_single_date_50q, args): args[0] 
                    for args in process_args
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_date):
                    date_str = future_to_date[future]
                    try:
                        result = future.result()
                        results.append(result)
                        logger.info(f"Completed processing date: {date_str}")
                        
                        # Continuously append to output file
                        if output_path:
                            self.append_result_to_file(result, output_path)
                            
                    except Exception as e:
                        logger.error(f"Error processing date {date_str}: {e}")
                        # Add error result
                        error_result = {
                            "date": date_str,
                            "broker_reports_count": 0,
                            "earnings_calls_count": 0,
                            "allocation_decision": {"error": str(e)}
                        }
                        results.append(error_result)
                        
                        # Also append error result to file
                        if output_path:
                            self.append_result_to_file(error_result, output_path)
            
            # Sort results by date to maintain chronological order
            results.sort(key=lambda x: x['date'])
        else:
            # Sequential processing
            logger.info("Using sequential processing")
            results = []
            for date_str in dates_to_process:
                logger.info(f"Processing date: {date_str}")
                target_date = datetime.strptime(date_str, '%Y-%m-%d')
                result = self.process_date(target_date, broker_df, earnings_df)
                results.append(result)
                
                # Continuously append to output file
                if output_path:
                    self.append_result_to_file(result, output_path)
        
        # Results are already saved continuously, just log completion
        if output_path:
            logger.info(f"All results continuously saved to {output_path}")
        
        return results


def process_single_date_50q(args):
    """Standalone function for processing a single date - used for multiprocessing."""
    date_str, broker_df_dict, earnings_df_dict, api_key, us_tickers_path = args
    
    # Recreate dataframes from dictionaries
    broker_df = pd.DataFrame(broker_df_dict)
    broker_df['selected_date'] = pd.to_datetime(broker_df['selected_date']).dt.tz_localize(None)
    
    earnings_df = pd.DataFrame(earnings_df_dict)
    earnings_df['date'] = pd.to_datetime(earnings_df['date']).dt.tz_localize(None)
    
    # Create agent instance for this process
    agent = PortfolioAgent50Q(api_key=api_key, us_tickers_path=us_tickers_path)
    
    # Load extracted ticker data in worker process (same logic as in run_agent)
    extracted_ticker_path = 'extracted_tickers.csv'
    if os.path.exists(extracted_ticker_path):
        logger.info(f"Loading existing extracted ticker data from {extracted_ticker_path}")
        agent.extracted_ticker_data = agent.load_csv_ticker_data(extracted_ticker_path)
    else:
        logger.warning(f"No extracted ticker data found at {extracted_ticker_path}")
        agent.extracted_ticker_data = None
    
    # Process the date
    target_date = datetime.strptime(date_str, '%Y-%m-%d')
    result = agent.process_date(target_date, broker_df, earnings_df)
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Portfolio Allocation Agent with 50 Questions')
    parser.add_argument('--credentials', default='credentials.json', 
                       help='Path to credentials JSON file (default: credentials.json)')
    parser.add_argument('--api-key', help='OpenAI API key (alternative to credentials file)')
    parser.add_argument('--broker-reports', default='folder_dates_results.csv', 
                       help='Path to broker reports CSV')
    parser.add_argument('--earnings-calls', default='earnings_calls_2023_2025.csv',
                       help='Path to earnings calls CSV')
    parser.add_argument('--us-tickers', default='open_to_open_returns.csv',
                       help='Path to open_to_open_returns.csv for US tickers filtering')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', default='portfolio_allocations_50q.json',
                       help='Output file path')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel processing')
    
    args = parser.parse_args()
    
    # Initialize agent with credentials file or API key
    if args.api_key:
        agent = PortfolioAgent50Q(api_key=args.api_key, us_tickers_path=args.us_tickers)
    else:
        agent = PortfolioAgent50Q(credentials_path=args.credentials, us_tickers_path=args.us_tickers)
    
    results = agent.run_agent(
        args.broker_reports,
        args.earnings_calls,
        args.start_date,
        args.end_date,
        args.output,
        parallel=not args.no_parallel,
        us_tickers_path=args.us_tickers
    )
    
    print(f"Processed {len(results)} dates. Results saved to {args.output}")


if __name__ == "__main__":
    main()