import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
import os
import logging
from typing import List, Dict, Optional, Tuple
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EarningsAnalysisAgent:
    def __init__(self, credentials_file: str = '/home/lichenhui/credentials.json', max_workers: int = 10):
        """
        Initialize the earnings analysis agent

        Args:
            credentials_file: Path to credentials JSON file containing OpenAI API key
            max_workers: Number of parallel workers for processing
        """
        # Load API key from credentials file
        api_key = self.load_api_key(credentials_file)
        self.client = openai.OpenAI(api_key=api_key)
        self.max_workers = max_workers

        # Load data
        self.earnings_df = None
        self.broker_reports_df = None
        self.load_data()

    def load_api_key(self, credentials_file: str) -> str:
        """Load OpenAI API key from credentials JSON file"""
        try:
            with open(credentials_file, 'r') as f:
                credentials = json.load(f)
                api_key = credentials.get('openai_api_key')
                if not api_key:
                    raise ValueError("openai_api_key not found in credentials file")
                logger.info("Successfully loaded OpenAI API key from credentials.json")
                return api_key
        except Exception as e:
            logger.error(f"Error loading API key from {credentials_file}: {e}")
            raise

    def load_data(self):
        """Load and prepare the datasets"""
        try:
            # Load earnings calls data
            self.earnings_df = pd.read_csv('/home/lichenhui/csv/earnings_calls_2023_2025.csv')
            self.earnings_df['date'] = pd.to_datetime(self.earnings_df['date'])
            self.earnings_df = self.earnings_df.sort_values(['ticker', 'date'])

            # Load broker reports data
            self.broker_reports_df = pd.read_csv('/home/lichenhui/csv/timestamps_tickers_joined.csv')
            self.broker_reports_df['report_timestamp_et'] = pd.to_datetime(self.broker_reports_df['report_timestamp_et'])

            # Filter for earnings calls from 2023 onwards
            self.earnings_df = self.earnings_df[self.earnings_df['date'] >= '2023-01-01']

            logger.info(f"Loaded {len(self.earnings_df)} earnings calls from {len(self.earnings_df['ticker'].unique())} unique tickers")
            logger.info(f"Loaded {len(self.broker_reports_df)} broker reports")

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def get_historical_calls(self, ticker: str, current_date: datetime, max_calls: int = 4) -> List[Dict]:
        """
        Get historical earnings calls for a ticker before the current date

        Args:
            ticker: Stock ticker
            current_date: Current earnings call date
            max_calls: Maximum number of historical calls to retrieve (1-4)

        Returns:
            List of historical earnings call data
        """
        ticker_calls = self.earnings_df[
            (self.earnings_df['ticker'] == ticker) &
            (self.earnings_df['date'] < current_date)
        ].sort_values('date', ascending=False)

        # Get at least 1, up to max_calls historical calls
        historical_calls = ticker_calls.head(max_calls).to_dict('records')

        if len(historical_calls) == 0:
            logger.warning(f"No historical calls found for {ticker} before {current_date}")

        return historical_calls

    def get_broker_reports(self, ticker: str, current_date: datetime, days_window: int = 5) -> List[Dict]:
        """
        Get broker reports for a ticker within the specified window before the current date

        Args:
            ticker: Stock ticker
            current_date: Current earnings call date
            days_window: Number of days before current_date to look for reports

        Returns:
            List of relevant broker reports
        """
        # Convert to ET timezone if not already
        if current_date.tzinfo is None:
            et_tz = pytz.timezone('US/Eastern')
            current_date = et_tz.localize(current_date)

        start_date = current_date - timedelta(days=days_window)

        # Filter broker reports
        relevant_reports = self.broker_reports_df[
            (self.broker_reports_df['ticker'] == ticker) &
            (self.broker_reports_df['report_timestamp_et'] >= start_date) &
            (self.broker_reports_df['report_timestamp_et'] <= current_date)
        ].sort_values('report_timestamp_et', ascending=False)

        return relevant_reports.to_dict('records')

    def load_file_content(self, file_path: str) -> str:
        """Load content from a file path"""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                logger.warning(f"File not found: {file_path}")
                return ""
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return ""

    def analyze_earnings_call(self, current_call: Dict, historical_calls: List[Dict], broker_reports: List[Dict]) -> str:
        """
        Analyze an earnings call using GPT-4o-mini

        Args:
            current_call: Current earnings call data
            historical_calls: Historical earnings calls
            broker_reports: Recent broker reports

        Returns:
            Analysis result as string
        """
        # Prepare historical calls text
        historical_text = ""
        for i, call in enumerate(historical_calls):
            historical_text += f"Historical Call {i+1} ({call['quarter']} {call['date'].strftime('%Y-%m-%d')}):\n"
            historical_text += f"{call['transcript'][:2000]}...\n\n"  # Truncate to manage token limits

        # Prepare broker reports text
        broker_text = ""
        for i, report in enumerate(broker_reports):
            content = self.load_file_content(report.get('full_file_path', ''))
            if content:
                broker_text += f"Broker Report {i+1} ({report['report_timestamp_et']}):\n"
                broker_text += f"{content[:1500]}...\n\n"  # Truncate to manage token limits

        # Create the analysis prompt
        prompt = f"""You are a portfolio manager tasked with evaluating a company's current earnings call transcript against its history transcripts. You also have access to broker reports published prior to the call.

The previous history of calls are:
{historical_text if historical_text else "No historical calls available"}

Recent views of equity research analysts are:
{broker_text if broker_text else "No recent broker reports available"}

The current earnings call is:
Ticker: {current_call['ticker']}
Quarter: {current_call['quarter']}
Date: {current_call['date'].strftime('%Y-%m-%d')}
Transcript: {current_call['transcript'][:3000]}...

Your objective is to determine how much the call deviates from prior expectations and to assess the likely market response. Produce a structured analysis that integrates both quantitative and qualitative insights across the following dimensions:
1. Element of Surprise : Assess how positively or negatively surprising the call is relative to prior broker expectations and historical guidance.
2. Forward-Looking Statements & Delivery : Evaluate credibility using the firm's past track record on meeting guidance and following through on promises.
3. Financial Performance: Summarise how key financial health indicators (eg. Revenue, EBITDA differ from previous quarters)
4. Key Events: Identify material strategic initiatives (M&A, partnerships, divestitures, new product launches, regulatory developments) and macroeconomic
5. Narrative & Language Quality , Q&A Integrity: Evaluate directness of answers, evasiveness, and transparency.  Evaluate concreteness, readability, tone shifts, and consistency of terminology compared to past calls.

Output a detailed evaluation where your response to each point is in a different paragraph. Your response should have 5 paragraphs in total. Do not reiterate the questions. Keep your response under 2000 tokens."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert financial analyst with deep experience in earnings analysis and market assessment."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.3
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error in GPT-4o-mini analysis for {current_call['ticker']}: {e}")
            return f"Error analyzing {current_call['ticker']}: {str(e)}"

    def process_single_call(self, call_data: Dict) -> Dict:
        """
        Process a single earnings call

        Args:
            call_data: Single earnings call record

        Returns:
            Analysis result dictionary
        """
        ticker = call_data['ticker']
        call_date = call_data['date']

        logger.info(f"Processing {ticker} earnings call from {call_date}")

        # Get historical calls (1-4 calls)
        historical_calls = self.get_historical_calls(ticker, call_date, max_calls=4)

        # Get broker reports (-5 days window)
        broker_reports = self.get_broker_reports(ticker, call_date, days_window=5)

        # Perform analysis
        analysis = self.analyze_earnings_call(call_data, historical_calls, broker_reports)

        return {
            'ticker': ticker,
            'quarter': call_data['quarter'],
            'date': call_date.strftime('%Y-%m-%d'),
            'historical_calls_count': len(historical_calls),
            'broker_reports_count': len(broker_reports),
            'analysis': analysis
        }

    def run_parallel_analysis(self, sample_size: Optional[int] = None) -> List[Dict]:
        """
        Run parallel analysis on all earnings calls from 2023 onwards

        Args:
            sample_size: Optional limit on number of calls to process (for testing)

        Returns:
            List of analysis results
        """
        # Get calls from 2023 onwards, sorted chronologically
        calls_to_process = self.earnings_df.copy()

        if sample_size:
            calls_to_process = calls_to_process.head(sample_size)
            logger.info(f"Processing sample of {len(calls_to_process)} calls")
        else:
            logger.info(f"Processing all {len(calls_to_process)} calls")

        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_call = {
                executor.submit(self.process_single_call, call.to_dict()): call.to_dict()
                for _, call in calls_to_process.iterrows()
            }

            # Process completed tasks
            for future in as_completed(future_to_call):
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completed analysis for {result['ticker']} ({result['date']})")
                except Exception as e:
                    call_data = future_to_call[future]
                    logger.error(f"Error processing {call_data['ticker']}: {e}")

        # Sort results chronologically
        results.sort(key=lambda x: x['date'])

        return results

    def save_results(self, results: List[Dict], output_file: str):
        """Save analysis results to CSV"""
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(results)} analysis results to {output_file}")

def main():
    """Main execution function"""
    # Initialize the agent (loads API key from credentials.json)
    agent = EarningsAnalysisAgent(max_workers=5)

    # Run analysis on a sample first (for testing)
    results = agent.run_parallel_analysis(sample_size=10)

    # Save results
    agent.save_results(results, '/home/lichenhui/csv/earnings_analysis_results.csv')

    # Print summary
    print(f"\n=== Analysis Complete ===")
    print(f"Processed {len(results)} earnings calls")
    print(f"Unique tickers: {len(set(r['ticker'] for r in results))}")
    print(f"Average historical calls per analysis: {np.mean([r['historical_calls_count'] for r in results]):.1f}")
    print(f"Average broker reports per analysis: {np.mean([r['broker_reports_count'] for r in results]):.1f}")

    # Show sample result
    if results:
        print(f"\n=== Sample Analysis ===")
        sample = results[0]
        print(f"Ticker: {sample['ticker']}")
        print(f"Quarter: {sample['quarter']}")
        print(f"Date: {sample['date']}")
        print(f"Analysis:\n{sample['analysis'][:500]}...")

if __name__ == "__main__":
    main()
