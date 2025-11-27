#!/usr/bin/env python3
"""
Broker Report Analysis Script
Processes broker reports and generates answers to 50 questions using GPT-4o-mini
"""

import pandas as pd
import numpy as np
import json
import openai
from datetime import datetime
import os
import logging
from typing import List, Dict, Any
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import threading

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BrokerReportProcessor:
    def __init__(self, credentials_path: str, csv_path: str, output_path: str):
        """Initialize the processor with credentials and data path"""
        self.csv_path = csv_path
        self.output_path = output_path
        self.load_credentials(credentials_path)
        self.setup_openai()

        # Initialize output CSV file with headers
        self.initialize_output_csv()

        # The 50 questions for analysis
        self.questions = [
            "On a scale of 0 to 10, how surprising or unexpected was the new information relative to market expectations? (0 = widely anticipated, 10 = completely unexpected)",
            "On a scale of 0 to 10, what is the main driver of the new information? (0 = purely event-driven such as M&A, regulatory, management changes; 10 = purely fundamentally driven, such as earnings, guidance, or margins)",
            "On a scale of 0 to 10, does the report anchor on an event that has already occurred or one that is about to happen? (0 = already occurred, 10 = about to happen)",
            "On a scale of 0 to 10, to what extent does the reporting broker appear to have unique or exclusive access to the event or new information (0 = fully public/consensus knowledge, 10 = unique broker-exclusive)?",
            "On a scale of 0 to 10, how contrarian or differentiated is the perspective relative to broader market commentary? (0 = the same perspective, 10 = complete new angle)",
            "On a scale of 0 to 10, how aligned is the broker's thesis with the prevailing market consensus? (0 = very well aligned, 10 = extreme contrary)",
            "On a scale of 0 to 10, how well does the report quantify the \"surprise\" element relative to previous broker or investor commentary? (0 = not quantified, 10 = very well quantified)",
            "How special or unique is the view point offer by the report (0 = not unique, 10 = very unique)",
            "On a scale of 0 to 10, how immediate is the expected market reaction to the new information? (0 = long-term, 10 = short-term)",
            "What is the analyst's conviction to their proposed investment thesis (0 = low conviction, 10 = high conviction)",
            "On a scale of 0 to 10, how positive or negative is the change in the investment thesis relative to the immediate prior report by the same broker? (0 = extremely negative, 10 = extremely positive)",
            "On a scale of 0 to 10, how positive or negative is the change in the investment thesis relative to reports made by other brokers? (0 = extremely negative, 10 = extremely positive)",
            "On a scale of 0 to 10, how positive or negative is the overall view of the investment thesis? (0 = extremely negative, 10 = extremely positive)",
            "On a scale of 0 to 10, how strongly does the report focus on company level development such as the the release of a new product, or change in the CEO compared with previous reports? (0 = no mention, 10 = significant emphasis)",
            "On a scale of 0–10, how strongly does the report focus on the escalation of a industry wide event compared with previous reports? (0 = not at all, 10 = very strong focus)",
            "On a scale of 0–10, how strongly does the report focus on discussing earnings/financial results compared with previous reports? (0 = not at all, 10 = very strong focus)",
            "On a scale of 0–10, how well does the analyst quantify catalysts with dates (trial readouts, launches, milestones)? (0 = vague/none, 10 = highly precise)",
            "On a scale of 0–10, how strong is the author's conviction behind the thesis of the report? (0 = tentative/hedged, 10 = highly confident)",
            "On a scale of 0–10, how much does the report emphasize the company's long-term strategic direction beyond the current cycle? (0 = not at all, 10 = very strongly)",
            "On a scale of 0–10, how well does the report link near-term catalysts to the long-term investment thesis compared to earlier reports? (0 = no connection, 10 = very strong linkage)",
            "On a scale of 0 to 10, how positive is the revision to forward EPS estimates? (0 = most negative revision, 10 = most positive revision)",
            "On a scale of 0 to 10, how positive is the revision to revenue growth assumptions? (0 = sharply reduced, 10 = sharply raised)",
            "On a scale of 0 to 10, how positive is the revision to valuation assumptions (such as growth rates, discount rates)? (0 = strongly negative, 10 = strongly positive)",
            "On a scale of 0 to 10, how clearly are price target changes justified by valuation drivers, are various valuation approaches considered? (0 = unclear, 10 = very precise)",
            "On a scale of 0 to 10, how strong is the trust in management's guidance? (0 = fully dismissive, 10 = fully trusting)",
            "On a scale of 0 to 10, how sound is the argument in favor of the valuations (passively following the market or an active update)? (0 = passive update on earnings, 10 = active update motivated by analyst views)",
            "On a scale of 0 to 10, are there segment-level forecasts? (0 = no granularity, 10 = highly granular)",
            "On a scale of 0 to 10, how robust is the scenario analysis? (0 = absent, 10 = highly detailed and probability-weighted)",
            "On a scale of 0 to 10, how conservative versus aggressive are revised assumptions relative to peers? (0 = extremely aggressive, 10 = extremely conservative)",
            "On a scale of 0 to 10, how explicit is the attribution of revisions to company-specific versus macro factors? (0 = vague, 10 = very explicit)",
            "In what direction did the analysts adjust forward looking P/E, EV/EBITDA or EV/sales (0 = significantly downwards, 10 = significantly upwards)",
            "On a scale of 0 to 10, how detailed is the discussion of valuation multiples (P/E, EV/EBITDA, EV/sales)? (0 = absent, 10 = very detailed)",
            "On a scale of 0 to 10, how clearly does the report attribute valuation changes to earnings/margin revisions versus multiple re-rating? (0 = unclear, 10 = very clear)",
            "Relative to peers, is the discussed firm trading at a premium or a discount? (0 = premium, 10 = discount)",
            "On a scale of 0 to 10, how well does the report benchmark valuation against peers? (0 = not at all, 10 = extensively benchmarked)",
            "On a scale of 0 to 10, how much does the report emphasize relative valuation vs intrinsic valuation? (0 = entirely intrinsic, 10 = entirely relative)",
            "Compared to other brokers, is the valuation approach used by the report a new and unique approach or is it similar to other brokers? (0 = similar to other brokers, 10 = unique approach)",
            "How justified and sound is the valuation approach used in the report versus other reports? (0 = very unsound, 10 = very sound)",
            "On a scale of 0 to 10, how clearly does the report frame upside/downside risks around valuation? (0 = absent, 10 = very clear)",
            "On a scale of 0 to 10, how much conviction does the analyst show in their valuation framework? (0 = very low conviction, 10 = very high conviction)",
            "On a scale of 0 to 10, how negative is the emphasis on external conditions? (0 = no focus, 10 = very heavy focus)",
            "On a scale of 0 to 10, how surprising / information rich are the headwinds versus prior reports? (0 = identical, 10 = entirely unique)",
            "On a scale of 0 to 10, how positive is the emphasis on company developments? (0 = none, 10 = very strong emphasis)",
            "On a scale of 0 to 10, how surprising / information rich are the tailwinds versus prior reports? (0 = identical, 10 = entirely unique)",
            "On a scale of 0 to 10, how durable is the thesis across cycles? (0 = fragile, 10 = highly robust)",
            "On a scale of 0 to 10, how is the relative importance of headwinds versus tailwinds (0 = headwinds more important, 10 = tailwinds more important)",
            "On a scale of 0 to 10, how balanced is the treatment of headwinds versus tailwinds? (0 = completely one-sided, 10 = evenly balanced)",
            "On a scale of 0 to 10, how positive is the analyst's opinion on liquidity/capital risks? (0 = very negative, 10 = very positive)",
            "On a scale of 0 to 10, how positive is the emphasis on operational risks such as supply chain or execution? (0 = very negative, 10 = very positive)",
            "On a scale of 0 to 10, how positive is the emphasis on regulatory or geopolitical risks? (0 = very negative, 10 = very positive)"
        ]

    def load_credentials(self, credentials_path: str):
        """Load API credentials from JSON file"""
        try:
            with open(credentials_path, 'r') as f:
                credentials = json.load(f)
            self.openai_api_key = credentials.get('openai_api_key')
            if not self.openai_api_key:
                raise ValueError("OpenAI API key not found in credentials file")
            logger.info("Credentials loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load credentials: {e}")
            raise

    def setup_openai(self):
        """Set up OpenAI client"""
        openai.api_key = self.openai_api_key
        logger.info("OpenAI client initialized")

    def initialize_output_csv(self):
        """Initialize the output CSV with headers"""
        import os
        if os.path.exists(self.output_path):
            logger.info(f"Output file {self.output_path} already exists, will append to it")
        else:
            # Create header row
            headers = ['ticker', 'firm', 'current_file', 'current_directory', 'current_date',
                      'current_et_timestamp', 'previous_file', 'previous_directory', 'previous_date',
                      'previous_et_timestamp', 'llm_summary', 'processing_timestamp']

            # Add 50 question columns
            for i in range(1, 51):
                headers.append(f'q{i}_score')

            import csv
            with open(self.output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                writer.writerow(headers)
            logger.info(f"Initialized output CSV at {self.output_path}")

    def save_single_result(self, result_row: Dict):
        """Save a single result row to CSV incrementally"""
        import csv
        from datetime import datetime

        # Update processing timestamp (don't add duplicate)
        result_row['processing_timestamp'] = datetime.now().isoformat()

        # Define the expected column order
        base_columns = ['ticker', 'firm', 'current_file', 'current_directory', 'current_date',
                       'current_et_timestamp', 'previous_file', 'previous_directory',
                       'previous_date', 'previous_et_timestamp', 'llm_summary', 'processing_timestamp']
        score_columns = [f'q{i+1}_score' for i in range(50)]
        all_columns = base_columns + score_columns

        # Write to CSV file with proper quoting and consistent column order
        with open(self.output_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_columns, quoting=csv.QUOTE_ALL)
            writer.writerow(result_row)

        logger.info(f"Saved result for {result_row['ticker']} - {result_row['firm']} to CSV")

    def load_data(self) -> pd.DataFrame:
        """Load the CSV data"""
        logger.info(f"Loading data from {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        logger.info(f"Loaded {len(df)} records")

        # Sort by timestamp first for consistent ordering
        df['current_et_timestamp_dt'] = pd.to_datetime(df['current_et_timestamp'], errors='coerce')
        df = df.sort_values('current_et_timestamp_dt', ascending=True).reset_index(drop=True)
        logger.info(f"Sorted data by timestamp")

        return df

    def find_previous_reports(self, df: pd.DataFrame, current_row: pd.Series, limit: int = 10) -> List[Dict]:
        """Find ALL previous reports for the same ticker from ANY broker before the current timestamp"""
        ticker = current_row['ticker']
        current_timestamp = pd.to_datetime(current_row['current_et_timestamp'])

        # Find all reports for this ticker that are before the current timestamp
        previous_reports = df[
            (df['ticker'] == ticker) &
            (pd.to_datetime(df['current_et_timestamp']) < current_timestamp) &
            (df['llm_summary'].notna()) & (df['llm_summary'] != '')  # Only reports with content
        ].copy()

        if len(previous_reports) == 0:
            return []

        # Sort by timestamp ascending (furthest to closest to current report)
        previous_reports = previous_reports.sort_values('current_et_timestamp', ascending=True)

        # Convert to list of dictionaries and return up to limit reports
        result = []
        for _, row in previous_reports.head(limit).iterrows():
            result.append({
                'ticker': row['ticker'],
                'firm': row['firm'],
                'current_file': row['current_file'],
                'current_directory': row['current_directory'],
                'current_date': row['current_date'],
                'current_et_timestamp': row['current_et_timestamp'],
                'llm_summary': row.get('llm_summary', '')
            })

        return result


    def generate_gpt_prompt(self, ticker: str, current_content: str, current_date: str,
                          current_firm: str, previous_reports: List[Dict]) -> str:
        """Generate the GPT prompt for analysis"""

        questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(self.questions)])

        # Build previous reports section
        previous_reports_text = ""
        if previous_reports:
            for i, prev_report in enumerate(previous_reports[:10], 1):  # Limit to 10 reports
                previous_reports_text += f"""
PREVIOUS REPORT {i}:
Timestamp: {prev_report['current_et_timestamp']}
Published by firm: {prev_report['firm']}
Content: {prev_report.get('llm_summary', '')}
"""
        else:
            previous_reports_text = "\nNo previous reports available for comparison.\n"

        # Adjust the instruction based on whether previous reports exist
        if previous_reports:
            instruction = f"Answer the 50 questions on notes produced on current report about {ticker}, with reference to previous reports on the same ticker"
        else:
            instruction = f"Answer the 50 questions on notes produced on current report about {ticker}. Since no previous reports are available, base your analysis on the absolute characteristics of this report and use your best judgment for comparative questions."

        prompt = f"""You are a portfolio manager reading equity research transcripts.
{instruction}

{previous_reports_text}

CURRENT REPORT:
Timestamp: {current_date}
Published by firm: {current_firm}
Content: {current_content}

Questions:
{questions_text}

Please provide your answers as a numbered list with only the numerical score (0-10) for each question. Format your response as:
1. [score]
2. [score]
...
50. [score]

Only provide numerical scores, no explanations."""

        return prompt

    def call_gpt4o_mini(self, prompt: str) -> str:
        """Call GPT-4o-mini API"""
        try:
            client = openai.OpenAI(api_key=self.openai_api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=1000
            )

            response_text = response.choices[0].message.content.strip()
            return response_text

        except Exception as e:
            logger.error(f"GPT API call failed: {e}")
            return ""

    def parse_gpt_response(self, response: str) -> List[float]:
        """Parse GPT response to extract numerical answers"""
        answers = []

        # Look for numbered patterns like "1. 5" or "1. 5.0"
        pattern = r'(\d+)\.\s*([0-9]+\.?[0-9]*)'
        matches = re.findall(pattern, response)

        for question_num, score in matches:
            try:
                score_float = float(score)
                # Clamp to 0-10 range
                score_float = max(0, min(10, score_float))
                answers.append(score_float)
            except ValueError:
                answers.append(5.0)  # Default middle score if parsing fails

        # Ensure we have exactly 50 answers
        while len(answers) < 50:
            answers.append(5.0)  # Default scores for missing answers

        return answers[:50]  # Take only first 50 in case of extras

    def process_single_report(self, args):
        """Process a single report - designed for parallel execution"""
        if len(args) == 4:
            idx, row, full_df, processor = args
        else:
            idx, row, full_df = args
            processor = self

        try:
            logger.info(f"Processing report {idx+1}: {row['ticker']} - {row['firm']}")

            # Find previous reports
            previous_reports = self.find_previous_reports(full_df, row)

            # Use the llm_summary column as content
            current_content = row.get('llm_summary', '')

            if not current_content:
                logger.warning(f"No LLM summary available for current {row['ticker']} report")
                answers = [5.0] * 50
            else:
                if not previous_reports:
                    logger.info(f"No previous reports found for {row['ticker']}, analyzing current report only")

                # Generate GPT prompt (works with or without previous reports)
                prompt = self.generate_gpt_prompt(
                    ticker=row['ticker'],
                    current_content=current_content,
                    current_date=row['current_et_timestamp'],  # Use full timestamp instead of just date
                    current_firm=row['firm'],
                    previous_reports=previous_reports
                )

                # Call GPT API
                response = self.call_gpt4o_mini(prompt)

                if response:
                    answers = self.parse_gpt_response(response)
                else:
                    answers = [5.0] * 50

            # Create result row, excluding temporary columns
            result_row = row.to_dict()
            # Remove the temporary datetime column if it exists
            if 'current_et_timestamp_dt' in result_row:
                del result_row['current_et_timestamp_dt']

            # Add the 50 answer columns
            for i, answer in enumerate(answers):
                result_row[f'q{i+1}_score'] = answer

            # Save result incrementally
            processor.save_single_result(result_row)

            return result_row

        except Exception as e:
            logger.error(f"Error processing report {idx+1}: {e}")
            # Return row with NaN scores on error
            result_row = row.to_dict()
            for i in range(50):
                result_row[f'q{i+1}_score'] = np.nan
            # Save error result incrementally
            processor.save_single_result(result_row)
            return result_row

    def process_reports(self, limit: int = 10) -> pd.DataFrame:
        """Process reports and generate analysis using parallel processing"""
        df = self.load_data()

        # Limit processing if specified
        if limit:
            df = df.head(limit)
            logger.info(f"Processing limited to {limit} reports")

        # Prepare arguments for parallel processing
        processing_args = [(idx, row, df) for idx, row in df.iterrows()]

        # Determine number of workers (use all available cores)
        max_workers = cpu_count()
        logger.info(f"Using {max_workers} parallel workers")

        results = []

        # Process reports in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(self.process_single_report, args): args[0]
                for args in processing_args
            }

            # Collect results as they complete
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completed processing report {idx+1}")
                except Exception as e:
                    logger.error(f"Error in parallel processing for report {idx+1}: {e}")
                    # Add result with NaN scores on error
                    row = df.iloc[idx]
                    result_row = row.to_dict()
                    for i in range(50):
                        result_row[f'q{i+1}_score'] = np.nan
                    results.append(result_row)

        # For small datasets, sorting is not critical
        # Results are already in the order they completed

        return pd.DataFrame(results)

    def save_results(self, df: pd.DataFrame, output_path: str):
        """Save results to CSV"""
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")

def main():
    """Main execution function"""
    output_path = "/home/lichenhui/broker_report_analysis_results.csv"

    processor = BrokerReportProcessor(
        credentials_path="/csv_mount/home/lichenhui/credentials.json",
        csv_path="/csv_mount/home/lichenhui/broker_report_comparisons_streaming.csv",
        output_path=output_path
    )

    # Process all reports
    completed_count = processor.process_reports(limit=None)

    logger.info(f"Processing completed successfully. Processed {completed_count} reports.")

if __name__ == "__main__":
    main()
