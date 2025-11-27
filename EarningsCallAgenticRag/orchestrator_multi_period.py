# This is a multi-period orchestrator for earnings call analysis.
import os
import json
import re
import pandas as pd
import concurrent.futures
from typing import Dict, List
from datetime import datetime
from pathlib import Path
import math
import threading
import time
from openai import OpenAI
from agents.prompts.prompts import main_agent_prompt

# ---------- Constants & paths ---------------------
DATA_FILE = "EarningsFilteredResults2.csv"
LOG_PATH = "FinalResultsQoQ.csv"
TIMEOUT_SEC = 1000
MAX_WORKERS = 10
CHUNK_SIZE = 300
TOKEN_LOG_DIR = "token_logs"
TIMING_LOG_DIR = "timing_logs"
STATEMENT_BASE_DIR = Path("financial_statements")
STATEMENT_FILES = {
    "cash_flow_statement": "_cash_flow_statement.csv",
    "income_statement": "_income_statement.csv",
    "balance_sheet": "_balance_sheet.csv",
}
KEY_METRICS = [
    "Cash and cash equivalents", "Accounts receivable", "Inventory", "Property, plant and equipment",
    "Short-term debt", "Total current liabilities", "Total liabilities", "Total Shareholders' Equity",
    "Main business income", "Operating Costs", "Net profit", "Gross profit",
    "Net Cash Flow from Operating Activities", "Net cash flow from investing activities", "Net Cash Flow from Financing Activities",
    "Diluted earnings per share-Common stock"
]

# How many historical periods to include
NUM_HISTORICAL_PERIODS = 4  # Current + 3 previous quarters

METRIC_MAPPING = {
    "Main business income": "Revenue",
    "主营收入": "Revenue",
    "Main Business Income": "Revenue",
    "MAIN BUSINESS INCOME": "Revenue",
}
openai_semaphore = threading.Semaphore(4)

def _q_sort_key(q: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d{4})-Q([1-4])", q)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

def get_historical_quarters(current_quarter: str, num_periods: int = NUM_HISTORICAL_PERIODS) -> List[str]:
    """Get list of quarters including current and previous periods."""
    current_q_key = _q_sort_key(current_quarter)
    quarters = [current_quarter]

    year, quarter = current_q_key
    for i in range(1, num_periods):
        quarter -= 1
        if quarter < 1:
            quarter = 4
            year -= 1
        prev_quarter = f"{year}-Q{quarter}"
        quarters.append(prev_quarter)

    return quarters

def load_latest_statements(ticker: str, as_of_date: pd.Timestamp, n: int = 6) -> Dict[str, List[dict]]:
    out = {}
    for key, suffix in STATEMENT_FILES.items():
        fname = STATEMENT_BASE_DIR / f"{ticker.upper()}{suffix}"
        if not fname.exists():
            out[key] = []
            continue
        df = pd.read_csv(fname, index_col=0)
        valid_cols = []
        for col in df.columns:
            try:
                d = pd.to_datetime(col.split(".")[0], format="%Y-%m-%d", errors="raise")
                if d < as_of_date:
                    valid_cols.append((d, col))
            except Exception:
                continue
        valid_cols = sorted(valid_cols, reverse=True)[:n]
        out[key] = [
            {"date": d.strftime("%Y-%m-%d"), "rows": df[c].dropna().to_dict()}
            for d, c in valid_cols
        ]
    return out

def extract_number_with_unit(val):
    val_str = str(val).replace(',', '')
    num_match = re.search(r'-?\d+(?:\.\d+)?', val_str)
    if not num_match:
        return None, None
    number = float(num_match.group(0))
    unit_start = num_match.end()
    unit = val_str[unit_start:].strip()
    return number, unit

def map_metric_name(metric_name: str) -> str:
    return METRIC_MAPPING.get(metric_name, metric_name)

def generate_financial_statement_facts(row: pd.Series, ticker: str, quarter: str) -> List[Dict]:
    as_of_date_str = row.get("parsed_date")
    if pd.isna(as_of_date_str):
        as_of_date = pd.Timestamp.now().tz_localize(None)
    else:
        as_of_date = pd.to_datetime(as_of_date_str).tz_localize(None)
    stmts = load_latest_statements(ticker, as_of_date, n=100)

    def parse_report_type_to_quarter(report_type: str) -> str:
        """
        Enhanced quarter parsing to handle ambiguous formatting and structured layouts.
        Handles various report formats including Q1, Q2, Q3, Q4, annual, semi-annual reports.
        """
        try:
            report_type_lower = report_type.lower().strip()

            # Pattern 1: Direct Q1, Q2, Q3, Q4 format (e.g., "2023/Q1", "2023-Q1", "2023 Q1")
            q_pattern = re.search(r'(\d{4})[\/\-\s]*(Q[1-4])', report_type, re.IGNORECASE)
            if q_pattern:
                year, quarter = q_pattern.groups()
                result = f"{year}-{quarter.upper()}"
                return result

            # Pattern 2: Year with quarter number (e.g., "2023/1", "2023-1", "2023 1")
            q_num_pattern = re.search(r'(\d{4})[\/\-\s]*([1-4])(?:\s|$)', report_type)
            if q_num_pattern:
                year, quarter_num = q_num_pattern.groups()
                result = f"{year}-Q{quarter_num}"
                return result

            # Pattern 3: Year with text descriptions
            if '/' in report_type:
                parts = report_type.split('/')
                if len(parts) >= 2:
                    year = parts[0].strip()
                    period_part = parts[1].split()[0].lower() if parts[1].split() else ""

                    # Handle various period descriptions
                    if 'annual' in period_part or 'year' in period_part:
                        result = f"{year}-Q4"
                        return result
                    elif 'Semi' in period_part or 'half' in period_part:
                        result = f"{year}-Q2"
                        return result
                    elif 'first' in period_part or '1' in period_part:
                        result = f"{year}-Q1"
                        return result
                    elif 'second' in period_part or '2' in period_part:
                        result = f"{year}-Q2"
                        return result
                    elif 'third' in period_part or '3' in period_part:
                        result = f"{year}-Q3"
                        return result
                    elif 'fourth' in period_part or '4' in period_part:
                        result = f"{year}-Q4"
                        return result

            # Pattern 4: Structured layout patterns (like in the image)
            # Look for patterns that might indicate structured quarterly reports
            structured_patterns = [
                r'(\d{4})[\/\-\s]*(?:quarter|qtr|q)[\s\-]*([1-4])',
                r'(\d{4})[\/\-\s]*(?:period|p)[\s\-]*([1-4])',
                r'(\d{4})[\/\-\s]*(?:report|rpt)[\s\-]*([1-4])',
            ]

            for pattern in structured_patterns:
                match = re.search(pattern, report_type, re.IGNORECASE)
                if match:
                    year, quarter_num = match.groups()
                    result = f"{year}-Q{quarter_num}"
                    return result

            # Pattern 5: Chinese/alternative formats
            chinese_patterns = [
                r'(\d{4})年[第\s]*([1-4])季度',
                r'(\d{4})[\/\-\s]*(?:第[1-4]季度)',
            ]

            for pattern in chinese_patterns:
                match = re.search(pattern, report_type)
                if match:
                    year, quarter_num = match.groups()
                    result = f"{year}-Q{quarter_num}"
                    return result

            # Fallback: try to extract year and make educated guess
            year_match = re.search(r'(\d{4})', report_type)
            if year_match:
                year = year_match.group(1)
                # If we can't determine quarter, assume Q4 (annual report)
                result = f"{year}-Q4"
                return result

            print(f"   ❌ Could not parse report type: '{report_type}', using fallback quarter: {quarter}")
            return quarter

        except Exception as e:
            print(f"   ❌ Error parsing report type '{report_type}': {e}, using fallback quarter: {quarter}")
            return quarter

    def process_financial_data(data: List[dict], statement_type: str) -> List[Dict]:
        if not data:
            return []
        facts = []
        current_quarter_key = _q_sort_key(quarter)

        for period_data in data:
            if isinstance(period_data, dict) and 'rows' in period_data:
                rows = period_data.get('rows', {})
                report_type = rows.get('Financial Report Type', '')
                statement_quarter = parse_report_type_to_quarter(report_type) if report_type else quarter
                statement_quarter_key = _q_sort_key(statement_quarter)
                if statement_quarter_key > current_quarter_key:
                    continue
                for metric, value in rows.items():
                    mapped_metric = map_metric_name(metric)
                    if mapped_metric in KEY_METRICS and value != '--':
                        fact = {
                            "ticker": ticker,
                            "quarter": statement_quarter,
                            "type": "Result",
                            "metric": f"{metric}",
                            "value": str(value),
                            "reason": f"{statement_type} data from {report_type}"
                        }
                        facts.append(fact)
        return facts

    cash_flow_facts = process_financial_data(stmts["cash_flow_statement"], 'CashFlow')
    income_facts = process_financial_data(stmts["income_statement"], 'Income')
    balance_facts = process_financial_data(stmts["balance_sheet"], 'Balance')
    return cash_flow_facts + income_facts + balance_facts

def check_financial_statement_files_exist(ticker: str) -> bool:
    for suffix in STATEMENT_FILES.values():
        fname = STATEMENT_BASE_DIR / f"{ticker.upper()}{suffix}"
        if fname.exists():
            return True
    return False

def ensure_log_directories():
    os.makedirs(TOKEN_LOG_DIR, exist_ok=True)
    os.makedirs(TIMING_LOG_DIR, exist_ok=True)

def initialize_log_file() -> None:
    print(f"Initializing fresh log file at: {LOG_PATH}")
    pd.DataFrame(
        columns=[
            "ticker", "quarter", "transcript", "parsed_and_analyzed_facts",
            "research_note", "actual_return", "error"
        ]
    ).to_csv(LOG_PATH, index=False)

def process_sector(sector_df: pd.DataFrame) -> List[dict]:
    import concurrent.futures
    from pathlib import Path
    from openai import OpenAI
    import json
    creds = json.loads(Path("credentials.json").read_text())
    openai_client = OpenAI(api_key=creds["openai_api_key"])
    model = "gpt-4o-mini"

    try:
        sector_df = sector_df.sort_values("parsed_date")
        for _, row in sector_df.iterrows():
            ticker = row['ticker']
            quarter = row.get("q")
            if pd.isna(quarter):
                print(f"   - ⚠️ Skipping {ticker}: No quarter available")
                continue

            result_dict = {
                "ticker"        : ticker,
                "quarter"       : quarter,
                "transcript"    : row["transcript"],
                "parsed_and_analyzed_facts": "[]",
                "research_note" : "",
                "actual_return" : row["future_3bday_cum_return"],
                "error"         : "",
            }

            try:
                if not check_financial_statement_files_exist(ticker):
                    print(f"   - ⏭️ Skipping {ticker}/{quarter}: No financial statement files found")
                    result_dict["error"] = "No financial statement files available"
                    pd.DataFrame([result_dict]).to_csv(LOG_PATH, mode="a", header=False, index=False)
                    continue

                financial_facts = generate_financial_statement_facts(row, ticker, quarter)

                # Get multiple historical quarters instead of just previous quarter
                target_quarters = get_historical_quarters(quarter, NUM_HISTORICAL_PERIODS)

                # Group facts by quarter
                facts_by_quarter = {}
                for q in target_quarters:
                    facts_by_quarter[q] = [f for f in financial_facts if f.get('quarter') == q and f.get('type') == 'Result']

                # Debug: Show what quarters are available
                available_quarters = sorted(set(f.get('quarter') for f in financial_facts if f.get('type') == 'Result'))
                print(f"   📊 Available quarters for {ticker}: {available_quarters}")
                print(f"   🔍 Looking for quarters: {target_quarters}")
                for q in target_quarters:
                    print(f"   📈 Found {len(facts_by_quarter[q])} facts for {q}")

                # Deduplicate facts by metric and quarter (keep the first occurrence)
                def deduplicate_facts(facts_list):
                    seen = set()
                    deduplicated = []
                    for fact in facts_list:
                        metric_quarter = (fact['metric'], fact['quarter'])
                        if metric_quarter not in seen:
                            seen.add(metric_quarter)
                            deduplicated.append(fact)
                    return deduplicated

                # Deduplicate facts for each quarter
                for q in target_quarters:
                    facts_by_quarter[q] = deduplicate_facts(facts_by_quarter[q])

                def format_multi_period_facts(facts_by_quarter, target_quarters):
                    lines = []

                    for i, q in enumerate(target_quarters):
                        facts = facts_by_quarter[q]
                        if i == 0:
                            period_label = f"Current Quarter ({q})"
                        else:
                            period_label = f"Quarter {q} ({i} quarters ago)"

                        lines.append(f"{period_label}:")
                        if facts:
                            for f in facts:
                                lines.append(f"• {f['metric']}: {f['value']} ({f['quarter']})")
                        else:
                            lines.append("No data available")

                    # Show what quarters are actually available
                    if available_quarters:
                        lines.append(f"\nAvailable quarters in data: {', '.join(available_quarters)}")

                    return '\n'.join(lines)

                financial_statements_facts_str = format_multi_period_facts(facts_by_quarter, target_quarters)

                notes = {"financials": "", "past": "", "peers": ""}
                llm_prompt = main_agent_prompt(notes, original_transcript = row["transcript"], financial_statements_facts=financial_statements_facts_str)
                print(llm_prompt)

                with openai_semaphore:
                    response = openai_client.chat.completions.create(
                        model=model,
                        messages=[{"role": "system", "content": "You are a seasoned portfolio manager."},
                                  {"role": "user", "content": llm_prompt}],
                        temperature=0,
                        top_p=1,
                    )
                    llm_output = response.choices[0].message.content.strip()

                result_dict["research_note"] = llm_output

                # Combine all facts from all quarters
                all_facts = []
                for q in target_quarters:
                    all_facts.extend(facts_by_quarter[q])

                result_dict["parsed_and_analyzed_facts"] = json.dumps(all_facts)
                result_dict["error"] = ""

            except Exception as e:
                result_dict["error"] = str(e)

            pd.DataFrame([result_dict]).to_csv(LOG_PATH, mode="a", header=False, index=False)

            if sector_df.index.get_loc(_) % 10 == 0:
                print(f"   📝 Written {sector_df.index.get_loc(_) + 1}/{len(sector_df)} rows for {ticker}")

    finally:
        pass

    return []

def main() -> None:
    print("🚀 Starting Multi-Period Parallel Facts Indexation")
    start_time = time.time()

    if not os.path.exists(LOG_PATH):
        initialize_log_file()
    ensure_log_directories()

    df = pd.read_csv(DATA_FILE).drop_duplicates()
    df["returns"] = df["future_3bday_cum_return"]
    # df = df.iloc[878:]
    df = df.dropna(subset=["parsed_date"])
    df['parsed_date'] = pd.to_datetime(df['parsed_date']).dt.tz_localize(None)
    # df = df.sort_values("parsed_date").reset_index(drop=True)
    df = df.dropna(subset=["future_3bday_cum_return"]).reset_index(drop=True)
    # df = df.merge(SECTOR_MAP_DF, on="ticker", how="left")
    # df = df.dropna(subset=["sector"])

    unique_tickers = df['ticker'].unique()
    total_tickers = len(unique_tickers)
    total_chunks = math.ceil(total_tickers / CHUNK_SIZE)

    print(f"📊 Processing {total_tickers} unique tickers in {total_chunks} chunks of {CHUNK_SIZE}")
    print(f"🔄 Using {NUM_HISTORICAL_PERIODS} periods (current + {NUM_HISTORICAL_PERIODS-1} historical)")

    for chunk_idx in range(total_chunks):
        start_ticker_idx = chunk_idx * CHUNK_SIZE
        end_ticker_idx = min((chunk_idx + 1) * CHUNK_SIZE, total_tickers)
        chunk_tickers = unique_tickers[start_ticker_idx:end_ticker_idx]

        print(f"\n🔄 Processing chunk {chunk_idx + 1}/{total_chunks} (tickers {start_ticker_idx + 1}-{end_ticker_idx})")
        print(f"📋 Tickers in this chunk: {', '.join(chunk_tickers[:5])}{'...' if len(chunk_tickers) > 5 else ''}")

        chunk_df = df[df['ticker'].isin(chunk_tickers)].copy()
        print(f"📈 Processing {len(chunk_df)} rows for {len(chunk_tickers)} tickers")

        ticker_groups = [g for _, g in chunk_df.groupby("ticker")]
        print(f"🏭 Processing {len(ticker_groups)} tickers in this chunk")

        max_w = 10
        print(f"🔄 Launching {len(ticker_groups)} tickers on {max_w} CPU workers...")

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_w) as pool:
            futures = {
                pool.submit(process_sector, grp): grp["ticker"].iat[0]
                for grp in ticker_groups
            }

            for fut in concurrent.futures.as_completed(futures):
                ticker = futures[fut]
                try:
                    fut.result()
                except Exception as err:
                    print(f"❌ Error in ticker {ticker}: {err}")
                print(f" ✅ finished ticker {ticker}")

        chunk_time = time.time() - start_time
        print(f"✅ Completed chunk {chunk_idx + 1}/{total_chunks} in {chunk_time:.2f} seconds")
        print(f"📊 Average time per ticker in chunk: {chunk_time/len(chunk_tickers):.2f} seconds")

        processed_tickers = (chunk_idx + 1) * CHUNK_SIZE
        if processed_tickers > total_tickers:
            processed_tickers = total_tickers
        progress_pct = (processed_tickers / total_tickers) * 100
        print(f"📈 Overall progress: {processed_tickers}/{total_tickers} tickers ({progress_pct:.1f}%)")

        if os.path.exists(LOG_PATH):
            try:
                df_check = pd.read_csv(LOG_PATH)
                print(f"📊 Total rows in CSV after chunk {chunk_idx + 1}: {len(df_check):,}")
            except Exception as e:
                print(f"⚠️  Could not check CSV file: {e}")

    total_time = time.time() - start_time
    print(f"\n🎉 Multi-Period Parallel Facts Indexation Complete!")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print(f"📊 Average time per row: {total_time/len(df):.2f} seconds")
    print(f"\n📁 Log files created in:")
    print(f"  - {TOKEN_LOG_DIR}/ (token usage logs)")
    print(f"  - {TIMING_LOG_DIR}/ (timing logs)")

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()