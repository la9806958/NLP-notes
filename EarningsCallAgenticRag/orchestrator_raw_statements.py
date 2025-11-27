# This is a raw statements orchestrator for earnings call analysis.
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
LOG_PATH = "FinalResultsRawStatements.csv"
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
NUM_HISTORICAL_PERIODS = 4

METRIC_MAPPING = {
    "Main business income": "Revenue",
    "主营收入": "Revenue",
    "Main Business Income": "Revenue",
    "MAIN BUSINESS INCOME": "Revenue",
}
openai_semaphore = threading.Semaphore(4)

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

def map_metric_name(metric_name: str) -> str:
    return METRIC_MAPPING.get(metric_name, metric_name)

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

                # No financial statement data - just use transcript
                financial_statements_facts_str = ""

                print(f"   📊 Processing transcript only for {ticker}")

                # Use empty financial statements with the prompt
                notes = {"financials": "", "past": "", "peers": ""}
                llm_prompt = main_agent_prompt(notes, original_transcript = row["transcript"], financial_statements_facts=financial_statements_facts_str)

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
                result_dict["parsed_and_analyzed_facts"] = "[]"  # No structured facts, just raw data
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
    print("🚀 Starting Raw Statements Parallel Processing")
    start_time = time.time()

    if not os.path.exists(LOG_PATH):
        initialize_log_file()
    ensure_log_directories()

    df = pd.read_csv(DATA_FILE).drop_duplicates()
    df["returns"] = df["future_3bday_cum_return"]
    df = df.dropna(subset=["parsed_date"])
    df['parsed_date'] = pd.to_datetime(df['parsed_date']).dt.tz_localize(None)
    df = df.dropna(subset=["future_3bday_cum_return"]).reset_index(drop=True)

    unique_tickers = df['ticker'].unique()
    total_tickers = len(unique_tickers)
    total_chunks = math.ceil(total_tickers / CHUNK_SIZE)

    print(f"📊 Processing {total_tickers} unique tickers in {total_chunks} chunks of {CHUNK_SIZE}")
    print(f"🔄 Using {NUM_HISTORICAL_PERIODS} historical periods of raw financial statement data")

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
    print(f"\n🎉 Raw Statements Parallel Processing Complete!")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print(f"📊 Average time per row: {total_time/len(df):.2f} seconds")
    print(f"\n📁 Log files created in:")
    print(f"  - {TOKEN_LOG_DIR}/ (token usage logs)")
    print(f"  - {TIMING_LOG_DIR}/ (timing logs)")

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()