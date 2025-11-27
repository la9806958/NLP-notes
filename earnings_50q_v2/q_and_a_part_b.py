import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from openai import OpenAI
import json
import threading

with open('credentials.json', 'r') as f:
    credentials = json.load(f)

client = OpenAI(api_key=credentials['openai_api_key'])

OUTPUT_FILE = "earnings_call_analysis_results_2.csv"
LOCK = threading.Lock()

def analyze_qna_with_llm(current_qna, previous_qnas, ticker, current_datetime, previous_datetimes):
    previous_text = "\n\n---\n\n".join([f"Previous Call {i+1} ({previous_datetimes[i]}):\n{qna}" for i, qna in enumerate(previous_qnas)])

    prompt = f"""You are analysing Q&A sections of earnings calls. Comment only on the current call vs the previous calls.

Previous Calls:
{previous_text}

Current Call ({current_datetime}):
{current_qna}

Perform the following tasks:

Task 1:
For each speaker from the firm in the current call, write 1–2 short sentences evaluating how the speaker's response to analysts' questions differs from previous calls. Evaluate (1) quality, persuasiveness, believability of the response and (2) content of the response (with an example). Use only comparative terms like more, less, further, clearer, vaguer.

Output must follow the format:
[Speaker Name]: Response is [comparative assessment of quality/persuasiveness]. Includes [comparative assessment of content with example].

Task 2:
Across the questions posited by the analysts, write 1-2 short sentences explaining how market's sentiment towards the firm has shifted. Evaluate (1) market sentiment (bullish or bearish), (2) specific areas of interest (eg. Balance sheet strength, growth oriented, probability of survival). Use only comparative terms like more, less, further, clearer, vaguer.

Output must follow the format:
Questions are [comparative assessment of bullish / bearish]. Includes [comparative assessment of content with example].

Task 3:
Evaluate the quality of the management's answers and the sentiment implied by analysts' questions.

Provide numerical scores [0–10] where:
* 0 = fully dodging / fearful / flamboyant
* 5 = neutral
* 10 = fully quantified / bullish / conservative


Management Response Quality
1. General Stance (More Dodging than Previous Calls vs More Direct than Previous Calls): 0 = Dodging, 10 = Direct
2. Existing Performance (More Vague than Previous Calls vs More Quantification than Previous Calls): 0 = Vague, 10 = Quantified
3. Forward-Looking Guidance (More Dodging than Previous Calls vs More Direct than Previous Calls): 0 = Dodging, 10 = Clear forecasts
4. Competitor Positioning (More Dodging than Previous Calls vs More Direct than Previous Calls): 0 = No specifics, 10 = Explicit metrics
5. Consistency (More Contradictory than Previous Calls vs More Aligned than Previous Calls): 0 = Contradicts, 10 = Consistent

Analyst Sentiment
6. Sentiment (More Cautious than Previous Calls vs More Bullish than Previous Calls): 0 = Fearful, 10 = Bullish
7. Focus (More Downside Protection than Previous Call vs More Upside Potential than Previous Call): 0 = Downside, 10 = Upside
8. Proportion of Well-Answered Questions: 0 = Almost none, 10 = Almost all

Language Style
9. Tone (Flamboyant vs Conservative): 0 = Flamboyant, 10 = Conservative
10. Clarity (Poorly Addressed vs Well Addressed): 0 = Poorly addressed, 10 = Well addressed

Respond in the following format:

Task 1:
[Your Task 1 response here]

Task 2:
[Your Task 2 response here]

Task 3:
score1, score2, score3, score4, score5, score6, score7, score8, score9, score10"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5000
        )

        content = response.choices[0].message.content.strip()

        task1_start = content.find("Task 1:")
        task2_start = content.find("Task 2:")
        task3_start = content.find("Task 3:")

        if task1_start == -1 or task2_start == -1 or task3_start == -1:
            raise ValueError("Could not find task markers in response")

        task1_text = content[task1_start + len("Task 1:"):task2_start].strip()
        task2_text = content[task2_start + len("Task 2:"):task3_start].strip()
        task3_text = content[task3_start + len("Task 3:"):].strip()

        scores = [int(x.strip()) for x in task3_text.split(',')]
        if len(scores) != 10:
            raise ValueError(f"Expected 10 scores, got {len(scores)}")

        return {
            "task1": task1_text,
            "task2": task2_text,
            "task3": scores
        }
    except Exception as e:
        print(f"Error analyzing ticker {ticker}: {e}")
        print(f"Response content: {response.choices[0].message.content if 'response' in locals() else 'N/A'}")
        return {
            "task1": f"Error: {str(e)}",
            "task2": f"Error: {str(e)}",
            "task3": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        }

def process_single_call(row_data, previous_calls_df):
    ticker = row_data['ticker']
    et_timestamp = row_data['et_timestamp']
    current_qna = row_data['task1_speaker_qualification'] + "\n\n" + row_data['task2_analyst_sentiment']

    previous_calls = previous_calls_df[
        (previous_calls_df['ticker'] == ticker) &
        (previous_calls_df['et_timestamp'] < et_timestamp)
    ].sort_values('et_timestamp', ascending=False).head(4)

    if len(previous_calls) < 1:
        return None

    previous_calls_sorted = previous_calls.sort_values('et_timestamp', ascending=True)
    previous_qnas = [str(row['task1_speaker_qualification']) + "\n\n" + str(row['task2_analyst_sentiment']) for _, row in previous_calls_sorted.iterrows()]
    previous_datetimes = previous_calls_sorted['et_timestamp'].astype(str).tolist()

    result = analyze_qna_with_llm(current_qna, previous_qnas, ticker, str(et_timestamp), previous_datetimes)

    output_row = {}
    for col in row_data.index:
        val = row_data[col]
        if isinstance(val, (list, dict)):
            output_row[col] = str(val)
        else:
            output_row[col] = val

    output_row['task1_response'] = result['task1']
    output_row['task2_response'] = result['task2']

    for i, score in enumerate(result['task3'], 1):
        output_row[f'task3_score_{i}'] = score

    return output_row

def append_to_csv(row, filename):
    with LOCK:
        df_row = pd.DataFrame([row])
        if not os.path.exists(filename):
            df_row.to_csv(filename, index=False, mode='w')
        else:
            df_row.to_csv(filename, index=False, mode='a', header=False)

def process_batch(batch_data, all_calls_df, worker_id):
    results = []
    for idx, (_, row) in enumerate(batch_data.iterrows()):
        try:
            print(f"Worker {worker_id}: Processing {idx+1}/{len(batch_data)} - Ticker: {row['ticker']}, Date: {row['date']}")
            result = process_single_call(row, all_calls_df)
            if result is not None:
                append_to_csv(result, OUTPUT_FILE)
                results.append(result)
                print(f"Worker {worker_id}: Completed {idx+1}/{len(batch_data)} - Logged to CSV")
        except Exception as e:
            print(f"Worker {worker_id}: Error processing row: {e}")
    return results

def main():
    print("Loading data...")
    df = pd.read_csv('earnings_call_analysis_results_modified.csv',
                     usecols=['ticker', 'date', 'et_timestamp', 'task1_speaker_qualification', 'task2_analyst_sentiment'])
    close_matrix = pd.read_csv('hourly_close_to_close_returns_matrix.csv')

    print(f"Original data: {len(df)} rows")

    df['et_timestamp'] = df['et_timestamp'].str.replace(" ET", "", regex=False)
    df['et_timestamp'] = pd.to_datetime(df['et_timestamp'], errors='coerce')
    df = df.sort_values('et_timestamp')

    df_filtered = df[~df["task1_speaker_qualification"].isna() & ~df["task2_analyst_sentiment"].isna()].copy()
    print(f"After filtering NaN task1_speaker_qualification or task2_analyst_sentiment: {len(df_filtered)} rows")

    df_filtered = df_filtered[df_filtered["ticker"].isin(close_matrix.columns.tolist())]
    print(f"After filtering tickers in close_matrix: {len(df_filtered)} rows")

    if os.path.exists(OUTPUT_FILE):
        print(f"Removing existing output file: {OUTPUT_FILE}")
        os.remove(OUTPUT_FILE)

    num_workers = 20
    batch_size = len(df_filtered) // num_workers + 1

    batches = []
    for i in range(num_workers):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df_filtered))
        if start_idx < len(df_filtered):
            batches.append((df_filtered.iloc[start_idx:end_idx], i))

    print(f"\nStarting parallel processing with {num_workers} workers...")
    print(f"Total batches: {len(batches)}")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_batch, batch, df_filtered, worker_id): worker_id
            for batch, worker_id in batches
        }

        completed = 0
        for future in as_completed(futures):
            worker_id = futures[future]
            try:
                results = future.result()
                completed += 1
                print(f"\n=== Worker {worker_id} completed ({completed}/{len(batches)} batches done) ===\n")
            except Exception as e:
                print(f"\n!!! Worker {worker_id} failed with error: {e} !!!\n")

    print(f"\nProcessing complete! Results saved to {OUTPUT_FILE}")

    if os.path.exists(OUTPUT_FILE):
        final_df = pd.read_csv(OUTPUT_FILE)
        print(f"Final output: {len(final_df)} rows")

if __name__ == "__main__":
    main()
