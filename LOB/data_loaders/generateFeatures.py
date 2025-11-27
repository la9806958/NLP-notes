import os
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
from features import LOBLoader

# Paths
data_folder = "/home/lichenhui/stitched_tickers"
output_folder = "/home/lichenhui/transformed_features"
os.makedirs(output_folder, exist_ok=True)

# Loader settings
T = 100
k = 10

# Worker function
def process_file(file, data_folder, output_folder, T, k):
    if not file.endswith("_stitched.csv"):
        return

    ticker = file.replace("_stitched.csv", "")
    csv_path = os.path.join(data_folder, file)

    try:
        #print(f"[{ticker}] Loading...")
        df_raw = pd.read_csv(csv_path)

        loader = LOBLoader(T=T, k=k)  # Instantiate inside process
        df_transformed = loader.transform_orderbook(ticker=ticker, df=df_raw)

        if df_transformed.empty:
            print(f"[{ticker}] ⚠️ Empty transformed DataFrame, skipping.")
            return

        # Save the transformed DataFrame to Parquet (efficient for large numeric tables)
        out_path = os.path.join(output_folder, f"{ticker}_transformed.parquet")
        df_transformed.to_parquet(out_path, index=False)
        # print(f"[{ticker}] ✅ Saved transformed DataFrame")

    except Exception as e:
        print(f"[{ticker}] ❌ Error: {e}")

# Main
if __name__ == "__main__":
    files = os.listdir(data_folder)
    num_workers = cpu_count()

    print(f"Using {num_workers} CPU cores...")

    with Pool(processes=num_workers) as pool:
        pool.map(
            partial(
                process_file,
                data_folder=data_folder,
                output_folder=output_folder,
                T=T,
                k=k
            ),
            files
        )
