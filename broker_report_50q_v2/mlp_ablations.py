#!/usr/bin/env python3
"""
Regression version of Broker Report MLP pipeline with efficient target computation.
Uses vectorized operations and sharded parallel processing for much faster execution.
MLP(50 -> 256 -> 128 -> 1), α=10^-3 (L2), batch=256, η_0=10^-3
Output: datetime by ticker continuous regression predictions
"""

import logging
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import resample
from datetime import datetime
import warnings
import multiprocessing as mp
from functools import partial
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import random
warnings.filterwarnings('ignore')

# ────────────────────────────── Fixed Random Seeds ──────────────────────────────
def set_all_seeds(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ────────────────────────────── Logging ──────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ────────────────────────────── Constants ────────────────────────────
HOURS_FWD = 80  # ten trading days - tried 10
PAST_WIN = 168  # 21 trading days * 8 hours
MIN_FWD = 15    # minimum periods for forward return calculation
TRAIN_START_DATE = "2023-07-01"
VAL_END_DATE = "2024-05-01"

# ────────────────────────────── Feature Set Definitions ──────────────────────────────
FEATURE_SETS = {
    'event_nature': [f'q{i}_score' for i in range(1, 11)],  # Q1-10
    'thesis': [f'q{i}_score' for i in range(11, 21)],        # Q11-20
    'valuations': [f'q{i}_score' for i in range(21, 41)],    # Q21-40
    'comparative': [f'q{i}_score' for i in [5, 6, 11, 12, 15, 16, 20, 23, 29, 34, 35, 37, 38, 42, 44, 45, 46, 47]],  # Cross-cutting
    'temporal_orientation': [f'q{i}_score' for i in [3, 13, 16, 17, 19, 20, 22, 27, 28]],  # Temporal: past/present/future
    'information_novelty': [f'q{i}_score' for i in [4, 5, 6, 7, 8, 12, 18, 32, 33, 34, 35, 36]]  # Novelty vs consensus
}

ALL_FEATURES = [f'q{i}_score' for i in range(1, 51)]

# ────────────────────────────── Efficient Data Processing ────────────────────────────

def exponential_smoothing(values, alpha=0.3):
    """
    Apply exponential smoothing to a series of values.

    Args:
        values: Array-like of values to smooth
        alpha: Smoothing parameter (0 < alpha <= 1). Higher alpha gives more weight to recent values.

    Returns:
        Smoothed values as numpy array
    """
    if len(values) == 0:
        return np.array([])

    values = np.array(values)
    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]

    for i in range(1, len(values)):
        smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i-1]

    return smoothed

def calculate_historical_volatility(returns_df, ticker, report_date, lookback_hours=168):
    """
    Calculate historical volatility for a ticker over the past lookback_hours.

    Args:
        returns_df: DataFrame with DateTime and ticker columns
        ticker: Ticker symbol
        report_date: Reference date for lookback
        lookback_hours: Number of hours to look back (default 168 = 21 days * 8 hours)

    Returns:
        volatility: Standard deviation of returns over the lookback period
    """
    if ticker not in returns_df.columns:
        return np.nan

    # Get historical returns before the report date
    historical_data = returns_df[returns_df['DateTime'] <= report_date]

    if len(historical_data) < lookback_hours:
        # Use all available data if we don't have enough history
        ticker_returns = historical_data[ticker].dropna()
    else:
        # Get the last lookback_hours periods
        ticker_returns = historical_data[ticker].tail(lookback_hours).dropna()

    if len(ticker_returns) < 10:  # Need minimum data points for meaningful volatility
        return np.nan

    # Calculate volatility as standard deviation of returns
    volatility = ticker_returns.std()

    # Avoid division by zero - return small positive number if volatility is zero
    return max(volatility, 1e-6)

# ────────────────────────────── Data Prep ────────────────────────────
def load_and_prepare_data():
    logger.info("Loading broker report data...")

    # Load broker report analysis results
    broker_df = pd.read_csv('/home/lichenhui/broker_report_analysis_results.csv')

    # Convert current_et_timestamp to datetime
    broker_df['precise_date'] = pd.to_datetime(broker_df['current_et_timestamp'], errors='coerce', utc=True)
    broker_df = broker_df.dropna(subset=['precise_date'])
    broker_df['precise_date'] = broker_df['precise_date'].dt.tz_localize(None)

    # Filter out samples reported at exact hours (0 minutes and 0 seconds)
    initial_count = len(broker_df)
    broker_df = broker_df[~((broker_df['precise_date'].dt.minute == 0) & (broker_df['precise_date'].dt.second == 0))]
    filtered_count = initial_count - len(broker_df)
    logger.info(f"Filtered out {filtered_count} samples reported at exact hours (0 minutes and 0 seconds)")

    # Load returns matrix (hourly data)
    returns_df = pd.read_csv('/home/lichenhui/close_to_close_returns_matrix_2023_2025.csv')
    returns_df['DateTime'] = pd.to_datetime(returns_df['DateTime'])

    logger.info(f"Loaded {len(broker_df)} broker report records")
    logger.info(f"Loaded returns data with {len(returns_df)} dates and {len(returns_df.columns)-1} tickers")

    return broker_df, returns_df

def process_broker_chunk(chunk_data):
    """Process a chunk of broker reports to calculate forward returns with proper time windows."""
    broker_chunk, returns_df, feature_cols = chunk_data
    training_data = []

    for _, row in broker_chunk.iterrows():
        ticker = row['ticker']
        report_date = row['precise_date']

        # Skip if ticker not in returns matrix
        if ticker not in returns_df.columns:
            continue

        # Get future returns after the report date
        future_returns = returns_df[returns_df['DateTime'] > report_date]
        if len(future_returns) <= HOURS_FWD:
            continue

        # Get the anchor date (first trading date after report)
        anchor_date = future_returns['DateTime'].iloc[0]

        # Get the next HOURS_FWD returns (skip first period, start from t+1)
        # ticker_returns = future_returns[ticker].iloc[1:HOURS_FWD+1]
        ticker_returns = future_returns[ticker].iloc[1:HOURS_FWD+1]
        valid_returns = ticker_returns.dropna()

        # Need minimum periods for valid calculation
        if len(valid_returns) < MIN_FWD:
            continue

        # Calculate the end of the forward window
        label_end_date = future_returns['DateTime'].iloc[HOURS_FWD] if len(future_returns) > HOURS_FWD else future_returns['DateTime'].iloc[-1]

        # Apply exponential smoothing to the returns before calculating cumulative return
        smoothed_returns = exponential_smoothing(valid_returns.values, alpha=0.3)

        # Calculate cumulative return from smoothed values: (1+r1)*(1+r2)*...*(1+rn) - 1
        cumulative_return = (1 + smoothed_returns).prod() - 1

        # Calculate historical volatility for normalization
        historical_vol = calculate_historical_volatility(returns_df, ticker, report_date, lookback_hours=168)

        # Normalize return by volatility (risk-adjusted return)
        if not np.isnan(historical_vol) and historical_vol > 0:
            volatility_adjusted_return = cumulative_return / historical_vol
        else:
            # Skip samples where we can't calculate volatility
            continue

        # Create training sample with time windows
        sample = {
            'ticker': ticker,
            'date': report_date,
            'anchor_date': anchor_date,  # First trading date after report
            'label_end_date': label_end_date,  # End of forward window
            'target_value': volatility_adjusted_return,  # Use volatility-adjusted return
            'cumulative_return': cumulative_return,
            'historical_volatility': historical_vol,
            'volatility_adjusted_return': volatility_adjusted_return
        }

        # Add feature columns
        for col in feature_cols:
            sample[col] = row[col]

        training_data.append(sample)

    return training_data

def create_training_data_optimized(broker_df, returns_df):
    """Create training data using efficient chunk-based parallel processing."""
    feature_cols = [f'q{i}_score' for i in range(1, 51)]

    # Check which feature columns actually exist
    existing_features = [col for col in feature_cols if col in broker_df.columns]
    if len(existing_features) < 50:
        logger.warning(f"Only found {len(existing_features)} feature columns out of 50 expected")
        feature_cols = existing_features

    logger.info(f"Using {len(feature_cols)} features")
    logger.info("Processing broker reports in parallel chunks...")

    # Parallel processing using chunks (much faster than merge_asof)
    num_processes = 30
    chunk_size = len(broker_df) // num_processes + 1
    chunks = [
        (broker_df.iloc[i:i+chunk_size], returns_df, feature_cols)
        for i in range(0, len(broker_df), chunk_size)
    ]

    with mp.Pool(processes=num_processes) as pool:
        chunk_results = pool.map(process_broker_chunk, chunks)

    # Combine all chunk results
    training_data = [row for chunk in chunk_results for row in chunk]
    training_df = pd.DataFrame(training_data)

    logger.info(f"Created {len(training_df)} training samples from {len(broker_df)} broker reports")
    return training_df

# ────────────────────────────── Mean Centering ──────────────────────────────
class MeanCenterer:
    """Mean-center features by subtracting 5 (since inputs range 0-10)"""
    def __init__(self):
        self.fitted = False

    def fit_transform(self, X):
        X = np.asarray(X)
        self.fitted = True
        return X - 5.0

    def transform(self, X):
        if not self.fitted:
            raise ValueError("Scaler not fitted")
        X = np.asarray(X)
        return X - 5.0

# ────────────────────────────── Model ──────────────────────────────
class MLPRegressor(nn.Module):
    def __init__(self, input_size=50, hidden1=128, hidden2=64, output_size=1, dropout_rate=0.3):
        super().__init__()
        # Smaller hidden layers for reduced capacity
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_size)
        self.relu = nn.ReLU()

        # Multiple dropout layers with higher rates
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate * 0.7)  # Slightly lower before output

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.dropout3(x)  # Additional dropout before output
        return self.fc3(x)

# ────────────────────────────── Metrics ──────────────────────────────
def calculate_regression_metrics(model, loader, device, return_predictions=False):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb.to(device))
            preds.extend(out.cpu().numpy().flatten())
            targets.extend(yb.numpy())

    preds, targets = np.array(preds), np.array(targets)

    # Calculate regression metrics
    mse = mean_squared_error(targets, preds)
    mae = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, preds)

    # Calculate correlations
    pearson_corr, pearson_p = pearsonr(preds, targets)
    spearman_corr, spearman_p = spearmanr(preds, targets)

    metrics = {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'pearson_corr': pearson_corr,
        'pearson_p': pearson_p,
        'spearman_corr': spearman_corr,
        'spearman_p': spearman_p
    }

    if return_predictions:
        return metrics, preds, targets
    return metrics

def analyze_weight_magnitude_saliency(model, feature_cols):
    """Analyze feature importance using weight magnitudes from input to first hidden layer."""
    # Get weights from input layer to first hidden layer
    first_layer_weights = model.fc1.weight.data.cpu().numpy()  # Shape: (hidden_size, input_size)

    # Calculate absolute weight magnitudes summed across all hidden units
    weight_magnitudes = np.abs(first_layer_weights).sum(axis=0)  # Sum across hidden dimension

    # Create feature importance dataframe
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'weight_magnitude': weight_magnitudes,
        'normalized_importance': weight_magnitudes / weight_magnitudes.sum()
    })

    # Sort by importance
    feature_importance = feature_importance.sort_values('weight_magnitude', ascending=False)

    return feature_importance, first_layer_weights

def create_saliency_visualization(feature_importance, save_path):
    """Create and save feature importance visualization."""
    plt.figure(figsize=(12, 8))

    # Plot top 20 features
    top_features = feature_importance.head(50)

    plt.barh(range(len(top_features)), top_features['weight_magnitude'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Summed Absolute Weight Magnitude')
    plt.title('Feature Importance via Weight Magnitude Saliency (Top 20)')
    plt.gca().invert_yaxis()

    # Add value labels
    for i, v in enumerate(top_features['weight_magnitude']):
        plt.text(v + 0.01 * max(top_features['weight_magnitude']), i, f'{v:.3f}',
                va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved feature importance plot: {save_path}")

    return save_path

def time_aware_split(training_df, test_cutoff="2024-05-01", embargo_hours=HOURS_FWD, val_ratio=0.2):
    """Split data with proper time windows and embargo periods to prevent ALL leakage."""
    logger.info("Performing time-aware split with embargo periods for ALL splits...")

    # Ensure deterministic sorting
    training_df = training_df.sort_values(['anchor_date', 'ticker', 'date']).reset_index(drop=True)

    # Convert cutoff to datetime
    test_cutoff_dt = pd.to_datetime(test_cutoff)

    # Test set: samples whose anchor_date >= test_cutoff
    test_mask = training_df['anchor_date'] >= test_cutoff_dt
    test_df = training_df[test_mask].copy()

    # Train/Val candidates: samples whose label_end_date < test_cutoff (with embargo)
    embargo_cutoff = test_cutoff_dt - pd.Timedelta(hours=embargo_hours)
    trainval_mask = training_df['label_end_date'] < embargo_cutoff
    trainval_df = training_df[trainval_mask].copy()

    # Temporal split within train/val to prevent window overlap
    if len(trainval_df) > 0:
        # Sort by anchor_date for temporal split
        trainval_df = trainval_df.sort_values('anchor_date').reset_index(drop=True)

        # Find validation cutoff point (70/30 split by count)
        val_start_idx = int(len(trainval_df) * (1 - val_ratio))

        # Get potential validation samples
        potential_val_df = trainval_df.iloc[val_start_idx:].copy()

        # Find actual validation start with embargo
        # Validation starts where NO training label windows overlap
        if len(potential_val_df) > 0:
            val_anchor_start = potential_val_df['anchor_date'].iloc[0]
            val_start_with_embargo = val_anchor_start - pd.Timedelta(hours=embargo_hours)

            # Training samples: those whose label_end_date < val_start_with_embargo
            train_df = trainval_df[trainval_df['label_end_date'] < val_start_with_embargo].copy()

            # Validation samples: those whose anchor_date >= val_anchor_start
            val_df = trainval_df[trainval_df['anchor_date'] >= val_anchor_start].copy()
        else:
            # Fallback: use all as training if validation set would be empty
            train_df = trainval_df.copy()
            val_df = pd.DataFrame()
    else:
        train_df = pd.DataFrame()
        val_df = pd.DataFrame()

    logger.info(f"Time-aware split completed with NO window overlaps:")
    logger.info(f"  Test cutoff: {test_cutoff_dt}")
    logger.info(f"  Test embargo: {embargo_cutoff}")
    if len(val_df) > 0:
        logger.info(f"  Val starts: {val_df['anchor_date'].min()}")
        logger.info(f"  Train ends: {train_df['label_end_date'].max() if len(train_df) > 0 else 'N/A'}")
    logger.info(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    return train_df, val_df, test_df

def train_mlp_regression_with_epoch_tracking(training_df, excluded_feature_set=None, variant_name="full"):
    # Set seeds for reproducibility - CRITICAL for fair ablation comparison
    set_all_seeds(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Use time-aware split with embargo periods
    train_df, val_df, test_df = time_aware_split(training_df)

    # Get feature columns - apply ablation if specified
    if excluded_feature_set is None:
        feature_cols = ALL_FEATURES.copy()
    else:
        # Exclude the specified feature set
        excluded_features = FEATURE_SETS[excluded_feature_set]
        feature_cols = [f for f in ALL_FEATURES if f not in excluded_features]
        logger.info(f"ABLATION: Excluding {excluded_feature_set} features: {excluded_features}")

    existing_features = [col for col in feature_cols if col in training_df.columns]
    feature_cols = existing_features

    logger.info(f"Using {len(feature_cols)} features for variant '{variant_name}'")
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Prepare features and targets (continuous regression targets)
    X_train = train_df[feature_cols].values
    y_train = train_df['target_value'].values

    X_val = val_df[feature_cols].values
    y_val = val_df['target_value'].values

    X_test = test_df[feature_cols].values
    y_test = test_df['target_value'].values

    # Log target statistics
    logger.info(f"TRAIN: n={len(y_train)}, target_mean={y_train.mean():.6f}, target_std={y_train.std():.6f}")
    logger.info(f"VAL: n={len(y_val)}, target_mean={y_val.mean():.6f}, target_std={y_val.std():.6f}")
    logger.info(f"TEST: n={len(y_test)}, target_mean={y_test.mean():.6f}, target_std={y_test.std():.6f}")

    # Mean-center features by subtracting 5
    scaler = MeanCenterer()
    X_train = scaler.fit_transform(X_train.astype(np.float32))
    X_val = scaler.transform(X_val.astype(np.float32))
    X_test = scaler.transform(X_test.astype(np.float32))

    # Create tensors and loaders with fixed random state
    X_train_t, y_train_t = torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    X_val_t, y_val_t = torch.FloatTensor(X_val), torch.FloatTensor(y_val)
    X_test_t, y_test_t = torch.FloatTensor(X_test), torch.FloatTensor(y_test)

    batch_size = 256
    # Set generator with fixed seed for reproducible shuffling
    generator = torch.Generator()
    generator.manual_seed(42)
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size)

    # Initialize model with heavy regularization settings and fixed initialization
    model = MLPRegressor(input_size=len(feature_cols), hidden1=128, hidden2=64, dropout_rate=0.4).to(device)

    # Initialize weights deterministically
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight, gain=1.0)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)

    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)  # 100x stronger weight decay
    criterion = nn.MSELoss()

    # Regularization hyperparameters
    l1_lambda = 1e-3  # L1 regularization strength
    l2_lambda = 1e-4  # Additional L2 regularization strength

    # Training loop
    best_val_loss, patience, patience_counter = float('inf'), 10, 0

    logger.info("Starting training with epoch-by-epoch metrics...")
    logger.info("-" * 100)
    logger.info("Epoch   Train Metrics                      |   Validation Metrics")
    logger.info("-" * 100)

    for epoch in range(1000):
        model.train()
        running, nobs = 0.0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb.to(device))

            # Base loss only - no additional regularization
            loss = criterion(out.flatten(), yb.to(device))

            loss.backward()
            optimizer.step()
            bs = yb.size(0)
            running += loss.item()*bs
            nobs += bs
        train_mse = running / max(nobs, 1)

        # Calculate metrics every epoch
        train_metrics = calculate_regression_metrics(model, train_loader, device)
        val_metrics = calculate_regression_metrics(model, val_loader, device)

        # Log metrics every epoch
        logger.info(f"Epoch {epoch+1:03d} - TrainMSE={train_mse:.6f} | "
                   f"Train: RMSE={train_metrics['rmse']:.6f} MAE={train_metrics['mae']:.6f} R²={train_metrics['r2']:.4f} | "
                   f"Val:   RMSE={val_metrics['rmse']:.6f}   MAE={val_metrics['mae']:.6f}   R²={val_metrics['r2']:.4f}")

        # Early stopping on val loss
        if val_metrics['mse'] < best_val_loss:
            best_val_loss, patience_counter, best_state = val_metrics['mse'], 0, model.state_dict().copy()
        else:
            patience_counter += 1
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)

    # Final metrics with detailed test analysis
    train_final = calculate_regression_metrics(model, train_loader, device)
    val_final = calculate_regression_metrics(model, val_loader, device)
    test_final, test_preds, test_targets = calculate_regression_metrics(
        model, test_loader, device, return_predictions=True)

    logger.info("="*80)
    logger.info("FINAL REGRESSION RESULTS")
    logger.info("="*80)
    logger.info(f"Train: RMSE={train_final['rmse']:.6f}, MAE={train_final['mae']:.6f}, R²={train_final['r2']:.4f}")
    logger.info(f"Val:   RMSE={val_final['rmse']:.6f}, MAE={val_final['mae']:.6f}, R²={val_final['r2']:.4f}")
    logger.info(f"Test:  RMSE={test_final['rmse']:.6f}, MAE={test_final['mae']:.6f}, R²={test_final['r2']:.4f}")
    logger.info(f"Test Pearson correlation: {test_final['pearson_corr']:.4f} (p={test_final['pearson_p']:.4e})")
    logger.info(f"Test Spearman correlation: {test_final['spearman_corr']:.4f} (p={test_final['spearman_p']:.4e})")

    # Analyze weight magnitude saliency
    feature_importance, first_layer_weights = analyze_weight_magnitude_saliency(model, feature_cols)

    logger.info("="*80)
    logger.info("FEATURE IMPORTANCE VIA WEIGHT MAGNITUDE SALIENCY")
    logger.info("="*80)
    logger.info("All 50 feature importance rankings:")
    for i, row in feature_importance.iterrows():
        logger.info(f"{row['feature']:>12}: {row['weight_magnitude']:8.4f} "
                    f"({row['normalized_importance']*100:5.2f}%)")

    return model, scaler, test_df, device, feature_cols, test_final, feature_importance

def process_prediction_chunk(chunk_data):
    """Process a chunk of broker reports for prediction matrix creation."""
    broker_chunk, returns_dates, model_state, scaler, feature_cols, device_str = chunk_data

    import torch
    import bisect

    # Recreate model and load state
    device = torch.device(device_str)
    model = MLPRegressor(input_size=len(feature_cols), hidden1=128, hidden2=64, dropout_rate=0.2).to(device)
    model.load_state_dict(model_state)
    model.eval()

    predictions_list = []

    with torch.no_grad():
        if len(broker_chunk) > 0:
            # Batch process features for this chunk
            features_matrix = broker_chunk[feature_cols].values.astype(np.float32)
            features_scaled = scaler.transform(features_matrix)
            features_tensor = torch.FloatTensor(features_scaled).to(device)

            # Get predictions for this chunk
            outputs = model(features_tensor)
            predictions = outputs.cpu().numpy().flatten()

            # Create prediction records for this chunk
            for i, (_, row) in enumerate(broker_chunk.iterrows()):
                ticker = row['ticker']
                report_date = row['precise_date']

                # Use binary search to find next trading date efficiently
                next_idx = bisect.bisect_right(returns_dates, report_date)
                if next_idx + 1 >= len(returns_dates):
                    continue
                target_date = returns_dates[next_idx + 1]  # shift one more hour to match label start (t+1)

                predictions_list.append({
                    'DateTime': target_date,
                    'ticker': ticker,
                    'prediction': predictions[i]
                })

    return predictions_list

def create_regression_prediction_matrix(broker_df, returns_df, model, scaler, device, feature_cols):
    """Create datetime by ticker regression prediction matrix using parallel processing."""
    logger.info("Creating regression prediction matrix...")

    # Get broker tickers that we can make predictions for
    broker_tickers = set(broker_df['ticker'].unique())
    returns_tickers = set(returns_df.columns[1:])  # Exclude 'DateTime' column
    common_tickers = sorted(broker_tickers.intersection(returns_tickers))

    # Filter broker data to only include tickers we can predict
    broker_filtered = broker_df[broker_df['ticker'].isin(common_tickers)].copy()
    logger.info(f"Processing {len(broker_filtered)} broker reports for {len(common_tickers)} tickers using parallel chunks")

    # Pre-sort returns dates for efficient lookup
    returns_dates = sorted(returns_df['DateTime'].unique())

    # Prepare chunks for parallel processing
    num_processes = 25
    chunk_size = len(broker_filtered) // num_processes + 1
    chunks = [
        (
            broker_filtered.iloc[i:i+chunk_size],
            returns_dates,
            model.state_dict(),
            scaler,
            feature_cols,
            str(device)
        )
        for i in range(0, len(broker_filtered), chunk_size)
    ]

    logger.info(f"Processing {len(chunks)} chunks in parallel...")

    # Process chunks in parallel
    with mp.Pool(processes=num_processes) as pool:
        chunk_results = pool.map(process_prediction_chunk, chunks)

    # Combine all prediction results
    all_predictions = []
    for chunk_preds in chunk_results:
        all_predictions.extend(chunk_preds)

    logger.info(f"Generated {len(all_predictions)} regression predictions")

    # Convert to matrix format efficiently
    if all_predictions:
        pred_df = pd.DataFrame(all_predictions)

        # Get all unique dates and tickers for the matrix
        all_dates = sorted(returns_df['DateTime'].unique())
        pred_matrix = pd.DataFrame(
            index=all_dates,
            columns=common_tickers,
            dtype=float
        )
        pred_matrix[:] = np.nan

        # Fill predictions efficiently using vectorized operations
        logger.info("Filling prediction matrix...")
        for _, row in pred_df.iterrows():
            pred_matrix.loc[row['DateTime'], row['ticker']] = row['prediction']

        logger.info(f"Created prediction matrix: {pred_matrix.shape[0]} dates × {pred_matrix.shape[1]} tickers")
        return pred_matrix
    else:
        logger.warning("No predictions generated")
        return pd.DataFrame()

def save_results(pred_matrix, model, scaler, feature_cols, test_metrics, feature_importance, variant_name="full"):
    """Save prediction matrix, model, and analysis results."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save regression prediction matrix
    matrix_path = f"/home/lichenhui/broker_report_regression_prediction_matrix_{variant_name}_{timestamp}.csv"
    pred_matrix.to_csv(matrix_path)
    logger.info(f"Saved regression prediction matrix: {matrix_path}")

    # Save test results
    results_path = f"/home/lichenhui/broker_report_regression_results_{variant_name}_{timestamp}.csv"
    results_df = pd.DataFrame({
        'metric': ['rmse', 'mae', 'r2', 'pearson_corr', 'pearson_p', 'spearman_corr', 'spearman_p'],
        'value': [test_metrics['rmse'], test_metrics['mae'], test_metrics['r2'],
                  test_metrics['pearson_corr'], test_metrics['pearson_p'],
                  test_metrics['spearman_corr'], test_metrics['spearman_p']]
    })
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved regression results: {results_path}")

    # Save feature importance results
    importance_path = f"/home/lichenhui/broker_report_regression_feature_importance_{variant_name}_{timestamp}.csv"
    feature_importance.to_csv(importance_path, index=False)
    logger.info(f"Saved feature importance: {importance_path}")

    # Create and save feature importance visualization
    plot_path = f"/home/lichenhui/broker_report_regression_feature_importance_plot_{variant_name}_{timestamp}.png"
    create_saliency_visualization(feature_importance, plot_path)

    # Save model components with additional analysis
    import pickle
    model_path = f"/home/lichenhui/broker_report_regression_mlp_model_{variant_name}_{timestamp}.pkl"
    model_data = {
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'feature_cols': feature_cols,
        'architecture': f'MLPRegressor({len(feature_cols)} -> 256 -> 128 -> 1) - {variant_name} variant',
        'hyperparameters': {
            'alpha': 1e-3,
            'batch_size': 256,
            'learning_rate_init': 1e-3
        },
        'target_description': f'10-day forward return (continuous) - {variant_name} ablation',
        'test_metrics': test_metrics,
        'feature_importance': feature_importance,
        'date_ranges': {
            'train_start': TRAIN_START_DATE,
        }
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    logger.info(f"Saved model: {model_path}")

    return matrix_path, model_path, results_path, importance_path, plot_path

def run_ablation_study():
    """Run ablation study with 7 variants excluding different feature sets."""
    logger.info("Starting Broker Report MLP Regression Ablation Study with Fixed Random Seeds...")

    # Set master seed for data processing
    set_all_seeds(42)

    # Load data once for all variants
    broker_df, returns_df = load_and_prepare_data()

    # Create training data using optimized approach
    training_df = create_training_data_optimized(broker_df, returns_df)
    logger.info(f"Created {len(training_df)} training samples")

    if len(training_df) == 0:
        logger.error("No training data created. Exiting.")
        return

    # Define ablation variants
    ablation_variants = [
        ('baseline_full_features', None),  # Non-ablated baseline
        ('excluding_event_nature', 'event_nature'),
        ('excluding_thesis', 'thesis'),
        ('excluding_valuations', 'valuations'),
        ('excluding_comparative', 'comparative'),
        ('excluding_temporal_orientation', 'temporal_orientation'),
        ('excluding_information_novelty', 'information_novelty')
    ]

    # Store results for comparison
    all_results = []

    # Run each ablation variant
    for variant_name, excluded_set in ablation_variants:
        logger.info(f"\n" + "="*100)
        logger.info(f"RUNNING ABLATION VARIANT: {variant_name.upper()} (excluding {excluded_set})")
        logger.info("="*100)

        # Train MLP regression with ablation
        model, scaler, test_df, device, feature_cols, test_metrics, feature_importance = train_mlp_regression_with_epoch_tracking(
            training_df, excluded_feature_set=excluded_set, variant_name=variant_name
        )

        # Create regression prediction matrix
        pred_matrix = create_regression_prediction_matrix(broker_df, returns_df, model, scaler, device, feature_cols)

        # Save results
        matrix_path, model_path, results_path, importance_path, plot_path = save_results(
            pred_matrix, model, scaler, feature_cols, test_metrics, feature_importance, variant_name=variant_name
        )

        # Store results for comparison
        variant_result = {
            'variant': variant_name,
            'excluded_set': excluded_set if excluded_set else 'none',
            'num_features': len(feature_cols),
            'excluded_features': len(FEATURE_SETS[excluded_set]) if excluded_set else 0,
            'test_rmse': test_metrics['rmse'],
            'test_mae': test_metrics['mae'],
            'test_r2': test_metrics['r2'],
            'test_pearson': test_metrics['pearson_corr'],
            'test_spearman': test_metrics['spearman_corr'],
            'matrix_path': matrix_path,
            'model_path': model_path
        }
        all_results.append(variant_result)

        logger.info(f"Completed {variant_name} - RMSE: {test_metrics['rmse']:.6f}, R²: {test_metrics['r2']:.4f}")

    # Create comparative summary
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_path = f"/home/lichenhui/broker_report_mlp_ablation_summary_{timestamp}.csv"
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(summary_path, index=False)

    # Print final comparison
    print("\n" + "="*120)
    print("ABLATION STUDY RESULTS SUMMARY")
    print("="*120)
    print(f"{'Variant':<30} {'Features':<10} {'Excluded':<10} {'RMSE':<12} {'R²':<8} {'Pearson':<8} {'Spearman':<8}")
    print("-"*120)

    for result in all_results:
        print(f"{result['variant']:<30} {result['num_features']:<10} {result['excluded_features']:<10} "
              f"{result['test_rmse']:<12.6f} {result['test_r2']:<8.4f} {result['test_pearson']:<8.4f} {result['test_spearman']:<8.4f}")

    print("="*120)
    print(f"Summary saved: {summary_path}")
    print("\nAblation study completed successfully!")

    return all_results

def main():
    """Main execution - runs ablation study."""
    return run_ablation_study()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
