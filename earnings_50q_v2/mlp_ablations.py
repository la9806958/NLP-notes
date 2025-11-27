#!/usr/bin/env python3
"""
MLP pipeline with ablation studies using merged_earnings_features.csv
Runs multiple feature configurations to test importance of different question groups.

Feature groups:
- q1-10: First 10 questions
- q11-20: Questions 11-20
- q21-30: Questions 21-30
- q31-40: Questions 31-40
- q41-50: Questions 41-50
- last_13: task3_score_1 through task3_score_10, task2_first_match_score,
           task1_response_bullish_count, task1_response_bearish_count

Ablation experiments:
1. All features (q1-50 + last_13) - baseline
2. Omit q1-10
3. Omit q11-20
4. Omit q21-30
5. Omit q31-40
6. Omit q41-50
7. Omit last_13
8. Only q1-10
9. Only q11-20
10. Only q21-30
11. Only q31-40
12. Only q41-50
13. Only last_13

Scaling: Standard scaling (z-score normalization) per feature (fit on train, apply to val/test)
Train/Val/Test split: 70% train, 30% validation (from train_val period), separate test windows
"""

import logging
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss, confusion_matrix
from sklearn.utils import resample
from datetime import datetime
import warnings
import multiprocessing as mp
from functools import partial
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import pearsonr, spearmanr
warnings.filterwarnings('ignore')

# ────────────────────────────── Logging ──────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ────────────────────────────── Reproducibility ──────────────────────────────
def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make cuDNN deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed} for reproducibility")

# ────────────────────────────── Constants ────────────────────────────
K = 3
CLASS_LABELS = [-1, 0, 1]
HOURS_FWD = 40 # five trading days (prediction period)
EMBARGO_HOURS = HOURS_FWD  # Embargo period equals prediction period
TRAIN_START_DATE = "2018-01-01"
FINAL_END_DATE = "2024-12-30"  # End of all available data
ROLLING_WINDOW_MONTHS = 3  # Roll forward every 3 months

# Feature configurations for ablation studies
FEATURE_CONFIGS = {
    'all_features': {
        'description': 'All features (q1-50 + last_13)',
        'features': [f'q{i}' for i in range(1, 51)] + [
            'task3_score_1', 'task3_score_2', 'task3_score_3', 'task3_score_4',
            'task3_score_5', 'task3_score_6', 'task3_score_7', 'task3_score_8',
            'task3_score_9', 'task3_score_10', 'task2_first_match_score',
            'task1_response_bullish_count', 'task1_response_bearish_count'
        ]
    },
    'omit_q1_10': {
        'description': 'Omit q1-10',
        'features': [f'q{i}' for i in range(11, 51)] + [
            'task3_score_1', 'task3_score_2', 'task3_score_3', 'task3_score_4',
            'task3_score_5', 'task3_score_6', 'task3_score_7', 'task3_score_8',
            'task3_score_9', 'task3_score_10', 'task2_first_match_score',
            'task1_response_bullish_count', 'task1_response_bearish_count'
        ]
    },
    'omit_q11_20': {
        'description': 'Omit q11-20',
        'features': [f'q{i}' for i in range(1, 11)] + [f'q{i}' for i in range(21, 51)] + [
            'task3_score_1', 'task3_score_2', 'task3_score_3', 'task3_score_4',
            'task3_score_5', 'task3_score_6', 'task3_score_7', 'task3_score_8',
            'task3_score_9', 'task3_score_10', 'task2_first_match_score',
            'task1_response_bullish_count', 'task1_response_bearish_count'
        ]
    },
    'omit_q21_30': {
        'description': 'Omit q21-30',
        'features': [f'q{i}' for i in range(1, 21)] + [f'q{i}' for i in range(31, 51)] + [
            'task3_score_1', 'task3_score_2', 'task3_score_3', 'task3_score_4',
            'task3_score_5', 'task3_score_6', 'task3_score_7', 'task3_score_8',
            'task3_score_9', 'task3_score_10', 'task2_first_match_score',
            'task1_response_bullish_count', 'task1_response_bearish_count'
        ]
    },
    'omit_q31_40': {
        'description': 'Omit q31-40',
        'features': [f'q{i}' for i in range(1, 31)] + [f'q{i}' for i in range(41, 51)] + [
            'task3_score_1', 'task3_score_2', 'task3_score_3', 'task3_score_4',
            'task3_score_5', 'task3_score_6', 'task3_score_7', 'task3_score_8',
            'task3_score_9', 'task3_score_10', 'task2_first_match_score',
            'task1_response_bullish_count', 'task1_response_bearish_count'
        ]
    },
    'omit_q41_50': {
        'description': 'Omit q41-50',
        'features': [f'q{i}' for i in range(1, 41)] + [
            'task3_score_1', 'task3_score_2', 'task3_score_3', 'task3_score_4',
            'task3_score_5', 'task3_score_6', 'task3_score_7', 'task3_score_8',
            'task3_score_9', 'task3_score_10', 'task2_first_match_score',
            'task1_response_bullish_count', 'task1_response_bearish_count'
        ]
    },
    'omit_last_13': {
        'description': 'Omit last 13 scores',
        'features': [f'q{i}' for i in range(1, 51)]
    },
    'only_q1_10': {
        'description': 'Only q1-10',
        'features': [f'q{i}' for i in range(1, 11)]
    },
    'only_q11_20': {
        'description': 'Only q11-20',
        'features': [f'q{i}' for i in range(11, 21)]
    },
    'only_q21_30': {
        'description': 'Only q21-30',
        'features': [f'q{i}' for i in range(21, 31)]
    },
    'only_q31_40': {
        'description': 'Only q31-40',
        'features': [f'q{i}' for i in range(31, 41)]
    },
    'only_q41_50': {
        'description': 'Only q41-50',
        'features': [f'q{i}' for i in range(41, 51)]
    },
    'only_last_13': {
        'description': 'Only last 13 scores',
        'features': [
            'task3_score_1', 'task3_score_2', 'task3_score_3', 'task3_score_4',
            'task3_score_5', 'task3_score_6', 'task3_score_7', 'task3_score_8',
            'task3_score_9', 'task3_score_10', 'task2_first_match_score',
            'task1_response_bullish_count', 'task1_response_bearish_count'
        ]
    }
}

# ────────────────────────────── Data Prep ────────────────────────────
def load_and_prepare_data():
    logger.info("Loading data...")

    # Load merged earnings features
    earnings_df = pd.read_csv('/home/lichenhui/merged_earnings_features.csv')

    # Parse the date/timestamp
    if 'et_timestamp' in earnings_df.columns:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FutureWarning, message='.*un-recognized timezone.*')
            earnings_df['precise_date'] = pd.to_datetime(earnings_df['et_timestamp'], errors='coerce', utc=True)
        earnings_df = earnings_df.dropna(subset=['precise_date'])
        earnings_df['precise_date'] = earnings_df['precise_date'].dt.tz_localize(None)
    elif 'date' in earnings_df.columns:
        earnings_df['precise_date'] = pd.to_datetime(earnings_df['date'], errors='coerce')
        earnings_df = earnings_df.dropna(subset=['precise_date'])
    else:
        raise ValueError("No date or et_timestamp column found in earnings data")

    # Load hourly returns data
    returns_df = pd.read_csv('/home/lichenhui/hourly_close_to_close_returns_matrix.csv')

    # Rename datetime column to DateTime for consistency
    if 'datetime' in returns_df.columns:
        returns_df.rename(columns={'datetime': 'DateTime'}, inplace=True)

    returns_df['DateTime'] = pd.to_datetime(returns_df['DateTime'])

    logger.info(f"Loaded {len(earnings_df)} earnings records")
    logger.info(f"Loaded {len(returns_df)} hourly return records")

    return earnings_df, returns_df

def process_earnings_chunk(chunk_data):
    earnings_chunk, returns_df, all_feature_cols = chunk_data
    training_data = []
    for _, row in earnings_chunk.iterrows():
        ticker = row['ticker']
        earnings_date = row['precise_date']

        # Skip if ticker not in returns matrix
        if ticker not in returns_df.columns:
            continue

        future_returns = returns_df[returns_df['DateTime'] > earnings_date]
        if len(future_returns) <= HOURS_FWD:
            continue
        ticker_returns = future_returns[ticker].iloc[1:HOURS_FWD+1]
        valid_returns = ticker_returns.dropna()
        if len(valid_returns) < 15:
            continue
        cumulative_return = (1 + valid_returns).prod() - 1
        sample = {'ticker': ticker, 'date': earnings_date, 'cumulative_return': cumulative_return}
        for col in all_feature_cols:
            sample[col] = row[col]
        training_data.append(sample)
    return training_data

def create_training_data(earnings_df, returns_df):
    # Get all possible features
    all_feature_cols = FEATURE_CONFIGS['all_features']['features']

    num_processes = 20
    chunk_size = len(earnings_df) // num_processes + 1
    chunks = [(earnings_df.iloc[i:i+chunk_size], returns_df, all_feature_cols)
              for i in range(0, len(earnings_df), chunk_size)]
    with mp.Pool(processes=num_processes) as pool:
        chunk_results = pool.map(process_earnings_chunk, chunks)
    training_data = [row for chunk in chunk_results for row in chunk]
    training_df = pd.DataFrame(training_data)

    # Sort by stable key to ensure deterministic order for train/val splits
    training_df = training_df.sort_values(['date', 'ticker']).reset_index(drop=True)
    logger.info("Sorted training data by ['date', 'ticker'] for deterministic splits")

    return training_df

def create_3_class_targets(returns):
    # Split by fixed absolute thresholds
    # Class 0: returns < -0.05 (negative)
    # Class 1: -0.05 <= returns <= 0.05 (neutral)
    # Class 2: returns > 0.05 (positive)
    lower_threshold = -0.025
    upper_threshold = 0.025
    classes = np.zeros(len(returns), dtype=int)
    classes[returns < lower_threshold] = 0
    classes[(returns >= lower_threshold) & (returns <= upper_threshold)] = 1
    classes[returns > upper_threshold] = 2
    return classes, lower_threshold, upper_threshold

def add_embargo_period(returns_df, end_date, embargo_hours):
    """
    Calculate the embargo end date by finding the date that is embargo_hours away in the returns data.
    """
    if embargo_hours >= 0:
        # Forward embargo: find date embargo_hours after end_date
        future_dates = returns_df[returns_df['DateTime'] > end_date]['DateTime']
        if len(future_dates) > embargo_hours:
            embargo_end_date = future_dates.iloc[embargo_hours]
        else:
            embargo_end_date = future_dates.iloc[-1] if len(future_dates) > 0 else end_date
    else:
        # Backward embargo: find date |embargo_hours| before end_date
        past_dates = returns_df[returns_df['DateTime'] < end_date]['DateTime']
        if len(past_dates) >= abs(embargo_hours):
            embargo_end_date = past_dates.iloc[-abs(embargo_hours)]
        else:
            embargo_end_date = past_dates.iloc[0] if len(past_dates) > 0 else end_date
    return embargo_end_date

# ────────────────────────────── Scaling ──────────────────────────────
class StandardScaler:
    """Mean centering only: X_scaled = (X - mean)"""
    def __init__(self):
        self.fitted = False
        self.means = None

    def fit_transform(self, X):
        self.means = np.mean(X, axis=0)
        self.fitted = True
        return X - self.means

    def transform(self, X):
        if not self.fitted:
            raise ValueError("Scaler not fitted")
        return X - self.means

# ────────────────────────────── CORAL Loss ──────────────────────────────
class CoralLoss(nn.Module):
    """
    CORAL (Consistent Rank Logits) ordinal loss.
    Predicts K-1 cumulative logits for K classes with ordinal structure.
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels, num_classes=3):
        batch_size = labels.size(0)
        num_thresholds = num_classes - 1

        # Create cumulative targets
        levels = labels.view(-1, 1).repeat(1, num_thresholds)
        thresholds = torch.arange(num_thresholds, device=labels.device).view(1, -1).repeat(batch_size, 1)
        cumulative_targets = (levels > thresholds).float()

        # Binary cross-entropy loss for each threshold
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, cumulative_targets, reduction='mean'
        )

        return loss

# ────────────────────────────── Model ──────────────────────────────
class MLP(nn.Module):
    def __init__(self, input_size, hidden1=256, hidden2=128, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        # For CORAL: output K-1 cumulative logits
        self.fc3 = nn.Linear(hidden2, num_classes - 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

    def predict_proba(self, x):
        """Convert cumulative logits to class probabilities."""
        logits = self.forward(x)
        # Apply sigmoid to get cumulative probabilities
        cumulative_probs = torch.sigmoid(logits)

        # Convert cumulative to per-class probabilities
        batch_size = cumulative_probs.size(0)
        probs = torch.zeros(batch_size, self.num_classes, device=x.device)

        # First class: 1 - cumulative_probs[:, 0]
        probs[:, 0] = 1.0 - cumulative_probs[:, 0]

        # Middle classes: differences
        for i in range(1, self.num_classes - 1):
            probs[:, i] = cumulative_probs[:, i-1] - cumulative_probs[:, i]

        # Last class: cumulative_probs[:, -1]
        probs[:, -1] = cumulative_probs[:, -1]

        # Clamp probabilities to [0, 1] to avoid numerical precision issues
        probs = torch.clamp(probs, min=0.0, max=1.0)

        # Renormalize to ensure they sum to 1
        probs = probs / probs.sum(dim=1, keepdim=True)

        return probs

# ────────────────────────────── Metrics ──────────────────────────────
def print_confusion_matrix(targets, preds, split_name=""):
    """Print confusion matrix in a readable format."""
    cm = confusion_matrix(targets, preds)
    logger.info(f"{split_name} Confusion Matrix:")
    logger.info(f"                Predicted")
    logger.info(f"              0 (Neg)  1 (Neu)  2 (Pos)")
    for i, label in enumerate(['0 (Neg)', '1 (Neu)', '2 (Pos)']):
        logger.info(f"Actual {label}  {cm[i, 0]:6d}   {cm[i, 1]:6d}   {cm[i, 2]:6d}")

def calculate_metrics(model, loader, device, pi_train=None, pi_test=None, returns=None):
    model.eval()
    preds, probs, targets = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            p = model.predict_proba(xb.to(device)).cpu().numpy()
            preds.extend(np.argmax(p, 1))
            probs.extend(p)
            targets.extend(yb.numpy())
    preds, probs, targets = np.array(preds), np.array(probs), np.array(targets)

    # Calculate soft predictions (weighted probabilities)
    soft_preds = (probs[:, 0] * (-1) + probs[:, 1] * 0 + probs[:, 2] * 1)

    # Calculate correlations with actual returns if provided
    correlations = {}
    if returns is not None:
        try:
            pearson_corr, pearson_p = pearsonr(soft_preds, returns)
            spearman_corr, spearman_p = spearmanr(soft_preds, returns)
            correlations = {
                'pearson_corr': pearson_corr,
                'pearson_p': pearson_p,
                'spearman_corr': spearman_corr,
                'spearman_p': spearman_p
            }
        except:
            correlations = {
                'pearson_corr': 0.0,
                'pearson_p': 1.0,
                'spearman_corr': 0.0,
                'spearman_p': 1.0
            }

    return ((preds == targets).mean(),
            balanced_accuracy_score(targets, preds),
            f1_score(targets, preds, average='weighted'),
            log_loss(targets, probs),
            correlations,
            preds)  # Also return predictions for confusion matrix

def train_single_model(train_df, val_df, device, feature_cols, max_epochs=1000, patience=10):
    """
    Train a single model on given train/val splits with specified features.
    """
    X_train = train_df[feature_cols].values
    y_train, lower, upper = create_3_class_targets(train_df['cumulative_return'])

    # Apply same fixed thresholds to val set
    X_val = val_df[feature_cols].values
    y_val, _, _ = create_3_class_targets(val_df['cumulative_return'])

    # Scale with standard scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.astype(np.float32))
    X_val = scaler.transform(X_val.astype(np.float32))

    # Tensors
    X_train_t, y_train_t = torch.FloatTensor(X_train), torch.LongTensor(y_train)
    X_val_t, y_val_t = torch.FloatTensor(X_val), torch.LongTensor(y_val)

    # Worker seed function for reproducibility
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        import random
        random.seed(worker_seed)

    # Generator for reproducible shuffling
    g = torch.Generator()
    g.manual_seed(42)

    # Loaders
    batch_size = 256
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True,
                               worker_init_fn=seed_worker, generator=g)
    train_eval_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=False)  # Non-shuffled for metrics
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size)

    # Model - input size based on number of features
    input_size = len(feature_cols)
    model = MLP(input_size=input_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # CORAL ordinal loss
    criterion = CoralLoss()

    # Train loop
    best_val_loss_value = float('inf')
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        running = 0.0
        nobs = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb.to(device))
            loss = criterion(out, yb.to(device))
            loss.backward()
            optimizer.step()
            bs = yb.size(0)
            running += loss.item() * bs
            nobs += bs
        train_ce = running / max(nobs, 1)

        # Calculate train metrics (for correlation tracking) - use non-shuffled loader
        train_acc, train_bal_acc, train_f1, train_loss, train_corr, train_preds = calculate_metrics(
            model, train_eval_loader, device, returns=train_df['cumulative_return'].values
        )

        # Calculate validation metrics
        val_acc, val_bal_acc, val_f1, val_loss, val_corr, val_preds = calculate_metrics(
            model, val_loader, device, returns=val_df['cumulative_return'].values
        )

        # Log progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            logger.info(f"  Epoch {epoch+1:03d} - TrainCE={train_ce:.5f} | "
                       f"Train: Pearson={train_corr.get('pearson_corr', 0):.4f} Spearman={train_corr.get('spearman_corr', 0):.4f} | "
                       f"Val: Loss={val_loss:.4f} Pearson={val_corr.get('pearson_corr', 0):.4f} Spearman={val_corr.get('spearman_corr', 0):.4f}")

        # Early stopping on val loss
        if val_loss < best_val_loss_value:
            best_val_loss_value = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        if patience_counter >= patience:
            logger.info(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)

    # Print final confusion matrices for train and val
    logger.info("\nFinal Model Performance:")
    train_acc, train_bal_acc, train_f1, train_loss, train_corr, train_preds = calculate_metrics(
        model, train_eval_loader, device, returns=train_df['cumulative_return'].values
    )
    val_acc, val_bal_acc, val_f1, val_loss, val_corr, val_preds = calculate_metrics(
        model, val_loader, device, returns=val_df['cumulative_return'].values
    )
    print_confusion_matrix(y_train, train_preds, "TRAIN")
    print_confusion_matrix(y_val, val_preds, "VAL")

    return model, scaler, best_val_loss_value

def train_mlp_with_rolling_windows(training_df, returns_df, feature_cols, config_name):
    """
    Train models using rolling 3-month windows with embargo periods.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Generate rolling window dates (every 3 months)
    start_date = pd.to_datetime(TRAIN_START_DATE)
    end_date = pd.to_datetime(FINAL_END_DATE)

    # First test window starts after sufficient training data
    first_test_start = start_date + pd.DateOffset(years=4)

    # Generate test window start dates every 3 months
    test_window_starts = pd.date_range(
        start=first_test_start,
        end=end_date,
        freq=f'{ROLLING_WINDOW_MONTHS}MS'  # MS = month start
    )

    all_models = []
    all_test_results = []

    logger.info("="*80)
    logger.info(f"ROLLING WINDOW TRAINING - {config_name}")
    logger.info("="*80)
    logger.info(f"Feature configuration: {FEATURE_CONFIGS[config_name]['description']}")
    logger.info(f"Number of features: {len(feature_cols)}")
    logger.info(f"Training windows: Every {ROLLING_WINDOW_MONTHS} months")
    logger.info(f"Embargo period: {EMBARGO_HOURS} hours ({EMBARGO_HOURS/24:.1f} days)")
    logger.info(f"Number of windows: {len(test_window_starts)}")
    logger.info("="*80)

    for i, test_window_start in enumerate(test_window_starts):
        logger.info("")
        logger.info(f"{'='*80}")
        logger.info(f"WINDOW {i+1}/{len(test_window_starts)}")
        logger.info(f"{'='*80}")

        # Calculate test window end (3 months after start)
        test_window_end = test_window_start + pd.DateOffset(months=ROLLING_WINDOW_MONTHS)

        # Calculate training end with embargo
        train_val_end = add_embargo_period(returns_df, test_window_start, -EMBARGO_HOURS)

        logger.info(f"Train/Val period: {start_date.date()} to {train_val_end.date()}")
        logger.info(f"Test period: {test_window_start.date()} to {test_window_end.date()}")

        # Split data
        train_val_data = training_df[
            (training_df['date'] >= start_date) &
            (training_df['date'] < train_val_end)
        ].copy()

        test_data = training_df[
            (training_df['date'] >= test_window_start) &
            (training_df['date'] < test_window_end)
        ].copy()

        if len(test_data) == 0:
            logger.info(f"No test data in this window, skipping...")
            continue

        # Further split train_val_data into train (70%) and val (30%) based on time
        # Sort by date to ensure temporal ordering
        train_val_data = train_val_data.sort_values('date').reset_index(drop=True)

        # Find the 70% time point
        train_end_idx = int(len(train_val_data) * 0.7)
        train_end_date = train_val_data.iloc[train_end_idx]['date']

        # Apply embargo period between train and val
        val_start_date = add_embargo_period(returns_df, train_end_date, EMBARGO_HOURS)

        # Split based on dates with embargo
        train_data = train_val_data[train_val_data['date'] < train_end_date].copy()
        val_data = train_val_data[train_val_data['date'] >= val_start_date].copy()

        logger.info(f"Train period: {start_date.date()} to {train_end_date.date()}")
        logger.info(f"Embargo period: {train_end_date.date()} to {val_start_date.date()} ({EMBARGO_HOURS} hours)")
        logger.info(f"Val period: {val_start_date.date()} to {train_val_end.date()}")
        logger.info(f"Train size: {len(train_data)}, Val size: {len(val_data)}, Test size: {len(test_data)}")

        # Log class distributions
        def describe_split(name, df):
            y, _, _ = create_3_class_targets(df['cumulative_return'])
            cc = np.bincount(y, minlength=3)
            logger.info(f"{name}: n={len(y)}, class_counts={cc}, p={cc/cc.sum() if cc.sum()>0 else cc}")

        describe_split("TRAIN", train_data)
        describe_split("VAL", val_data)
        describe_split("TEST", test_data)

        # Train model for this window
        logger.info("Training model...")
        model, scaler, best_val_loss = train_single_model(train_data, val_data, device, feature_cols)

        # Evaluate on test set
        X_test = test_data[feature_cols].values.astype(np.float32)
        y_test, _, _ = create_3_class_targets(test_data['cumulative_return'])
        X_test_scaled = scaler.transform(X_test)
        X_test_t = torch.FloatTensor(X_test_scaled)
        y_test_t = torch.LongTensor(y_test)
        test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=256)

        test_acc, test_bal_acc, test_f1, test_loss, test_corr, test_preds = calculate_metrics(
            model, test_loader, device, returns=test_data['cumulative_return'].values
        )

        logger.info(f"Test Results: Acc={test_acc:.4f}, BalAcc={test_bal_acc:.4f}, F1={test_f1:.4f}, Loss={test_loss:.4f}")
        logger.info(f"  Pearson corr: {test_corr.get('pearson_corr', 0):.4f} (p={test_corr.get('pearson_p', 1):.4e})")
        logger.info(f"  Spearman corr: {test_corr.get('spearman_corr', 0):.4f} (p={test_corr.get('spearman_p', 1):.4e})")
        print_confusion_matrix(y_test, test_preds, "TEST")

        # Store model and results
        all_models.append({
            'model': model,
            'scaler': scaler,
            'test_start': test_window_start,
            'test_end': test_window_end,
            'train_val_end': train_val_end,
            'metrics': {
                'test_acc': test_acc,
                'test_bal_acc': test_bal_acc,
                'test_f1': test_f1,
                'test_loss': test_loss,
                'test_corr': test_corr
            }
        })
        all_test_results.append(test_data)

    logger.info("")
    logger.info("="*80)
    logger.info(f"ROLLING WINDOW TRAINING COMPLETE - {config_name}")
    logger.info("="*80)
    logger.info(f"Trained {len(all_models)} models")

    # Calculate aggregate statistics
    if len(all_models) > 0:
        avg_acc = np.mean([m['metrics']['test_acc'] for m in all_models])
        avg_bal_acc = np.mean([m['metrics']['test_bal_acc'] for m in all_models])
        avg_f1 = np.mean([m['metrics']['test_f1'] for m in all_models])
        avg_loss = np.mean([m['metrics']['test_loss'] for m in all_models])
        avg_pearson = np.mean([m['metrics']['test_corr'].get('pearson_corr', 0) for m in all_models])
        avg_spearman = np.mean([m['metrics']['test_corr'].get('spearman_corr', 0) for m in all_models])

        logger.info(f"Average Test Metrics Across All Windows:")
        logger.info(f"  Acc={avg_acc:.4f}, BalAcc={avg_bal_acc:.4f}, F1={avg_f1:.4f}, Loss={avg_loss:.4f}")
        logger.info(f"  Pearson corr: {avg_pearson:.4f}")
        logger.info(f"  Spearman corr: {avg_spearman:.4f}")

    return all_models, all_test_results

def run_ablation_study(training_df, returns_df, config_name):
    """Run training for a specific feature configuration."""
    logger.info("\n" + "="*80)
    logger.info(f"STARTING ABLATION: {config_name}")
    logger.info(f"Description: {FEATURE_CONFIGS[config_name]['description']}")
    logger.info("="*80)

    feature_cols = FEATURE_CONFIGS[config_name]['features']
    all_models, all_test_results = train_mlp_with_rolling_windows(
        training_df, returns_df, feature_cols, config_name
    )

    # Calculate aggregate metrics
    if len(all_models) > 0:
        avg_metrics = {
            'config': config_name,
            'description': FEATURE_CONFIGS[config_name]['description'],
            'num_features': len(feature_cols),
            'num_windows': len(all_models),
            'avg_acc': np.mean([m['metrics']['test_acc'] for m in all_models]),
            'avg_bal_acc': np.mean([m['metrics']['test_bal_acc'] for m in all_models]),
            'avg_f1': np.mean([m['metrics']['test_f1'] for m in all_models]),
            'avg_loss': np.mean([m['metrics']['test_loss'] for m in all_models]),
            'avg_pearson': np.mean([m['metrics']['test_corr'].get('pearson_corr', 0) for m in all_models]),
            'avg_spearman': np.mean([m['metrics']['test_corr'].get('spearman_corr', 0) for m in all_models])
        }
    else:
        avg_metrics = {
            'config': config_name,
            'description': FEATURE_CONFIGS[config_name]['description'],
            'num_features': len(feature_cols),
            'num_windows': 0,
            'avg_acc': 0.0,
            'avg_bal_acc': 0.0,
            'avg_f1': 0.0,
            'avg_loss': 0.0,
            'avg_pearson': 0.0,
            'avg_spearman': 0.0
        }

    return avg_metrics, all_models

def create_weighted_prediction_matrix_rolling(earnings_df, returns_df, all_models, feature_cols, config_name):
    """
    Create datetime by ticker weighted probability matrix using rolling models.
    Each model is only used for predictions in its corresponding test window.
    Similar to simple_mlp_pipeline_50q_pre2023.py implementation.

    Args:
        earnings_df: Earnings data with features
        returns_df: Returns data with DateTime column
        all_models: List of model dictionaries from rolling window training
        feature_cols: List of feature column names used in this configuration
        config_name: Name of the configuration for logging

    Returns:
        pred_matrix: DataFrame with DateTime as index, tickers as columns, weighted predictions as values
    """
    logger.info(f"Creating weighted probability prediction matrix for {config_name}...")

    # Get earnings tickers that we can make predictions for
    earnings_tickers = set(earnings_df['ticker'].unique())
    returns_tickers = set(returns_df.columns[1:])
    common_tickers = sorted(earnings_tickers.intersection(returns_tickers))

    # Filter to common tickers
    earnings_filtered = earnings_df[earnings_df['ticker'].isin(common_tickers)].copy()

    logger.info(f"Processing {len(earnings_filtered)} earnings announcements for {len(common_tickers)} tickers")

    # Pre-sort returns_df DateTime for binary search
    returns_df_sorted = returns_df.sort_values('DateTime').reset_index(drop=True)
    datetime_index = returns_df_sorted['DateTime'].values

    # Collect all predictions from all models
    all_result_dfs = []

    for model_info in all_models:
        model = model_info['model']
        scaler = model_info['scaler']
        test_start = model_info['test_start']
        test_end = model_info['test_end']
        device = next(model.parameters()).device

        # Filter earnings to this test window
        earnings_window = earnings_filtered[
            (earnings_filtered['precise_date'] >= test_start) &
            (earnings_filtered['precise_date'] < test_end)
        ].copy()

        if len(earnings_window) == 0:
            continue

        # Check if all required features are available
        missing_features = [f for f in feature_cols if f not in earnings_window.columns]
        if missing_features:
            logger.warning(f"Missing features in earnings data: {missing_features}")
            continue

        # Vectorized feature extraction
        features_array = earnings_window[feature_cols].values.astype(np.float32)
        features_scaled = scaler.transform(features_array)

        # Batch prediction
        model.eval()
        batch_size = 512
        window_predictions = []

        with torch.no_grad():
            for i in range(0, len(features_scaled), batch_size):
                batch = features_scaled[i:i+batch_size]
                batch_tensor = torch.FloatTensor(batch).to(device)
                probs = model.predict_proba(batch_tensor).cpu().numpy()
                # Weighted predictions: Class 0 -> -1, Class 1 -> 0, Class 2 -> 1
                weighted = probs[:, 0] * (-1) + probs[:, 2] * 1
                window_predictions.append(weighted)

        window_predictions = np.concatenate(window_predictions)

        # Find target dates using vectorized searchsorted
        earnings_dates = earnings_window['precise_date'].values
        target_date_indices = np.searchsorted(datetime_index, earnings_dates, side='right')

        # Filter out cases where no future date exists
        valid_mask = target_date_indices < len(datetime_index)
        target_date_indices = target_date_indices[valid_mask]

        # Build sparse matrix data for this window
        tickers = earnings_window['ticker'].values[valid_mask]
        target_dates = datetime_index[target_date_indices]
        predictions = window_predictions[valid_mask]

        # Create result dataframe for this window
        window_result_df = pd.DataFrame({
            'DateTime': target_dates,
            'ticker': tickers,
            'prediction': predictions
        })

        all_result_dfs.append(window_result_df)

    # Combine all windows
    if len(all_result_dfs) > 0:
        result_df = pd.concat(all_result_dfs, ignore_index=True)

        # Pivot to matrix form
        pred_matrix = result_df.pivot_table(
            index='DateTime',
            columns='ticker',
            values='prediction',
            aggfunc='first'  # In case of duplicates, take first
        )

        # Reindex to include all dates from returns_df
        all_dates = pd.Index(sorted(returns_df['DateTime'].unique()))
        pred_matrix = pred_matrix.reindex(index=all_dates, columns=common_tickers)

        # Count non-null predictions
        total_predictions = pred_matrix.notna().sum().sum()
        logger.info(f"Generated {total_predictions} weighted predictions from {len(all_models)} rolling models")
    else:
        # Empty matrix if no predictions
        all_dates = pd.Index(sorted(returns_df['DateTime'].unique()))
        pred_matrix = pd.DataFrame(index=all_dates, columns=common_tickers)
        logger.info("No predictions generated (no data in test windows)")

    return pred_matrix

def main():
    """Main execution with all ablation studies."""
    # Set random seed for reproducibility
    set_seed(42)

    logger.info("Starting MLP Ablation Study Pipeline with Merged Features...")

    # Load data
    earnings_df, returns_df = load_and_prepare_data()

    # Create training data
    training_df = create_training_data(earnings_df, returns_df)
    logger.info(f"Created training data with {len(training_df)} samples")

    # Run all ablation studies
    all_results = []
    all_prediction_matrices = {}

    # Order of experiments
    experiment_order = [
        'all_features',      # Baseline
        'omit_q1_10',        # Ablations
        'omit_q11_20',
        'omit_q21_30',
        'omit_q31_40',
        'omit_q41_50',
        'omit_last_13',
        'only_q1_10',        # Individual groups
        'only_q11_20',
        'only_q21_30',
        'only_q31_40',
        'only_q41_50',
        'only_last_13'
    ]

    for config_name in experiment_order:
        avg_metrics, models = run_ablation_study(training_df, returns_df, config_name)
        all_results.append(avg_metrics)

        # Create prediction matrix for this configuration
        if len(models) > 0:
            feature_cols = FEATURE_CONFIGS[config_name]['features']
            pred_matrix = create_weighted_prediction_matrix_rolling(
                earnings_df, returns_df, models, feature_cols, config_name
            )
            all_prediction_matrices[config_name] = pred_matrix

    # Create summary dataframe
    summary_df = pd.DataFrame(all_results)

    # Save summary and prediction matrices
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_path = f"/home/lichenhui/ablation_summary_{timestamp}.csv"
    summary_df.to_csv(summary_path, index=False)

    # Save prediction matrices for each configuration
    matrix_paths = {}
    for config_name, pred_matrix in all_prediction_matrices.items():
        matrix_path = f"/home/lichenhui/ablation_predictions_2_{config_name}_{timestamp}.csv"
        pred_matrix.to_csv(matrix_path)
        matrix_paths[config_name] = matrix_path
        logger.info(f"Saved prediction matrix for {config_name}: {matrix_path}")
        logger.info(f"  Shape: {pred_matrix.shape}, Non-null predictions: {pred_matrix.notna().sum().sum()}")

    # Print summary
    print("\n" + "="*100)
    print("ABLATION STUDY SUMMARY")
    print("="*100)
    print(summary_df.to_string(index=False))
    print("="*100)
    print(f"\nSummary saved to: {summary_path}")
    print(f"\nPrediction matrices saved ({len(matrix_paths)} configurations):")
    for config_name, path in matrix_paths.items():
        pred_matrix = all_prediction_matrices[config_name]
        print(f"  {config_name}: {path}")
        print(f"    Shape: {pred_matrix.shape}, Non-null: {pred_matrix.notna().sum().sum()}")
    print("="*100)

    logger.info("Ablation study pipeline completed successfully!")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
