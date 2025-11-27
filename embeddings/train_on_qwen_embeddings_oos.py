#!/usr/bin/env python3
"""
Train MLP regressor on Qwen embeddings for OOS prediction

Given existing QWEN embeddings saved down, this script:

This script:
1. Loads training embeddings from earnings_qwen_analyst_target_output
2. Trains an MLP regressor with proper regularization
3. Saves the trained model and scalers
4. Loads OOS embeddings from earnings_qwen_oos_embeddings_output
5. Makes predictions and evaluates OOS performance

"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import matthews_corrcoef
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import warnings

# Suppress FutureWarnings
warnings.filterwarnings('ignore', category=FutureWarning)


class MLPRegressor(nn.Module):
    """MLP for predicting returns from embeddings

    Heavily regularized architecture to prevent severe overfitting:
    - Single hidden layer [128] (reduced from [256, 128])
    - High dropout rate (0.5)
    - Dropout before output layer
    - Layer normalization for stability
    """

    def __init__(self, input_dim: int, hidden_dims = [128], dropout_rate=0.5):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))  # Layer norm for stability
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))  # Dropout after every layer

            prev_dim = hidden_dim

        # Dropout before output layer
        layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(prev_dim, 1))

        self.model = nn.Sequential(*layers)

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/He initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.model(x).squeeze(-1)


class OOSPredictor:
    """Handles training and prediction for OOS evaluation"""

    def __init__(self,
                 train_dir: str = "earnings_qwen_oos_embeddings_output",
                 oos_dir: str = "earnings_qwen_oos_embeddings_output",
                 output_dir: str = "earnings_qwen_oos_predictions_output"):
        self.train_dir = train_dir
        self.oos_dir = oos_dir
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Scalers
        self.x_scaler = StandardScaler()
        self.y_scaler = RobustScaler()

        # Model
        self.model = None

        print(f"Using device: {self.device}")
        print(f"Training directory: {self.train_dir}")
        print(f"OOS directory: {self.oos_dir}")
        print(f"Output directory: {self.output_dir}")

    def load_training_data(self):
        """Load training embeddings and targets"""
        print("\n" + "="*60)
        print("LOADING TRAINING DATA")
        print("="*60)

        # Load embeddings
        embeddings_path = os.path.join(self.train_dir, "in_sample_embeddings_all.npy")
        print(f"Loading embeddings from: {embeddings_path}")
        embeddings = np.load(embeddings_path)
        print(f"Training embeddings shape: {embeddings.shape}")

        # Load original CSV (for ticker and et_timestamp)
        csv_path = os.path.join(self.train_dir, "in_sample_metadata_all.csv")
        print(f"Loading data from: {csv_path}")
        df = pd.read_csv(csv_path)

        # Verify shapes match FIRST
        if len(embeddings) != len(df):
            raise ValueError(f"Mismatch: {len(embeddings)} embeddings vs {len(df)} rows in CSV. "
                           f"Embeddings must be in same order as CSV rows!")

        # Load new target CSV
        target_csv_path = "data/earnings_calls_new_target.csv"
        print(f"Loading targets from: {target_csv_path}")
        df_targets = pd.read_csv(target_csv_path)
        print(f"Target CSV has {len(df_targets)} rows")

        # Add index to preserve original row order
        df = df.reset_index(drop=False).rename(columns={'index': 'original_idx'})

        # Standardize et_timestamp formats for both dataframes
        print("Standardizing timestamp formats...")
        df['et_timestamp'] = pd.to_datetime(df['et_timestamp'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
        df_targets['et_timestamp'] = pd.to_datetime(df_targets['et_timestamp'], format='mixed', errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

        # Join on ticker and et_timestamp to get return_T1_to_T40
        print(f"Joining on 'ticker' and 'et_timestamp' to get return_T1_to_T40...")
        df = df.merge(df_targets[['ticker', 'et_timestamp', 'return_T1_to_T40']],
                      on=['ticker', 'et_timestamp'],
                      how='left',
                      suffixes=('', '_new'))

        # If there's a duplicate column from merge, use the new one
        if 'return_T1_to_T40_new' in df.columns:
            df['return_T1_to_T40'] = df['return_T1_to_T40_new']
            df = df.drop(columns=['return_T1_to_T40_new'])

        print(f"After join: {len(df)} rows")

        # Check if merge created duplicates (should match embeddings length)
        if len(df) != len(embeddings):
            print(f"WARNING: Merge created {len(df) - len(embeddings)} extra rows due to duplicate matches.")
            # Keep only the first occurrence for each original index
            df = df.drop_duplicates(subset=['original_idx'], keep='first')
            # Sort by original index to maintain order
            df = df.sort_values('original_idx').reset_index(drop=True)
            print(f"After deduplication: {len(df)} rows")

        # Final verification
        if len(df) != len(embeddings):
            raise ValueError(f"After merge and dedup: {len(df)} rows in CSV but {len(embeddings)} embeddings!")

        # Verify alignment - check required columns exist
        required_cols = ['ticker', 'et_timestamp', 'return_T1_to_T40']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in CSV: {missing_cols}")

        # Extract return_T1_to_T40 column
        target_col = "return_T1_to_T40"
        targets = df[target_col].values
        print(f"Targets shape: {targets.shape}")

        # Print sample alignment check
        print(f"\nAlignment check (first 3 rows):")
        for i in range(min(3, len(df))):
            target_str = f"{targets[i]:.4f}" if not np.isnan(targets[i]) else "NaN"
            print(f"  Row {i}: ticker={df.iloc[i]['ticker']}, "
                  f"et_timestamp={df.iloc[i]['et_timestamp']}, "
                  f"target={target_str}")

        print(f"\n✓ Embeddings and targets are aligned (both have {len(embeddings)} samples in same order)")

        # Remove NaN targets
        valid_mask = ~np.isnan(targets)
        embeddings = embeddings[valid_mask]
        targets = targets[valid_mask]

        print(f"After removing NaN: {len(embeddings)} samples")
        print(f"Target stats BEFORE winsorization - Mean: {targets.mean():.4f}, Std: {targets.std():.4f}, "
              f"Min: {targets.min():.4f}, Max: {targets.max():.4f}")

        # Winsorize targets at 0.01 and 0.99 quantiles to remove extreme outliers
        lower_bound = np.quantile(targets, 0.01)
        upper_bound = np.quantile(targets, 0.99)
        original_targets = targets.copy()
        targets = np.clip(targets, lower_bound, upper_bound)

        print(f"Target stats AFTER winsorization at [0.01, 0.99] - Mean: {targets.mean():.4f}, Std: {targets.std():.4f}, "
              f"Min: {targets.min():.4f}, Max: {targets.max():.4f}")
        print(f"Winsorization bounds: [{lower_bound:.4f}, {upper_bound:.4f}]")

        # Count how many were winsorized
        n_winsorized_low = (original_targets < lower_bound).sum()
        n_winsorized_high = (original_targets > upper_bound).sum()
        print(f"Winsorized {n_winsorized_low} targets below {lower_bound:.4f} and {n_winsorized_high} targets above {upper_bound:.4f}")

        return embeddings, targets

    def load_oos_data(self):
        """Load OOS embeddings and targets from earnings_qwen_oos_embeddings_output"""
        print("\n" + "="*60)
        print("LOADING OOS DATA")
        print("="*60)

        # Load embeddings - use oos_embeddings_with_targets.npy
        embeddings_path = os.path.join(self.oos_dir, "oos_embeddings_all.npy")
        print(f"Loading OOS embeddings from: {embeddings_path}")
        embeddings_oos = np.load(embeddings_path)
        print(f"OOS embeddings shape: {embeddings_oos.shape}")

        # Load OOS CSV metadata
        csv_path = os.path.join(self.oos_dir, "oos_metadata_all.csv")
        print(f"Loading OOS data from: {csv_path}")
        df_oos = pd.read_csv(csv_path)

        # Verify shapes match FIRST
        if len(embeddings_oos) != len(df_oos):
            raise ValueError(f"Mismatch: {len(embeddings_oos)} embeddings vs {len(df_oos)} rows in CSV. "
                           f"Embeddings must be in same order as CSV rows!")

        # Check if return_T1_to_T40 already exists in the metadata
        if 'return_T1_to_T40' not in df_oos.columns:
            # Load new target CSV if not present
            target_csv_path = "data/earnings_calls_new_target.csv"
            print(f"Loading targets from: {target_csv_path}")
            df_targets = pd.read_csv(target_csv_path)
            print(f"Target CSV has {len(df_targets)} rows")

            # Add index to preserve original row order
            df_oos = df_oos.reset_index(drop=False).rename(columns={'index': 'original_idx'})

            # Standardize et_timestamp formats for both dataframes
            print("Standardizing timestamp formats...")
            df_oos['et_timestamp'] = pd.to_datetime(df_oos['et_timestamp'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
            df_targets['et_timestamp'] = pd.to_datetime(df_targets['et_timestamp'], format='mixed', errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

            # Join on ticker and et_timestamp to get return_T1_to_T40
            print(f"Joining on 'ticker' and 'et_timestamp' to get return_T1_to_T40...")
            df_oos = df_oos.merge(df_targets[['ticker', 'et_timestamp', 'return_T1_to_T40']],
                                  on=['ticker', 'et_timestamp'],
                                  how='left',
                                  suffixes=('', '_new'))

            # If there's a duplicate column from merge, use the new one
            if 'return_T1_to_T40_new' in df_oos.columns:
                df_oos['return_T1_to_T40'] = df_oos['return_T1_to_T40_new']
                df_oos = df_oos.drop(columns=['return_T1_to_T40_new'])

            print(f"After join: {len(df_oos)} rows")

            # Check if merge created duplicates (should match embeddings length)
            if len(df_oos) != len(embeddings_oos):
                print(f"WARNING: Merge created {len(df_oos) - len(embeddings_oos)} extra rows due to duplicate matches.")
                # Keep only the first occurrence for each original index
                df_oos = df_oos.drop_duplicates(subset=['original_idx'], keep='first')
                # Sort by original index to maintain order
                df_oos = df_oos.sort_values('original_idx').reset_index(drop=True)
                print(f"After deduplication: {len(df_oos)} rows")

            # Final verification
            if len(df_oos) != len(embeddings_oos):
                raise ValueError(f"After merge and dedup: {len(df_oos)} rows in CSV but {len(embeddings_oos)} embeddings!")
        else:
            print("return_T1_to_T40 column already exists in metadata")

        # Use return_T1_to_T40 as target
        target_col = "return_T1_to_T40"

        # Verify alignment - check required columns exist
        required_cols = ['ticker', 'et_timestamp', target_col]
        missing_cols = [col for col in required_cols if col not in df_oos.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in OOS CSV: {missing_cols}")

        targets_oos = df_oos[target_col].values
        print(f"OOS targets shape: {targets_oos.shape}")

        # Print sample alignment check
        print(f"\nOOS alignment check (first 3 rows):")
        for i in range(min(3, len(df_oos))):
            target_str = f"{targets_oos[i]:.4f}" if not np.isnan(targets_oos[i]) else "NaN"
            print(f"  Row {i}: ticker={df_oos.iloc[i]['ticker']}, "
                  f"et_timestamp={df_oos.iloc[i]['et_timestamp']}, "
                  f"target={target_str}")

        print(f"\n✓ OOS embeddings and targets are aligned (both have {len(embeddings_oos)} samples in same order)")

        # Remove NaN targets
        valid_mask = ~np.isnan(targets_oos)
        embeddings_oos = embeddings_oos[valid_mask]
        targets_oos = targets_oos[valid_mask]
        df_oos = df_oos[valid_mask].reset_index(drop=True)

        print(f"After removing NaN: {len(embeddings_oos)} samples")
        print(f"OOS target stats - Mean: {targets_oos.mean():.4f}, Std: {targets_oos.std():.4f}, "
              f"Min: {targets_oos.min():.4f}, Max: {targets_oos.max():.4f}")
        print("(OOS targets are NOT winsorized - kept at original values)")

        return embeddings_oos, targets_oos, df_oos

    def train_mlp(self, embeddings, targets, epochs=100, batch_size=64, lr=5e-4, val_split=0.2):
        """Train MLP with improved training loop"""
        print("\n" + "="*60)
        print("TRAINING MLP REGRESSOR")
        print("="*60)

        # Split into train/val
        n_samples = len(embeddings)
        n_val = int(n_samples * val_split)
        n_train = n_samples - n_val

        # Shuffle
        indices = np.random.RandomState(42).permutation(n_samples)
        train_idx, val_idx = indices[:n_train], indices[n_train:]

        X_train, X_val = embeddings[train_idx], embeddings[val_idx]
        y_train, y_val = targets[train_idx], targets[val_idx]

        print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

        # Fit scalers on training data ONLY
        print("\nFitting scalers on training data...")
        X_train_scaled = self.x_scaler.fit_transform(X_train)
        X_val_scaled = self.x_scaler.transform(X_val)

        y_train_scaled = self.y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = self.y_scaler.transform(y_val.reshape(-1, 1)).flatten()

        print(f"Scaled target stats - Train mean: {y_train_scaled.mean():.4f}, std: {y_train_scaled.std():.4f}")

        # Create datasets
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_scaled),
            torch.FloatTensor(y_train_scaled)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_scaled),
            torch.FloatTensor(y_val_scaled)
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model
        self.model = MLPRegressor(input_dim=embeddings.shape[1]).to(self.device)
        print(f"\nModel architecture:")
        print(self.model)
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Optimizer with very high weight decay to prevent overfitting
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-2)

        # Learning rate scheduler - more aggressive
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        # Loss function
        criterion = nn.MSELoss()

        # L1 regularization strength
        l1_lambda = 1e-4

        # Label noise for regularization
        label_noise_std = 0.05  # Add 5% noise to targets during training

        # Training loop
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        patience = 20

        train_losses = []
        val_losses = []

        print(f"\nStarting training for {epochs} epochs...")

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_batches = 0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                # Add noise to targets during training for regularization
                if label_noise_std > 0:
                    noise = torch.randn_like(batch_y) * label_noise_std
                    batch_y_noisy = batch_y + noise
                else:
                    batch_y_noisy = batch_y

                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y_noisy)

                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss += loss.item()
                train_batches += 1

            train_loss /= train_batches
            train_losses.append(train_loss)

            # Validation phase
            self.model.eval()
            val_loss = 0
            val_batches = 0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                    predictions = self.model(batch_X)
                    loss = criterion(predictions, batch_y)

                    val_loss += loss.item()
                    val_batches += 1

            val_loss /= val_batches
            val_losses.append(val_loss)

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, "
                      f"Val Loss: {val_loss:.6f}, Best Val: {best_val_loss:.6f}")

            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\nLoaded best model with val loss: {best_val_loss:.6f}")

        # Evaluate on train and val sets
        print("\n" + "="*60)
        print("TRAIN/VAL CORRELATIONS")
        print("="*60)

        # Get predictions on train set (in original scale)
        # IMPORTANT: Create non-shuffled loader for evaluation to maintain alignment
        self.model.eval()
        train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_eval_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            train_preds_scaled = []
            for batch_X, _ in train_eval_loader:
                batch_preds = self.model(batch_X.to(self.device)).cpu().numpy()
                train_preds_scaled.append(batch_preds)
            train_preds_scaled = np.concatenate(train_preds_scaled)
        train_preds = self.y_scaler.inverse_transform(train_preds_scaled.reshape(-1, 1)).flatten()

        # Get predictions on val set (in original scale)
        with torch.no_grad():
            val_preds_scaled = []
            for batch_X, _ in val_eval_loader:
                batch_preds = self.model(batch_X.to(self.device)).cpu().numpy()
                val_preds_scaled.append(batch_preds)
            val_preds_scaled = np.concatenate(val_preds_scaled)
        val_preds = self.y_scaler.inverse_transform(val_preds_scaled.reshape(-1, 1)).flatten()

        # Compute correlations
        train_spearman, train_spearman_p = spearmanr(train_preds, y_train)
        train_pearson, train_pearson_p = pearsonr(train_preds, y_train)
        val_spearman, val_spearman_p = spearmanr(val_preds, y_val)
        val_pearson, val_pearson_p = pearsonr(val_preds, y_val)

        print(f"\nTrain Set:")
        print(f"  Spearman: {train_spearman:.4f} (p={train_spearman_p:.4e})")
        print(f"  Pearson:  {train_pearson:.4f} (p={train_pearson_p:.4e})")

        print(f"\nVal Set:")
        print(f"  Spearman: {val_spearman:.4f} (p={val_spearman_p:.4e})")
        print(f"  Pearson:  {val_pearson:.4f} (p={val_pearson_p:.4e})")

        # Plot training curves
        self._plot_training_curves(train_losses, val_losses)

        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'train_spearman': train_spearman,
            'train_pearson': train_pearson,
            'val_spearman': val_spearman,
            'val_pearson': val_pearson
        }

    def _plot_training_curves(self, train_losses, val_losses):
        """Plot training and validation loss curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss', alpha=0.8)
        plt.plot(val_losses, label='Val Loss', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('MLP Training Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')

        plot_path = os.path.join(self.output_dir, "mlp_training_curves.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {plot_path}")
        plt.close()

    def predict(self, embeddings):
        """Make predictions on embeddings"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_mlp first.")

        self.model.eval()

        # Scale embeddings using fitted scaler
        embeddings_scaled = self.x_scaler.transform(embeddings)

        # Predict in batches
        predictions_scaled = []
        batch_size = 256

        with torch.no_grad():
            for i in range(0, len(embeddings_scaled), batch_size):
                batch = embeddings_scaled[i:i+batch_size]
                batch_tensor = torch.FloatTensor(batch).to(self.device)
                batch_preds = self.model(batch_tensor).cpu().numpy()
                predictions_scaled.append(batch_preds)

        predictions_scaled = np.concatenate(predictions_scaled)

        # Inverse transform to original scale
        predictions = self.y_scaler.inverse_transform(
            predictions_scaled.reshape(-1, 1)
        ).flatten()

        return predictions

    def evaluate(self, predictions, targets):
        """Compute evaluation metrics"""
        print("\n" + "="*60)
        print("EVALUATION METRICS")
        print("="*60)

        # Basic statistics
        print(f"\nPrediction statistics (original):")
        print(f"  Mean: {predictions.mean():.4f}")
        print(f"  Std: {predictions.std():.4f}")
        print(f"  Min: {predictions.min():.4f}")
        print(f"  Max: {predictions.max():.4f}")
        print(f"  Unique values: {len(np.unique(predictions))}")

        print(f"\nTarget statistics (original):")
        print(f"  Mean: {targets.mean():.4f}")
        print(f"  Std: {targets.std():.4f}")
        print(f"  Min: {targets.min():.4f}")
        print(f"  Max: {targets.max():.4f}")

        # Winsorize both predictions and targets for metric computation
        predictions_winsorized = np.clip(predictions,
                                         np.quantile(predictions, 0.01),
                                         np.quantile(predictions, 0.99))
        targets_winsorized = np.clip(targets,
                                      np.quantile(targets, 0.01),
                                      np.quantile(targets, 0.99))

        print(f"\nPrediction statistics (winsorized at [0.01, 0.99] for metrics):")
        print(f"  Mean: {predictions_winsorized.mean():.4f}")
        print(f"  Std: {predictions_winsorized.std():.4f}")
        print(f"  Min: {predictions_winsorized.min():.4f}")
        print(f"  Max: {predictions_winsorized.max():.4f}")

        print(f"\nTarget statistics (winsorized at [0.01, 0.99] for metrics):")
        print(f"  Mean: {targets_winsorized.mean():.4f}")
        print(f"  Std: {targets_winsorized.std():.4f}")
        print(f"  Min: {targets_winsorized.min():.4f}")
        print(f"  Max: {targets_winsorized.max():.4f}")

        # Regression metrics (using winsorized values)
        mse = np.mean((predictions_winsorized - targets_winsorized) ** 2)
        mae = np.mean(np.abs(predictions_winsorized - targets_winsorized))
        rmse = np.sqrt(mse)

        # Correlation metrics (using winsorized values)
        spearman_corr, spearman_pval = spearmanr(predictions_winsorized, targets_winsorized)
        pearson_corr, pearson_pval = pearsonr(predictions_winsorized, targets_winsorized)

        # Classification metrics (using winsorized values)
        pred_classes = np.zeros(len(predictions_winsorized), dtype=int)
        pred_classes[predictions_winsorized < -0.05] = 0
        pred_classes[(predictions_winsorized >= -0.05) & (predictions_winsorized <= 0.05)] = 1
        pred_classes[predictions_winsorized > 0.05] = 2

        target_classes = np.zeros(len(targets_winsorized), dtype=int)
        target_classes[targets_winsorized < -0.05] = 0
        target_classes[(targets_winsorized >= -0.05) & (targets_winsorized <= 0.05)] = 1
        target_classes[targets_winsorized > 0.05] = 2

        mcc = matthews_corrcoef(target_classes, pred_classes)
        accuracy = (pred_classes == target_classes).mean()

        # Print results
        print(f"\n{'Regression Metrics':-^60}")
        print(f"MSE:  {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE:  {mae:.6f}")

        print(f"\n{'Correlation Metrics':-^60}")
        print(f"Spearman: {spearman_corr:.4f} (p={spearman_pval:.4e})")
        print(f"Pearson:  {pearson_corr:.4f} (p={pearson_pval:.4e})")

        print(f"\n{'Classification Metrics (threshold-based)':-^60}")
        print(f"MCC:      {mcc:.4f}")
        print(f"Accuracy: {accuracy:.4f}")

        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'spearman_correlation': spearman_corr,
            'spearman_pvalue': spearman_pval,
            'pearson_correlation': pearson_corr,
            'pearson_pvalue': pearson_pval,
            'mcc': mcc,
            'accuracy': accuracy
        }

        return metrics

    def save_artifacts(self):
        """Save model, scalers, and config"""
        print("\n" + "="*60)
        print("SAVING ARTIFACTS")
        print("="*60)

        # Save model
        model_path = os.path.join(self.output_dir, "mlp_regressor.pth")
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")

        # Save scalers
        scalers_path = os.path.join(self.output_dir, "scalers.pkl")
        with open(scalers_path, 'wb') as f:
            pickle.dump({
                'x_scaler': self.x_scaler,
                'y_scaler': self.y_scaler
            }, f)
        print(f"Scalers saved to: {scalers_path}")

    def create_scatter_plot(self, predictions, targets, metrics, title="OOS Predictions"):
        """Create scatter plot of predictions vs actuals"""
        plt.figure(figsize=(10, 8))

        # Scatter plot
        plt.scatter(targets, predictions, alpha=0.5, s=20, edgecolors='none')

        # Perfect prediction line
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

        # Threshold lines
        plt.axhline(y=-0.05, color='gray', linestyle=':', alpha=0.5)
        plt.axhline(y=0.05, color='gray', linestyle=':', alpha=0.5)
        plt.axvline(x=-0.05, color='gray', linestyle=':', alpha=0.5)
        plt.axvline(x=0.05, color='gray', linestyle=':', alpha=0.5)

        plt.xlabel('Actual Returns', fontsize=12)
        plt.ylabel('Predicted Returns', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')

        # Metrics text box
        metrics_text = (
            f"Spearman: {metrics['spearman_correlation']:.4f}\n"
            f"Pearson: {metrics['pearson_correlation']:.4f}\n"
            f"MCC: {metrics['mcc']:.4f}\n"
            f"RMSE: {metrics['rmse']:.4f}\n"
            f"MAE: {metrics['mae']:.4f}"
        )
        plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save
        plot_path = os.path.join(self.output_dir, "oos_predictions_scatter.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Scatter plot saved to: {plot_path}")
        plt.close()


def main():
    """Main execution"""
    print("\n" + "="*60)
    print("OOS MLP TRAINING AND PREDICTION")
    print("="*60)

    # Initialize predictor
    predictor = OOSPredictor()

    # Load training data
    embeddings_train, targets_train = predictor.load_training_data()

    # Train MLP
    train_history = predictor.train_mlp(
        embeddings_train,
        targets_train,
        epochs=100,
        batch_size=64,
        lr=5e-4,
        val_split=0.2
    )

    # Save model and scalers
    predictor.save_artifacts()

    # Load OOS data
    embeddings_oos, targets_oos, df_oos = predictor.load_oos_data()

    # Make OOS predictions
    print("\n" + "="*60)
    print("MAKING OOS PREDICTIONS")
    print("="*60)
    predictions_oos = predictor.predict(embeddings_oos)

    # Evaluate OOS performance
    oos_metrics = predictor.evaluate(predictions_oos, targets_oos)

    # Save OOS results
    print("\n" + "="*60)
    print("SAVING OOS RESULTS")
    print("="*60)

    # Update CSV with new predictions
    df_oos['predicted_return_T1_to_T40'] = predictions_oos
    oos_csv_path = os.path.join(predictor.output_dir, "oos_predictions_with_return_T1_to_T40.csv")
    df_oos.to_csv(oos_csv_path, index=False)
    print(f"OOS predictions saved to: {oos_csv_path}")

    # Save metrics
    metrics_path = os.path.join(predictor.output_dir, "oos_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(oos_metrics, f, indent=2)
    print(f"OOS metrics saved to: {metrics_path}")

    # Create scatter plot
    predictor.create_scatter_plot(
        predictions_oos,
        targets_oos,
        oos_metrics,
        title="OOS Predictions vs Actual Returns (return_T1_to_T40)"
    )

    print("\n" + "="*60)
    print("COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    main()
