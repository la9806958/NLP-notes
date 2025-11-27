#!/usr/bin/env python3
"""
Script to examine Qwen embeddings and diagnose why OOS predictions are all identical
"""

import numpy as np
import pandas as pd
import os

# Load embeddings
print("Loading embeddings...")
embeddings_train = np.load("earnings_pipeline_output_only_analysis/qwen_embeddings_train.npy")
embeddings_oos = np.load("earnings_pipeline_output_only_analysis/qwen_embeddings_oos.npy")

print(f"\n=== TRAINING EMBEDDINGS ===")
print(f"Shape: {embeddings_train.shape}")
print(f"Mean: {embeddings_train.mean():.6f}")
print(f"Std: {embeddings_train.std():.6f}")
print(f"Min: {embeddings_train.min():.6f}")
print(f"Max: {embeddings_train.max():.6f}")
print(f"Number of NaN values: {np.isnan(embeddings_train).sum()}")
print(f"Number of Inf values: {np.isinf(embeddings_train).sum()}")

# Check if all rows are identical
train_unique_rows = len(np.unique(embeddings_train, axis=0))
print(f"Unique rows: {train_unique_rows} / {len(embeddings_train)}")

# Check variance across features
train_feature_variance = embeddings_train.var(axis=0)
zero_variance_features_train = (train_feature_variance < 1e-10).sum()
print(f"Features with zero variance: {zero_variance_features_train} / {embeddings_train.shape[1]}")

print(f"\n=== OOS EMBEDDINGS ===")
print(f"Shape: {embeddings_oos.shape}")
print(f"Mean: {embeddings_oos.mean():.6f}")
print(f"Std: {embeddings_oos.std():.6f}")
print(f"Min: {embeddings_oos.min():.6f}")
print(f"Max: {embeddings_oos.max():.6f}")
print(f"Number of NaN values: {np.isnan(embeddings_oos).sum()}")
print(f"Number of Inf values: {np.isinf(embeddings_oos).sum()}")

# Check if all rows are identical
oos_unique_rows = len(np.unique(embeddings_oos, axis=0))
print(f"Unique rows: {oos_unique_rows} / {len(embeddings_oos)}")

# Check variance across features
oos_feature_variance = embeddings_oos.var(axis=0)
zero_variance_features_oos = (oos_feature_variance < 1e-10).sum()
print(f"Features with zero variance: {zero_variance_features_oos} / {embeddings_oos.shape[1]}")

# Check if OOS embeddings are all identical
if oos_unique_rows == 1:
    print("\n⚠️  WARNING: ALL OOS EMBEDDINGS ARE IDENTICAL!")
    print("This is the root cause of identical predictions!")
    print(f"The constant embedding vector (first 10 dims): {embeddings_oos[0, :10]}")
else:
    print(f"\n✓ OOS embeddings are diverse ({oos_unique_rows} unique rows)")

# Load OOS predictions to verify
if os.path.exists("earnings_pipeline_output_only_analysis/oos_predictions_2023_onwards.csv"):
    print("\n=== OOS PREDICTIONS ===")
    df_oos = pd.read_csv("earnings_pipeline_output_only_analysis/oos_predictions_2023_onwards.csv")
    predictions = df_oos['predicted_return'].values
    print(f"Number of predictions: {len(predictions)}")
    print(f"Unique predictions: {len(np.unique(predictions))}")
    print(f"Mean prediction: {predictions.mean():.6f}")
    print(f"Std prediction: {predictions.std():.6f}")
    print(f"Min prediction: {predictions.min():.6f}")
    print(f"Max prediction: {predictions.max():.6f}")

    if len(np.unique(predictions)) == 1:
        print(f"\n⚠️  CONFIRMED: All predictions are {predictions[0]:.6f}")
    else:
        print(f"\n✓ Predictions are diverse")

# Check some sample embeddings
print("\n=== SAMPLE EMBEDDINGS (first 5 samples, first 10 dims) ===")
print("Train embeddings:")
print(embeddings_train[:5, :10])
print("\nOOS embeddings:")
print(embeddings_oos[:5, :10])

# Check if embeddings are normalized
train_norms = np.linalg.norm(embeddings_train, axis=1)
oos_norms = np.linalg.norm(embeddings_oos, axis=1)
print(f"\n=== EMBEDDING NORMS ===")
print(f"Train - Mean: {train_norms.mean():.6f}, Std: {train_norms.std():.6f}")
print(f"OOS - Mean: {oos_norms.mean():.6f}, Std: {oos_norms.std():.6f}")
