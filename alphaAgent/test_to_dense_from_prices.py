#!/usr/bin/env python3
"""
Test script for to_dense_from_prices function.
Creates dummy input data and visualizes the output.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys

# Import the function we want to test
from factor_evaluation import to_dense_from_prices

# Set random seed for reproducibility
np.random.seed(42)


def create_dummy_data(n_tickers=5, n_days=10, freq_minutes=1):
    """
    Create dummy factor_data and price_data dictionaries.

    Args:
        n_tickers: Number of ticker symbols to generate
        n_days: Number of trading days
        freq_minutes: Frequency in minutes (default: 1 for 1-minute data)

    Returns:
        Tuple of (factor_data, price_data) dictionaries
    """
    tickers = [f'TICK{i:02d}' for i in range(n_tickers)]

    # Generate timestamps for US regular session (09:30-16:00 ET)
    # Approximately 390 minutes per trading day
    start_date = datetime(2024, 1, 2, 9, 30)  # Start on a Tuesday

    factor_data = {}
    price_data = {}

    for ticker in tickers:
        timestamps = []

        # Generate timestamps for each trading day
        for day in range(n_days):
            current_date = start_date + timedelta(days=day)
            # Skip weekends (Monday=0, Sunday=6)
            while current_date.weekday() >= 5:
                current_date += timedelta(days=1)

            # Generate timestamps for regular session (09:30-16:00)
            session_start = current_date.replace(hour=9, minute=30)
            session_end = current_date.replace(hour=16, minute=0)

            current_time = session_start
            while current_time < session_end:
                timestamps.append(current_time)
                current_time += timedelta(minutes=freq_minutes)

        # Create price series: random walk with drift
        n_points = len(timestamps)
        initial_price = 100 + np.random.randn() * 10
        returns = np.random.randn(n_points) * 0.001 + 0.00005  # Small drift
        prices = initial_price * np.exp(np.cumsum(returns))

        # Create factor series: some signal with noise
        # Use a mix of trend and mean-reverting components
        trend = np.linspace(-1, 1, n_points) * np.random.randn()
        noise = np.random.randn(n_points) * 0.5
        factor_values = trend + noise

        # Add some random NaN values to simulate missing data (5% missing)
        nan_mask = np.random.rand(n_points) < 0.05
        factor_values[nan_mask] = np.nan
        prices[nan_mask] = np.nan

        # Create pandas Series
        factor_data[ticker] = pd.Series(factor_values, index=timestamps)
        price_data[ticker] = pd.Series(prices, index=timestamps)

    return factor_data, price_data


def visualize_dense_matrices(ts_index, tickers, F, R):
    """
    Visualize the output of to_dense_from_prices.

    Args:
        ts_index: pd.Index of timestamps (length T)
        tickers: List of ticker symbols (length N)
        F: Factor matrix of shape (T, N), dtype float32
        R: Returns matrix of shape (T, N), dtype float32
    """
    T, N = F.shape

    print(f"\n{'='*60}")
    print(f"Output Dimensions:")
    print(f"  Timestamps (T): {T}")
    print(f"  Tickers (N): {N}")
    print(f"  Factor matrix shape: {F.shape}")
    print(f"  Returns matrix shape: {R.shape}")
    print(f"{'='*60}\n")

    print(f"Tickers: {tickers}")
    print(f"\nFirst few timestamps:")
    print(f"  {ts_index[:5].tolist()}")
    print(f"\nLast few timestamps:")
    print(f"  {ts_index[-5:].tolist()}")

    # Calculate statistics
    print(f"\n{'='*60}")
    print(f"Factor Matrix Statistics:")
    print(f"  Non-NaN values: {np.isfinite(F).sum()} / {F.size} ({np.isfinite(F).mean()*100:.2f}%)")
    print(f"  Mean: {np.nanmean(F):.4f}")
    print(f"  Std: {np.nanstd(F):.4f}")
    print(f"  Min: {np.nanmin(F):.4f}")
    print(f"  Max: {np.nanmax(F):.4f}")

    print(f"\n{'='*60}")
    print(f"Returns Matrix Statistics:")
    print(f"  Non-NaN values: {np.isfinite(R).sum()} / {R.size} ({np.isfinite(R).mean()*100:.2f}%)")
    print(f"  Mean: {np.nanmean(R):.6f}")
    print(f"  Std: {np.nanstd(R):.6f}")
    print(f"  Min: {np.nanmin(R):.6f}")
    print(f"  Max: {np.nanmax(R):.6f}")
    print(f"{'='*60}\n")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('to_dense_from_prices Output Visualization', fontsize=16, fontweight='bold')

    # Plot 1: Factor matrix heatmap
    ax1 = axes[0, 0]
    im1 = ax1.imshow(F.T, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    ax1.set_title('Factor Matrix (F)', fontsize=13)
    ax1.set_xlabel('Time Index')
    ax1.set_ylabel('Ticker Index')
    ax1.set_yticks(range(N))
    ax1.set_yticklabels(tickers, fontsize=8)
    plt.colorbar(im1, ax=ax1, label='Factor Value')

    # Plot 2: Returns matrix heatmap
    ax2 = axes[0, 1]
    im2 = ax2.imshow(R.T, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    ax2.set_title('Returns Matrix (R)', fontsize=13)
    ax2.set_xlabel('Time Index')
    ax2.set_ylabel('Ticker Index')
    ax2.set_yticks(range(N))
    ax2.set_yticklabels(tickers, fontsize=8)
    plt.colorbar(im2, ax=ax2, label='Return Value')

    # Plot 3: Factor values over time for first ticker
    ax3 = axes[1, 0]
    if N > 0:
        ax3.plot(range(T), F[:, 0], linewidth=1, alpha=0.7, label=f'{tickers[0]}')
        if N > 1:
            ax3.plot(range(T), F[:, 1], linewidth=1, alpha=0.7, label=f'{tickers[1]}')
        ax3.set_title('Factor Values Over Time (First 2 Tickers)', fontsize=13)
        ax3.set_xlabel('Time Index')
        ax3.set_ylabel('Factor Value')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # Plot 4: Returns over time for first ticker
    ax4 = axes[1, 1]
    if N > 0:
        ax4.plot(range(T), R[:, 0], linewidth=1, alpha=0.7, label=f'{tickers[0]}')
        if N > 1:
            ax4.plot(range(T), R[:, 1], linewidth=1, alpha=0.7, label=f'{tickers[1]}')
        ax4.set_title('Returns Over Time (First 2 Tickers)', fontsize=13)
        ax4.set_xlabel('Time Index')
        ax4.set_ylabel('Return Value')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = '/home/lichenhui/data/alphaAgent/to_dense_from_prices_output.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")

    # Also show the plot
    plt.show()


def main():
    """Main function to test to_dense_from_prices."""
    print("="*60)
    print("Testing to_dense_from_prices function")
    print("="*60)

    # Create dummy input data
    print("\nCreating dummy input data...")
    n_tickers = 5
    n_days = 10
    factor_data, price_data = create_dummy_data(n_tickers=n_tickers, n_days=n_days)

    print(f"  Generated {n_tickers} tickers with {n_days} trading days")
    print(f"  Tickers: {list(factor_data.keys())}")
    print(f"  Factor data points per ticker: ~{len(factor_data[list(factor_data.keys())[0]])}")
    print(f"  Price data points per ticker: ~{len(price_data[list(price_data.keys())[0]])}")

    # Call the function
    print("\nCalling to_dense_from_prices...")
    ts_index, tickers, F, R = to_dense_from_prices(
        factor_data=factor_data,
        price_data=price_data,
        resample_freq='30T'  # Resample to 30-minute intervals
    )

    # Visualize the output
    print("\nVisualizing output...")
    visualize_dense_matrices(ts_index, tickers, F, R)

    print("\n" + "="*60)
    print("Test completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
