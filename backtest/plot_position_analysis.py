import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/data/backtest')
from utils import calculate_trading_year_length

# Load the target price (positions) file
positions_file = 'target_price_hourly_forward_ewma_40h_sp_K15_N50.csv'
returns_file = '/data/csv/hourly_close_to_close_returns_matrix.csv'

print(f"Loading positions from {positions_file}...")
positions = pd.read_csv(positions_file, index_col='date', parse_dates=True)
print(f"Positions shape: {positions.shape}")

print(f"Loading returns from {returns_file}...")
returns = pd.read_csv(returns_file, index_col='datetime', parse_dates=True)
print(f"Returns shape: {returns.shape}")

# Calculate trading year length
trading_year_length = calculate_trading_year_length(returns_file)

# Get common columns (tickers) between positions and returns
common_tickers = positions.columns.intersection(returns.columns)
print(f"Common tickers: {len(common_tickers)}")

# Filter to common tickers
positions = positions[common_tickers]
returns = returns[common_tickers]

# Calculate metrics across time
# (1) Gross exposure: sum of absolute positions
gross_exposure = positions.abs().sum(axis=1)

# (2) Net exposure: sum of positions
net_exposure = positions.sum(axis=1)

# (3) Trade size between periods: difference in positions
trades = positions.diff()
# Distribution of trade size per period: min and max across names
trade_size_min_per_period = trades.abs().min(axis=1)
trade_size_max_per_period = trades.abs().max(axis=1)

# (4) Proxy of risk: portfolio returns = positions * returns (dot product)
# Align indices - positions at time t, returns at time t+1
# Shift positions back by 1 to align with returns
positions_aligned = positions.shift(1)

# Get common dates
common_dates = positions_aligned.index.intersection(returns.index)
positions_aligned = positions_aligned.loc[common_dates]
returns_aligned = returns.loc[common_dates]

# Portfolio returns: sum of (position * return) across all names
portfolio_returns = (positions_aligned * returns_aligned).sum(axis=1)

# Calculate annualized std of PnL
pnl_std = portfolio_returns.std()
pnl_std_annualized = pnl_std * np.sqrt(trading_year_length)

# Create figure with 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Gross Exposure
ax1 = axes[0, 0]
ax1.plot(gross_exposure.index, gross_exposure.values, linewidth=0.5, color='blue')
ax1.set_title('Gross Exposure Across Names', fontsize=12, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Gross Exposure')
ax1.tick_params(axis='x', rotation=45)
ax1.grid(True, alpha=0.3)

# Plot 2: Net Exposure
ax2 = axes[0, 1]
ax2.plot(net_exposure.index, net_exposure.values, linewidth=0.5, color='green')
ax2.axhline(y=0, color='red', linestyle='--', linewidth=0.5)
ax2.set_title('Net Exposure Across Names', fontsize=12, fontweight='bold')
ax2.set_xlabel('Date')
ax2.set_ylabel('Net Exposure')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(True, alpha=0.3)

# Plot 3: Distribution of Trade Size (min and max)
ax3 = axes[1, 0]
ax3.fill_between(trade_size_min_per_period.index, trade_size_min_per_period.values,
                 trade_size_max_per_period.values, alpha=0.3, color='orange', label='Min-Max Range')
ax3.plot(trade_size_min_per_period.index, trade_size_min_per_period.values, linewidth=0.5, color='green', label='Min')
ax3.plot(trade_size_max_per_period.index, trade_size_max_per_period.values, linewidth=0.5, color='red', label='Max')
ax3.set_title('Trade Size Distribution Per Period (Min/Max)', fontsize=12, fontweight='bold')
ax3.set_xlabel('Date')
ax3.set_ylabel('Trade Size Per Name')
ax3.tick_params(axis='x', rotation=45)
ax3.grid(True, alpha=0.3)
ax3.legend(loc='upper right')

# Plot 4: Portfolio Returns (Proxy of Risk) with annualized std
ax4 = axes[1, 1]
ax4.plot(portfolio_returns.index, portfolio_returns.values, linewidth=0.5, color='purple')
ax4.axhline(y=0, color='red', linestyle='--', linewidth=0.5)
ax4.set_title(f'Portfolio Returns (Ann. Std: {pnl_std_annualized:.4f})', fontsize=12, fontweight='bold')
ax4.set_xlabel('Date')
ax4.set_ylabel('Portfolio Return')
ax4.tick_params(axis='x', rotation=45)
ax4.grid(True, alpha=0.3)

# Add cumulative return as secondary info
cumulative_return = (1 + portfolio_returns).cumprod()
ax4_twin = ax4.twinx()
ax4_twin.plot(cumulative_return.index, cumulative_return.values, linewidth=0.5, color='darkviolet', alpha=0.5)
ax4_twin.set_ylabel('Cumulative Return', color='darkviolet')

plt.suptitle('Position Analysis: target_price_hourly_forward_ewma_40h_sp_K15_N50', fontsize=14, fontweight='bold')
plt.tight_layout()

# Save the plot
plt.savefig('position_analysis.png', dpi=150, bbox_inches='tight')
print("Plot saved to position_analysis.png")

# Print summary statistics
print("\n=== Summary Statistics ===")
print(f"Trading Year Length: {trading_year_length} periods")
print(f"Gross Exposure - Mean: {gross_exposure.mean():.4f}, Std: {gross_exposure.std():.4f}")
print(f"Net Exposure - Mean: {net_exposure.mean():.6f}, Std: {net_exposure.std():.6f}")
print(f"Trade Size Min - Mean: {trade_size_min_per_period.mean():.6f}, Max: {trade_size_min_per_period.max():.6f}")
print(f"Trade Size Max - Mean: {trade_size_max_per_period.mean():.6f}, Max: {trade_size_max_per_period.max():.6f}")
print(f"Portfolio Returns - Mean: {portfolio_returns.mean():.6f}, Std: {portfolio_returns.std():.6f}")
print(f"Portfolio Returns - Annualized Std: {pnl_std_annualized:.6f}")
print(f"Sharpe Ratio (annualized): {portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(trading_year_length):.4f}")

plt.show()
