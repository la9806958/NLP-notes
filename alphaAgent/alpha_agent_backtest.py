#!/usr/bin/env python3
"""
AlphaAgent Backtesting Framework
Loads 1-minute OHLCV data and tests factor strategies
Target: Use 300 minutes of history to predict 30-minute forward returns
"""

import pandas as pd
import numpy as np
import os
import sys
import psutil
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import glob
from alpha_agent_operators import Operators
from alpha_agent_factor import VolumeReactionAlpha, FactorAgent
import warnings
warnings.filterwarnings('ignore')


class MemoryProfiler:
    """Track memory usage throughout backtest execution"""

    def __init__(self):
        self.process = psutil.Process()
        self.checkpoints = []
        self.start_memory = None

    def start(self):
        """Record initial memory state"""
        gc.collect()  # Force garbage collection for accurate baseline
        mem_info = self.process.memory_info()
        self.start_memory = mem_info.rss / (1024 ** 3)  # Convert to GB
        self.checkpoints.append({
            'stage': 'start',
            'rss_gb': self.start_memory,
            'vms_gb': mem_info.vms / (1024 ** 3),
            'delta_gb': 0.0
        })
        print(f"Memory Profiling Started - Initial: {self.start_memory:.3f} GB")

    def checkpoint(self, stage: str, force_gc: bool = False):
        """Record memory at a specific stage"""
        if force_gc:
            gc.collect()

        mem_info = self.process.memory_info()
        rss_gb = mem_info.rss / (1024 ** 3)
        vms_gb = mem_info.vms / (1024 ** 3)
        delta_gb = rss_gb - self.start_memory

        self.checkpoints.append({
            'stage': stage,
            'rss_gb': rss_gb,
            'vms_gb': vms_gb,
            'delta_gb': delta_gb
        })

        print(f"  [{stage}] Memory: {rss_gb:.3f} GB (Δ {delta_gb:+.3f} GB)")

    def report(self) -> pd.DataFrame:
        """Generate memory usage report"""
        df = pd.DataFrame(self.checkpoints)
        print("\n" + "="*70)
        print("MEMORY USAGE REPORT")
        print("="*70)
        print(df.to_string(index=False))

        if len(self.checkpoints) > 1:
            peak_memory = max(cp['rss_gb'] for cp in self.checkpoints)
            total_increase = self.checkpoints[-1]['rss_gb'] - self.start_memory
            print(f"\nPeak Memory: {peak_memory:.3f} GB")
            print(f"Total Increase: {total_increase:+.3f} GB")

        return df

    def get_object_sizes(self, frame_locals: dict, top_n: int = 10):
        """Get sizes of top N largest objects in memory"""
        sizes = []
        for name, obj in frame_locals.items():
            try:
                size_mb = sys.getsizeof(obj) / (1024 ** 2)
                if isinstance(obj, pd.DataFrame):
                    size_mb = obj.memory_usage(deep=True).sum() / (1024 ** 2)
                elif isinstance(obj, np.ndarray):
                    size_mb = obj.nbytes / (1024 ** 2)
                sizes.append({'name': name, 'type': type(obj).__name__, 'size_mb': size_mb})
            except:
                pass

        df_sizes = pd.DataFrame(sizes).sort_values('size_mb', ascending=False).head(top_n)
        print("\n" + "="*70)
        print(f"TOP {top_n} LARGEST OBJECTS IN MEMORY")
        print("="*70)
        print(df_sizes.to_string(index=False))
        return df_sizes


class DataLoader:
    """Load and prepare 1-minute OHLCV data"""
    
    def __init__(self, data_dir: str = "/data/1min"):
        self.data_dir = Path(data_dir)
        self.ops = Operators()
        
    def load_ticker_data(self, ticker: str, start_date: Optional[str] = None, 
                        end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load 1-minute OHLCV data for a ticker
        
        Returns:
            DataFrame with columns: open, high, low, close, volume
            Index: datetime
        """
        ticker_file = self.data_dir / f"{ticker}.csv"
        
        if not ticker_file.exists():
            print(f"Warning: {ticker_file} not found")
            return pd.DataFrame()
        
        # Load data
        df = pd.read_csv(ticker_file)
        
        # Standardize column names
        df.columns = df.columns.str.lower()
        
        # Parse datetime
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        elif 'date' in df.columns and 'time' in df.columns:
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        elif 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'])
        else:
            # Assume first column is datetime
            df['datetime'] = pd.to_datetime(df.iloc[:, 0])
        
        df.set_index('datetime', inplace=True)
        
        # Select OHLCV columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        available_cols = [col for col in required_cols if col in df.columns]
        df = df[available_cols]
        
        # Filter by date range if specified
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
        
        return df
    
    def load_multiple_tickers(self, tickers: List[str], start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load data for multiple tickers
        
        Returns:
            DataFrame with MultiIndex (datetime, ticker) and columns: open, high, low, close, volume
        """
        all_data = []
        
        for ticker in tickers:
            print(f"Loading {ticker}...")
            ticker_data = self.load_ticker_data(ticker, start_date, end_date)
            
            if not ticker_data.empty:
                ticker_data['ticker'] = ticker
                ticker_data.reset_index(inplace=True)
                all_data.append(ticker_data)
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all data
        combined = pd.concat(all_data, ignore_index=True)
        combined.set_index(['datetime', 'ticker'], inplace=True)
        
        return combined
    
    def get_available_tickers(self, min_days: int = 100) -> List[str]:
        """Get list of available tickers with sufficient data"""
        tickers = []
        
        for file_path in self.data_dir.glob("*.csv"):
            ticker = file_path.stem
            
            # Quick check for file size (rough proxy for data availability)
            if file_path.stat().st_size > 1000 * min_days:  # Rough estimate
                tickers.append(ticker)
        
        return sorted(tickers)


class AlphaBacktester:
    """Backtest alpha strategies on 1-minute data"""
    
    def __init__(self, lookback_minutes: int = 300, forward_minutes: int = 30):
        """
        Initialize backtester
        
        Args:
            lookback_minutes: Historical data to use for signals (default 300 = 5 hours)
            forward_minutes: Forward return prediction horizon (default 30 minutes)
        """
        self.lookback_minutes = lookback_minutes
        self.forward_minutes = forward_minutes
        self.ops = Operators()
        
    def prepare_features_and_targets(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare feature matrix and target returns
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            features: DataFrame of lagged features
            targets: DataFrame of forward returns
        """
        # Pivot data to wide format if needed
        if isinstance(data.index, pd.MultiIndex):
            close = data['close'].unstack('ticker')
            volume = data['volume'].unstack('ticker')
            high = data['high'].unstack('ticker')
            low = data['low'].unstack('ticker')
            open_price = data['open'].unstack('ticker')
        else:
            close = data['close']
            volume = data['volume']
            high = data['high']
            low = data['low']
            open_price = data['open']
        
        # Calculate forward returns (target)
        forward_returns = close.shift(-self.forward_minutes) / close - 1
        forward_returns = forward_returns.shift(-1)  # Avoid look-ahead bias
        
        # Package data for factor computation
        feature_data = pd.DataFrame({
            'close': close,
            'volume': volume,
            'high': high,
            'low': low,
            'open': open_price
        })
        
        return feature_data, forward_returns
    
    def compute_signals(self, data: pd.DataFrame, alpha_model) -> pd.DataFrame:
        """
        Compute alpha signals using the model
        
        Args:
            data: Feature data
            alpha_model: Alpha model instance (e.g., VolumeReactionAlpha)
            
        Returns:
            DataFrame of signals
        """
        return alpha_model.compute_factors(data)
    
    def backtest(self, data: pd.DataFrame, alpha_model,
                start_date: Optional[str] = None,
                enable_memory_profiling: bool = False) -> Dict[str, Any]:
        """
        Run backtest of alpha strategy

        Args:
            data: OHLCV data
            alpha_model: Alpha model to test
            start_date: Start date for backtest (to allow warmup)
            enable_memory_profiling: If True, track memory usage at each stage

        Returns:
            Dictionary with backtest results including PNL timeseries
        """
        # Initialize memory profiler if requested
        mem_profiler = None
        if enable_memory_profiling:
            mem_profiler = MemoryProfiler()
            mem_profiler.start()
            mem_profiler.checkpoint("backtest_start")

        print(f"Preparing data with {self.lookback_minutes}-min lookback, "
              f"{self.forward_minutes}-min forward horizon...")

        # Prepare features and targets
        features, targets = self.prepare_features_and_targets(data)
        if mem_profiler:
            mem_profiler.checkpoint("after_prepare_features")

        # Compute signals
        print("Computing alpha signals...")
        signals = self.compute_signals(features, alpha_model)
        if mem_profiler:
            mem_profiler.checkpoint("after_compute_signals")

        # Align signals with targets
        if isinstance(signals, pd.Series):
            signals = signals.to_frame('signal')

        if isinstance(targets, pd.DataFrame):
            # Flatten for analysis
            targets_flat = targets.stack().to_frame('return')
            signals_flat = signals

            # Ensure alignment
            aligned = pd.concat([signals_flat, targets_flat], axis=1).dropna()
        else:
            aligned = pd.concat([signals, targets.to_frame('return')], axis=1).dropna()

        if mem_profiler:
            mem_profiler.checkpoint("after_alignment")

        # Filter to backtest period
        if start_date:
            backtest_start = pd.to_datetime(start_date) + timedelta(minutes=self.lookback_minutes)
            aligned = aligned[aligned.index.get_level_values(0) >= backtest_start]

        if aligned.empty:
            print("No valid data after alignment")
            return {}

        # Calculate positions (simple: long top quintile, short bottom quintile)
        signal_col = aligned.columns[0] if 'signal' not in aligned.columns else 'signal'

        # Quintile positions
        aligned['signal_rank'] = aligned.groupby(level=0)[signal_col].rank(pct=True)
        aligned['position'] = 0
        aligned.loc[aligned['signal_rank'] > 0.8, 'position'] = 1
        aligned.loc[aligned['signal_rank'] < 0.2, 'position'] = -1

        if mem_profiler:
            mem_profiler.checkpoint("after_position_calculation")

        # Calculate returns
        aligned['strategy_return'] = aligned['position'].shift(1) * aligned['return']

        # Calculate PNL timeseries (aggregate by timestamp)
        pnl_timeseries = aligned.groupby(level=0)['strategy_return'].mean()
        cumulative_pnl = (1 + pnl_timeseries).cumprod() - 1

        if mem_profiler:
            mem_profiler.checkpoint("after_pnl_calculation")

        # Aggregate results
        results = self._calculate_metrics(aligned)

        # Add PNL timeseries to results
        results['pnl_timeseries'] = pnl_timeseries
        results['cumulative_pnl'] = cumulative_pnl
        results['aligned_data'] = aligned

        if mem_profiler:
            mem_profiler.checkpoint("backtest_complete", force_gc=True)
            mem_df = mem_profiler.report()
            mem_profiler.get_object_sizes(locals(), top_n=10)
            results['memory_profile'] = mem_df

        return results
    
    def _calculate_metrics(self, aligned: pd.DataFrame) -> Dict[str, float]:
        """Calculate backtest metrics"""

        # Daily returns (assuming ~390 minutes per trading day)
        minutes_per_day = 390
        periods_per_day = minutes_per_day / self.forward_minutes

        # Clean returns
        returns = aligned['strategy_return'].replace([np.inf, -np.inf], 0).fillna(0)

        if len(returns) == 0:
            return {}

        # Calculate metrics
        total_return = (1 + returns).prod() - 1
        mean_return = returns.mean()
        std_return = returns.std()

        # Annualized metrics (252 trading days)
        ann_factor = np.sqrt(252 * periods_per_day)
        sharpe = (mean_return / std_return * ann_factor) if std_return > 0 else 0

        # Additional metrics
        win_rate = (returns > 0).mean()

        # Information ratio (vs equal weight benchmark)
        benchmark_return = aligned['return'].mean()
        excess_return = returns - benchmark_return
        tracking_error = excess_return.std()
        ir = (excess_return.mean() / tracking_error * ann_factor) if tracking_error > 0 else 0

        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        metrics = {
            'total_return': total_return,
            'annualized_return': (1 + total_return) ** (252 * periods_per_day / len(returns)) - 1,
            'sharpe_ratio': sharpe,
            'information_ratio': ir,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'mean_return_per_period': mean_return,
            'std_return_per_period': std_return,
            'num_periods': len(returns),
            'avg_position_size': aligned['position'].abs().mean()
        }

        return metrics



def main():
    """Main execution function"""

    print("="*60)
    print("AlphaAgent Backtest System (with Memory Profiling)")
    print("="*60)

    # Initialize memory profiler for entire pipeline
    mem_profiler = MemoryProfiler()
    mem_profiler.start()

    # Initialize components
    loader = DataLoader("/data/1min")
    backtester = AlphaBacktester(lookback_minutes=300, forward_minutes=30)
    mem_profiler.checkpoint("after_init")

    # Get available tickers
    print("\nScanning for available tickers...")
    available_tickers = loader.get_available_tickers(min_days=100)

    if not available_tickers:
        print("No tickers found in /data/1min")
        return

    print(f"Found {len(available_tickers)} tickers with sufficient data")

    # Select subset for testing (e.g., liquid stocks)
    test_tickers = available_tickers[:50]  # Use first 50 tickers
    print(f"Using {len(test_tickers)} tickers for backtest")
    mem_profiler.checkpoint("after_ticker_scan")

    # Load data
    print("\nLoading 1-minute OHLCV data...")
    start_date = "2024-01-01"
    end_date = "2024-12-31"

    data = loader.load_multiple_tickers(test_tickers, start_date, end_date)

    if data.empty:
        print("Failed to load data")
        return

    print(f"Loaded data shape: {data.shape}")
    print(f"Date range: {data.index.get_level_values(0).min()} to {data.index.get_level_values(0).max()}")
    mem_profiler.checkpoint("after_data_load")

    # Test with Volume Reaction Alpha
    print("\n" + "="*60)
    print("Testing Volume Reaction Hypothesis")
    print("="*60)

    print("\nHypothesis: Volume reaction leads price reaction")
    print("Large, unusual shifts in trading activity reflect order-flow imbalances")
    print("and information arrival. Price incorporates this with a lag.")

    alpha_model = VolumeReactionAlpha()

    # Run backtest with memory profiling enabled
    print("\nRunning backtest with memory profiling...")
    results = backtester.backtest(data, alpha_model, start_date="2024-02-01",
                                  enable_memory_profiling=True)

    # Display results
    if results:
        print("\n" + "="*60)
        print("Backtest Results")
        print("="*60)

        # Display numeric metrics only (skip dataframes/series)
        for metric, value in results.items():
            if isinstance(value, float):
                if 'return' in metric or 'ratio' in metric:
                    print(f"{metric:25s}: {value:>10.4f}")
                else:
                    print(f"{metric:25s}: {value:>10.6f}")
            elif isinstance(value, int):
                print(f"{metric:25s}: {value}")

        # Performance summary
        print("\n" + "="*60)
        print("Performance Summary")
        print("="*60)

        if results['sharpe_ratio'] > 1.0:
            print("✅ Strong performance: Sharpe ratio > 1.0")
        elif results['sharpe_ratio'] > 0.5:
            print("⚠️  Moderate performance: Sharpe ratio 0.5-1.0")
        else:
            print("❌ Weak performance: Sharpe ratio < 0.5")

        print(f"\nStrategy generated {results['avg_position_size']:.1%} average position size")
        print(f"Win rate: {results['win_rate']:.1%}")
        print(f"Maximum drawdown: {results['max_drawdown']:.1%}")

    # Final memory report for entire pipeline
    mem_profiler.checkpoint("pipeline_complete", force_gc=True)
    mem_profiler.report()
    mem_profiler.get_object_sizes(locals(), top_n=15)

    # Save memory profile to CSV if available
    if results and 'memory_profile' in results:
        output_path = Path("backtest_results")
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        mem_file = output_path / f"memory_profile_{timestamp}.csv"
        results['memory_profile'].to_csv(mem_file, index=False)
        print(f"\nBacktest memory profile saved to: {mem_file}")

    print("\n" + "="*60)
    print("Backtest complete!")
    print("="*60)


if __name__ == "__main__":
    main()