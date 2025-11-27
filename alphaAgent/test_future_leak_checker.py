#!/usr/bin/env python3
"""
Unit tests for future_leak_checker.py

Tests validate that:
1. Leakage IS detected when it exists
2. Leakage IS NOT detected when it doesn't exist
"""

import pytest
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta

from future_leak_checker import (
    check_no_future_leakage,
    check_factor_dict_for_leakage,
    FutureLeakageError
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# TEST DATA GENERATION
# =============================================================================

def generate_sample_ohlcv_data(n_periods: int = 1000, freq: str = '1min') -> pd.DataFrame:
    """
    Generate synthetic OHLCV data for testing.

    Args:
        n_periods: Number of time periods
        freq: Frequency string for pandas DatetimeIndex

    Returns:
        DataFrame with OHLCV columns and datetime index
    """
    # Generate datetime index
    start_date = datetime(2024, 1, 1, 9, 30)
    dates = pd.date_range(start=start_date, periods=n_periods, freq=freq)

    # Generate realistic price data using random walk
    np.random.seed(42)
    returns = np.random.randn(n_periods) * 0.001  # 0.1% std per period
    close_prices = 100 * np.exp(np.cumsum(returns))

    # Generate OHLC from close
    high_prices = close_prices * (1 + np.abs(np.random.randn(n_periods)) * 0.005)
    low_prices = close_prices * (1 - np.abs(np.random.randn(n_periods)) * 0.005)
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]

    # Generate volume
    volumes = np.random.randint(1000, 10000, n_periods)

    return pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }, index=dates)


# =============================================================================
# TEST FACTOR FUNCTIONS
# =============================================================================

def factor_no_leakage_simple_sma(data: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Factor WITHOUT leakage: Simple moving average of past close prices.
    Uses only current and historical data (shift ensures causality).
    """
    return data['close'].rolling(window=window).mean()


def factor_no_leakage_momentum(data: pd.DataFrame, lookback: int = 10) -> pd.Series:
    """
    Factor WITHOUT leakage: Price momentum (current price - past price).
    Uses only historical data.
    """
    return data['close'] - data['close'].shift(lookback)


def factor_no_leakage_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Factor WITHOUT leakage: Relative Strength Index.
    Classic technical indicator using only historical data.
    """
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def factor_with_leakage_future_return(data: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """
    Factor WITH LEAKAGE: Uses future returns (next N periods).
    This should be detected as having leakage!
    """
    # Calculate future return - THIS IS LEAKAGE!
    future_price = data['close'].shift(-horizon)
    current_price = data['close']
    return (future_price / current_price - 1)


def factor_with_leakage_future_high(data: pd.DataFrame, horizon: int = 10) -> pd.Series:
    """
    Factor WITH LEAKAGE: Uses maximum future price.
    This should be detected as having leakage!
    """
    # Look ahead to future highs - THIS IS LEAKAGE!
    return data['high'].rolling(window=horizon, min_periods=1).max().shift(-horizon)


def factor_with_leakage_zscore_including_future(data: pd.DataFrame, window: int = 50) -> pd.Series:
    """
    Factor WITH LEAKAGE: Z-score computed using centered window (includes future data).
    This should be detected as having leakage!
    """
    # Using centered window includes future data - THIS IS LEAKAGE!
    mean = data['close'].rolling(window=window, center=True).mean()
    std = data['close'].rolling(window=window, center=True).std()
    return (data['close'] - mean) / std


# =============================================================================
# TESTS FOR FACTORS WITHOUT LEAKAGE
# =============================================================================

def test_no_leakage_simple_sma():
    """Test that simple moving average factor does NOT trigger leakage detection."""
    data = generate_sample_ohlcv_data(n_periods=500)

    # Should NOT raise FutureLeakageError
    result = check_no_future_leakage(
        factor_fn=factor_no_leakage_simple_sma,
        data=data,
        n_test_points=10,
        tolerance=1e-8
    )

    assert result is True, "SMA factor should pass (no leakage)"
    logger.info("✅ Test passed: SMA factor has no leakage")


def test_no_leakage_momentum():
    """Test that momentum factor does NOT trigger leakage detection."""
    data = generate_sample_ohlcv_data(n_periods=500)

    # Should NOT raise FutureLeakageError
    result = check_no_future_leakage(
        factor_fn=factor_no_leakage_momentum,
        data=data,
        n_test_points=10,
        tolerance=1e-8
    )

    assert result is True, "Momentum factor should pass (no leakage)"
    logger.info("✅ Test passed: Momentum factor has no leakage")


def test_no_leakage_rsi():
    """Test that RSI factor does NOT trigger leakage detection."""
    data = generate_sample_ohlcv_data(n_periods=500)

    # Should NOT raise FutureLeakageError
    result = check_no_future_leakage(
        factor_fn=factor_no_leakage_rsi,
        data=data,
        n_test_points=10,
        tolerance=1e-8
    )

    assert result is True, "RSI factor should pass (no leakage)"
    logger.info("✅ Test passed: RSI factor has no leakage")


# =============================================================================
# TESTS FOR FACTORS WITH LEAKAGE (SHOULD BE DETECTED)
# =============================================================================

def test_leakage_detected_future_return():
    """Test that future return factor DOES trigger leakage detection."""
    data = generate_sample_ohlcv_data(n_periods=500)

    # Should raise FutureLeakageError
    with pytest.raises(FutureLeakageError) as excinfo:
        check_no_future_leakage(
            factor_fn=factor_with_leakage_future_return,
            data=data,
            n_test_points=10,
            tolerance=1e-8
        )

    assert "leakage" in str(excinfo.value).lower(), "Error message should mention leakage"
    logger.info("✅ Test passed: Future return leakage was detected")


def test_leakage_detected_future_high():
    """Test that future high factor DOES trigger leakage detection."""
    data = generate_sample_ohlcv_data(n_periods=500)

    # Should raise FutureLeakageError
    with pytest.raises(FutureLeakageError) as excinfo:
        check_no_future_leakage(
            factor_fn=factor_with_leakage_future_high,
            data=data,
            n_test_points=10,
            tolerance=1e-8
        )

    assert "leakage" in str(excinfo.value).lower(), "Error message should mention leakage"
    logger.info("✅ Test passed: Future high leakage was detected")


def test_leakage_detected_centered_window():
    """Test that centered window (includes future) DOES trigger leakage detection."""
    data = generate_sample_ohlcv_data(n_periods=500)

    # Should raise FutureLeakageError
    with pytest.raises(FutureLeakageError) as excinfo:
        check_no_future_leakage(
            factor_fn=factor_with_leakage_zscore_including_future,
            data=data,
            n_test_points=10,
            tolerance=1e-8
        )

    assert "leakage" in str(excinfo.value).lower(), "Error message should mention leakage"
    logger.info("✅ Test passed: Centered window leakage was detected")


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

def test_constant_factor_no_leakage():
    """Test that a constant factor (always returns same value) passes."""
    data = generate_sample_ohlcv_data(n_periods=500)

    def constant_factor(data: pd.DataFrame) -> pd.Series:
        """Factor that returns constant value."""
        return pd.Series(1.0, index=data.index)

    result = check_no_future_leakage(
        factor_fn=constant_factor,
        data=data,
        n_test_points=10,
        tolerance=1e-8
    )

    assert result is True, "Constant factor should pass (no leakage)"
    logger.info("✅ Test passed: Constant factor has no leakage")


def test_very_small_dataset():
    """Test behavior with very small dataset."""
    data = generate_sample_ohlcv_data(n_periods=50)

    # Should still work with small dataset
    result = check_no_future_leakage(
        factor_fn=factor_no_leakage_simple_sma,
        data=data,
        n_test_points=5,
        tolerance=1e-8
    )

    assert result is True, "Should handle small dataset"
    logger.info("✅ Test passed: Small dataset handled correctly")


def test_custom_test_timestamps():
    """Test using custom test timestamps instead of random sampling."""
    data = generate_sample_ohlcv_data(n_periods=500)

    # Select specific timestamps from middle of data
    test_timestamps = data.index[100:110]

    result = check_no_future_leakage(
        factor_fn=factor_no_leakage_simple_sma,
        data=data,
        test_timestamps=test_timestamps,
        tolerance=1e-8
    )

    assert result is True, "Should work with custom timestamps"
    logger.info("✅ Test passed: Custom timestamps work correctly")


def test_tolerance_sensitivity():
    """Test that tolerance parameter works correctly."""
    data = generate_sample_ohlcv_data(n_periods=500)

    # With very strict tolerance, numerical errors might be detected
    # But for non-leaky factor, should still pass
    result = check_no_future_leakage(
        factor_fn=factor_no_leakage_simple_sma,
        data=data,
        n_test_points=10,
        tolerance=1e-10  # Very strict
    )

    assert result is True, "Should pass even with strict tolerance"
    logger.info("✅ Test passed: Strict tolerance works correctly")


# =============================================================================
# TESTS FOR check_factor_dict_for_leakage
# =============================================================================

def test_factor_dict_mixed_leakage():
    """Test checking multiple factors with mixed leakage."""
    data = generate_sample_ohlcv_data(n_periods=500)

    # Create list of factors with mixed leakage
    compiled_factors = [
        {'name': 'sma_no_leak', 'callable': factor_no_leakage_simple_sma},
        {'name': 'momentum_no_leak', 'callable': factor_no_leakage_momentum},
        {'name': 'future_return_WITH_LEAK', 'callable': factor_with_leakage_future_return},
        {'name': 'rsi_no_leak', 'callable': factor_no_leakage_rsi},
    ]

    # Check all factors with discard_on_failure=True
    passed_factors, results = check_factor_dict_for_leakage(
        compiled_factors=compiled_factors,
        ticker_data=data,
        n_test_points=5,
        discard_on_failure=True
    )

    # Should have 3 passed factors (all except the one with leakage)
    assert len(passed_factors) == 3, f"Should have 3 passed factors, got {len(passed_factors)}"

    # Check specific results
    assert results['sma_no_leak'] is True, "SMA should pass"
    assert results['momentum_no_leak'] is True, "Momentum should pass"
    assert results['future_return_WITH_LEAK'] is False, "Future return should fail (leakage detected)"
    assert results['rsi_no_leak'] is True, "RSI should pass"

    # Check that the leaky factor was removed
    passed_names = [f['name'] for f in passed_factors]
    assert 'future_return_WITH_LEAK' not in passed_names, "Leaky factor should be removed"

    logger.info("✅ Test passed: Mixed leakage correctly identified")


def test_factor_dict_all_pass():
    """Test checking multiple factors where all pass."""
    data = generate_sample_ohlcv_data(n_periods=500)

    # Create list of factors WITHOUT leakage
    compiled_factors = [
        {'name': 'sma', 'callable': factor_no_leakage_simple_sma},
        {'name': 'momentum', 'callable': factor_no_leakage_momentum},
        {'name': 'rsi', 'callable': factor_no_leakage_rsi},
    ]

    passed_factors, results = check_factor_dict_for_leakage(
        compiled_factors=compiled_factors,
        ticker_data=data,
        n_test_points=5,
        discard_on_failure=True
    )

    # All should pass
    assert len(passed_factors) == 3, "All factors should pass"
    assert all(results.values()), "All results should be True"

    logger.info("✅ Test passed: All clean factors correctly passed")


def test_factor_dict_all_fail():
    """Test checking multiple factors where all have leakage."""
    data = generate_sample_ohlcv_data(n_periods=500)

    # Create list of factors WITH leakage
    compiled_factors = [
        {'name': 'future_return', 'callable': factor_with_leakage_future_return},
        {'name': 'future_high', 'callable': factor_with_leakage_future_high},
        {'name': 'centered_zscore', 'callable': factor_with_leakage_zscore_including_future},
    ]

    passed_factors, results = check_factor_dict_for_leakage(
        compiled_factors=compiled_factors,
        ticker_data=data,
        n_test_points=5,
        discard_on_failure=True
    )

    # None should pass
    assert len(passed_factors) == 0, "No factors should pass"
    assert all(v is False for v in results.values()), "All results should be False"

    logger.info("✅ Test passed: All leaky factors correctly detected")


def test_factor_dict_discard_false():
    """Test that discard_on_failure=False keeps all factors."""
    data = generate_sample_ohlcv_data(n_periods=500)

    compiled_factors = [
        {'name': 'sma', 'callable': factor_no_leakage_simple_sma},
        {'name': 'future_return', 'callable': factor_with_leakage_future_return},
    ]

    # Check with discard_on_failure=False
    returned_factors, results = check_factor_dict_for_leakage(
        compiled_factors=compiled_factors,
        ticker_data=data,
        n_test_points=5,
        discard_on_failure=False
    )

    # Should return all factors (not filtered)
    assert len(returned_factors) == 2, "Should keep all factors when discard_on_failure=False"

    # But results should still show which passed/failed
    assert results['sma'] is True, "SMA should pass"
    assert results['future_return'] is False, "Future return should fail"

    logger.info("✅ Test passed: discard_on_failure=False works correctly")


# =============================================================================
# STATISTICAL VALIDATION TESTS
# =============================================================================

def test_perturbation_magnitude():
    """Test that perturbations are actually significant."""
    data = generate_sample_ohlcv_data(n_periods=500)

    # Test with factor that definitely uses future data
    with pytest.raises(FutureLeakageError):
        check_no_future_leakage(
            factor_fn=factor_with_leakage_future_return,
            data=data,
            n_test_points=10,
            perturbation_scale=0.1,  # 10% perturbation
            tolerance=1e-8
        )

    # Even with smaller perturbation, should still detect
    with pytest.raises(FutureLeakageError):
        check_no_future_leakage(
            factor_fn=factor_with_leakage_future_return,
            data=data,
            n_test_points=10,
            perturbation_scale=0.01,  # 1% perturbation
            tolerance=1e-8
        )

    logger.info("✅ Test passed: Perturbation magnitude works correctly")


def test_multiple_runs_consistency():
    """Test that checker gives consistent results across runs."""
    data = generate_sample_ohlcv_data(n_periods=500)

    # Run check multiple times on same factor
    results = []
    for _ in range(5):
        try:
            result = check_no_future_leakage(
                factor_fn=factor_no_leakage_simple_sma,
                data=data,
                n_test_points=10,
                tolerance=1e-8
            )
            results.append(True)
        except FutureLeakageError:
            results.append(False)

    # All runs should give same result (True - no leakage)
    assert all(results), "Should consistently pass across multiple runs"

    # Test with leaky factor
    results_leaky = []
    for _ in range(5):
        try:
            result = check_no_future_leakage(
                factor_fn=factor_with_leakage_future_return,
                data=data,
                n_test_points=10,
                tolerance=1e-8
            )
            results_leaky.append(True)
        except FutureLeakageError:
            results_leaky.append(False)

    # All runs should detect leakage (False)
    assert not any(results_leaky), "Should consistently detect leakage across multiple runs"

    logger.info("✅ Test passed: Checker is consistent across runs")


# =============================================================================
# RUN ALL TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
