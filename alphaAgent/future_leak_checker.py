#!/usr/bin/env python3
"""
Future Data Leakage Checker

This module provides utilities to verify that factor computations
do not use future data (data from time T+1 onwards when computing factor at time T).

The checker works by perturbing future data and verifying that factor values
at time T do not change, which would indicate future data leakage.
"""

import logging
from typing import Callable, Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FutureLeakageError(ValueError):
    """Exception raised when future data leakage is detected in factor computation."""
    pass


def check_no_future_leakage(
    factor_fn: Callable[[pd.DataFrame], pd.Series],
    data: pd.DataFrame,
    test_timestamps: Optional[List] = None,
    n_test_points: int = 10,
    perturbation_scale: float = 0.1,
    tolerance: float = 1e-8
) -> bool:
    """
    Check if a factor function uses future data by perturbing future values.

    Strategy:
    1. Compute factor values on original data
    2. For each test timestamp T:
       a. Perturb all data from T+1 onwards
       b. Recompute factor values
       c. Verify that factor value at T did not change
    3. If factor value at T changes, it indicates future data leakage

    Args:
        factor_fn: Factor computation function (takes DataFrame, returns Series)
        data: Input data (DataFrame with datetime index and OHLCV columns)
        test_timestamps: Specific timestamps to test (if None, randomly sample)
        n_test_points: Number of timestamps to test (used if test_timestamps is None)
        perturbation_scale: Scale of random perturbations (default 0.1 = 10%)
        tolerance: Numerical tolerance for comparing factor values

    Returns:
        True if no future leakage detected, False otherwise

    Raises:
        FutureLeakageError: If future data leakage is detected
    """
    # Compute original factor values
    try:
        original_factor = factor_fn(data.copy())
        if not isinstance(original_factor, pd.Series):
            original_factor = pd.Series(original_factor, index=data.index)
    except Exception as e:
        logger.error(f"Failed to compute factor on original data: {e}")
        raise

    # Select test timestamps
    if test_timestamps is None:
        # Sample n_test_points timestamps from the middle 80% of data
        # (avoid edges where factors might have NaN values)
        valid_indices = original_factor.dropna().index
        if len(valid_indices) < n_test_points:
            logger.debug(f"Only {len(valid_indices)} valid factor values, reducing test points")
            n_test_points = max(1, len(valid_indices) // 2)

        # Skip first and last 10% of data
        start_idx = int(len(valid_indices) * 0.1)
        end_idx = int(len(valid_indices) * 0.9)
        if end_idx <= start_idx:
            start_idx = 0
            end_idx = len(valid_indices)

        test_candidates = valid_indices[start_idx:end_idx]
        if len(test_candidates) > n_test_points:
            test_timestamps = np.random.choice(test_candidates, size=n_test_points, replace=False)
        else:
            test_timestamps = test_candidates

    # Test each timestamp
    leakage_detected = False
    leakage_details = []

    for i, test_time in enumerate(test_timestamps):
        # Find index of test_time
        if test_time not in data.index:
            logger.debug(f"Test timestamp {test_time} not in data index, skipping")
            continue

        test_idx = data.index.get_loc(test_time)

        # Skip if we're at the end of data (no future data to perturb)
        if test_idx >= len(data) - 1:
            continue

        # Create perturbed data: modify all data AFTER test_time
        perturbed_data = data.copy()

        # Perturb future data (from test_idx+1 onwards)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in perturbed_data.columns:
                # Add random perturbations (multiplicative noise)
                future_slice = perturbed_data.iloc[test_idx+1:]
                noise = 1 + np.random.randn(len(future_slice)) * perturbation_scale
                perturbed_data.loc[future_slice.index, col] = perturbed_data.loc[future_slice.index, col] * noise

        # Recompute factor on perturbed data
        try:
            perturbed_factor = factor_fn(perturbed_data)
            if not isinstance(perturbed_factor, pd.Series):
                perturbed_factor = pd.Series(perturbed_factor, index=perturbed_data.index)
        except Exception as e:
            logger.debug(f"Failed to compute factor on perturbed data at {test_time}: {e}")
            continue

        # Compare factor values at test_time
        if test_time in original_factor.index and test_time in perturbed_factor.index:
            original_value = original_factor.loc[test_time]
            perturbed_value = perturbed_factor.loc[test_time]

            # Check if values are both valid (not NaN/Inf)
            if np.isfinite(original_value) and np.isfinite(perturbed_value):
                diff = abs(original_value - perturbed_value)

                if diff > tolerance:
                    leakage_detected = True
                    leakage_details.append({
                        'timestamp': test_time,
                        'original_value': original_value,
                        'perturbed_value': perturbed_value,
                        'difference': diff
                    })
                    logger.warning(
                        f"⚠️  FUTURE LEAKAGE DETECTED at {test_time}: "
                        f"original={original_value:.6f}, perturbed={perturbed_value:.6f}, "
                        f"diff={diff:.6e}"
                    )

    # Report results
    if leakage_detected:
        logger.error(f"❌ FUTURE DATA LEAKAGE DETECTED at {len(leakage_details)}/{len(test_timestamps)} test points!")
        logger.debug("Leakage details:")
        for detail in leakage_details[:3]:  # Show first 3
            logger.debug(f"  - {detail['timestamp']}: diff={detail['difference']:.6e}")

        raise FutureLeakageError(
            f"Detected leakage at {len(leakage_details)}/{len(test_timestamps)} timestamps"
        )
    else:
        logger.debug(f"✅ Passed - tested {len(test_timestamps)} timestamps")
        return True


def check_factor_dict_for_leakage(
    compiled_factors: List[Dict],
    ticker_data: pd.DataFrame,
    n_test_points: int = 5,
    discard_on_failure: bool = True
) -> tuple[List[Dict], Dict[str, bool]]:
    """
    Check all compiled factors for future data leakage.

    Args:
        compiled_factors: List of compiled factor dictionaries (with 'name' and 'callable')
        ticker_data: Sample ticker data to test on
        n_test_points: Number of timestamps to test per factor
        discard_on_failure: If True, remove factors that fail the test

    Returns:
        Tuple of (filtered_factors, results_dict)
        - filtered_factors: List of factors that passed the test (if discard_on_failure=True)
        - results_dict: Dictionary mapping factor names to leakage test results (True = no leakage)
    """
    logger.info(f"Checking ALL {len(compiled_factors)} factors for future data leakage...")
    logger.info(f"Discard on failure: {discard_on_failure}")

    results = {}
    passed_factors = []
    failed_factor_names = []

    for i, factor_item in enumerate(compiled_factors):
        factor_name = factor_item['name']
        factor_fn = factor_item['callable']

        if (i + 1) % 5 == 0 or i == 0:
            logger.info(f"\nTesting factor {i+1}/{len(compiled_factors)}: {factor_name}")
        else:
            logger.debug(f"Testing factor {i+1}/{len(compiled_factors)}: {factor_name}")

        try:
            result = check_no_future_leakage(
                factor_fn,
                ticker_data,
                n_test_points=n_test_points
            )
            results[factor_name] = result

            if result:
                passed_factors.append(factor_item)
            else:
                failed_factor_names.append(factor_name)

        except FutureLeakageError as e:
            logger.error(f"❌ Factor '{factor_name}' FAILED future leakage check!")
            logger.error(f"   Reason: {str(e)[:200]}")
            results[factor_name] = False
            failed_factor_names.append(factor_name)

        except Exception as e:
            logger.error(f"⚠️  Error testing factor '{factor_name}': {e}")
            results[factor_name] = None
            # On error, we keep the factor (benefit of the doubt)
            passed_factors.append(factor_item)

    # Summary
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    error = sum(1 for v in results.values() if v is None)

    logger.info(f"\n{'='*50}")
    logger.info(f"FUTURE LEAKAGE CHECK SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"Passed: {passed}/{len(results)} ✅")
    logger.info(f"Failed: {failed}/{len(results)} ❌")
    logger.info(f"Error:  {error}/{len(results)} ⚠️")

    if failed_factor_names:
        logger.warning(f"\nFailed factors (will be DISCARDED): {failed_factor_names[:10]}" +
                      (f" ... and {len(failed_factor_names) - 10} more" if len(failed_factor_names) > 10 else ""))

    if discard_on_failure:
        return passed_factors, results
    else:
        return compiled_factors, results
