#!/usr/bin/env python3
"""
Lookahead-Free Test Suite for AlphaAgent Operators

This script tests whether all operators in alpha_agent_operators.py are strictly lookahead-free.
An operator is lookahead-free if the output at time t only depends on data at times <= t.

Test methodology:
1. Create synthetic time series data with known structure
2. Apply each operator and capture output at specific time t
3. Modify future data (times > t) and re-run the operator
4. If output at time t changes, the operator has lookahead bias (TEST FAILS)
5. If output at time t remains unchanged, the operator is lookahead-free (TEST PASSES)
"""

import numpy as np
import pandas as pd
import sys
from typing import Callable, Dict, List, Tuple, Any
from alpha_agent_operators import Operators

# Test configuration
TEST_T = 100  # Time point to test (middle of series)
FUTURE_CONTAMINATION_T = 110  # Future time point to modify
N_ASSETS = 10
N_TIMESTAMPS = 200

class LookaheadTester:
    """Test framework for verifying operators are lookahead-free"""

    def __init__(self, test_t: int = TEST_T, future_t: int = FUTURE_CONTAMINATION_T):
        self.test_t = test_t
        self.future_t = future_t
        self.results = {}

    def create_synthetic_data(self, seed: int = 42) -> Dict[str, pd.DataFrame]:
        """Create synthetic time series data for testing"""
        np.random.seed(seed)

        dates = pd.date_range('2020-01-01', periods=N_TIMESTAMPS, freq='30min')

        # Create various data types
        data = {
            'close': pd.DataFrame(
                np.cumsum(np.random.randn(N_TIMESTAMPS, N_ASSETS) * 0.01, axis=0) + 100,
                index=dates,
                columns=[f'ASSET_{i}' for i in range(N_ASSETS)]
            ),
            'volume': pd.DataFrame(
                np.random.lognormal(10, 1, (N_TIMESTAMPS, N_ASSETS)),
                index=dates,
                columns=[f'ASSET_{i}' for i in range(N_ASSETS)]
            ),
            'high': None,  # Will be derived from close
            'low': None,   # Will be derived from close
            'open': None   # Will be derived from close
        }

        # Derive OHLC from close
        data['high'] = data['close'] * (1 + np.abs(np.random.randn(N_TIMESTAMPS, N_ASSETS) * 0.005))
        data['low'] = data['close'] * (1 - np.abs(np.random.randn(N_TIMESTAMPS, N_ASSETS) * 0.005))
        data['open'] = data['close'].shift(1).fillna(data['close'])

        return data

    def modify_future_data(self, data: Dict[str, pd.DataFrame], magnitude: float = 10.0) -> Dict[str, pd.DataFrame]:
        """Modify data at future time points to test for lookahead"""
        modified_data = {}
        for key, df in data.items():
            modified_df = df.copy()
            # Add large values to future time points
            modified_df.iloc[self.future_t:] = modified_df.iloc[self.future_t:] + magnitude
            modified_data[key] = modified_df
        return modified_data

    def test_operator(
        self,
        op_func: Callable,
        op_args: Tuple,
        op_kwargs: Dict,
        data: Dict[str, pd.DataFrame],
        op_name: str
    ) -> Dict[str, Any]:
        """
        Test if an operator is lookahead-free

        Returns dict with:
            - passed: bool
            - reason: str
            - max_diff: float
        """
        try:
            # Run operator on original data
            result_original = op_func(*op_args, **op_kwargs)

            if not isinstance(result_original, pd.DataFrame):
                result_original = pd.DataFrame(result_original)

            # Get output at test time point
            output_at_t = result_original.iloc[self.test_t].copy()

            # Modify future data
            modified_data = self.modify_future_data(data)

            # Reconstruct arguments with modified data
            modified_args = []
            for arg in op_args:
                if isinstance(arg, pd.DataFrame):
                    # Find which data key this corresponds to
                    for key, df in data.items():
                        if arg is df:
                            modified_args.append(modified_data[key])
                            break
                    else:
                        modified_args.append(arg)
                else:
                    modified_args.append(arg)

            # Run operator on modified data
            result_modified = op_func(*modified_args, **op_kwargs)

            if not isinstance(result_modified, pd.DataFrame):
                result_modified = pd.DataFrame(result_modified)

            # Get output at test time point from modified data
            output_at_t_modified = result_modified.iloc[self.test_t].copy()

            # Compare outputs
            diff = np.abs(output_at_t - output_at_t_modified)
            max_diff = float(np.nanmax(diff))

            # Test passes if outputs are identical (within numerical precision)
            tolerance = 1e-10
            passed = max_diff < tolerance

            if passed:
                reason = f"✓ Lookahead-free: Output at t={self.test_t} unchanged when future modified (max_diff={max_diff:.2e})"
            else:
                reason = f"✗ LOOKAHEAD DETECTED: Output at t={self.test_t} changed by {max_diff:.6f} when future modified"

            return {
                'passed': passed,
                'reason': reason,
                'max_diff': max_diff
            }

        except Exception as e:
            return {
                'passed': False,
                'reason': f"✗ ERROR: {str(e)}",
                'max_diff': np.nan
            }

    def run_all_tests(self) -> Dict[str, Dict]:
        """Run lookahead tests for all operators"""
        data = self.create_synthetic_data()

        print("=" * 80)
        print("LOOKAHEAD-FREE TEST SUITE FOR ALPHA AGENT OPERATORS")
        print("=" * 80)
        print(f"Testing at time t={self.test_t}, modifying future at t={self.future_t}")
        print(f"Data shape: {N_TIMESTAMPS} timestamps × {N_ASSETS} assets")
        print("=" * 80)
        print()

        # Define test cases for each operator
        test_cases = [
            # Time-Series Operators
            ('delay', Operators.delay, (data['close'], 5), {}),
            ('diff', Operators.diff, (data['close'], 5), {}),
            ('ts_mean', Operators.ts_mean, (data['close'], 20), {}),
            ('ts_std', Operators.ts_std, (data['close'], 20), {}),
            ('ts_sum', Operators.ts_sum, (data['close'], 20), {}),
            ('ts_min', Operators.ts_min, (data['close'], 20), {}),
            ('ts_max', Operators.ts_max, (data['close'], 20), {}),
            ('ts_argmin', Operators.ts_argmin, (data['close'], 20), {}),
            ('ts_argmax', Operators.ts_argmax, (data['close'], 20), {}),
            ('ema', Operators.ema, (data['close'], 20), {}),
            ('wma', Operators.wma, (data['close'], 20), {}),
            ('ts_zscore', Operators.ts_zscore, (data['close'], 20), {}),
            ('ts_pctile', Operators.ts_pctile, (data['close'], 20, 0.5), {}),
            ('ts_rank', Operators.ts_rank, (data['close'], 20), {}),
            ('corr_ts', Operators.corr_ts, (data['close'], data['volume'], 20), {}),
            ('cov_ts', Operators.cov_ts, (data['close'], data['volume'], 20), {}),
            ('beta_ts', Operators.beta_ts, (data['close'], data['volume'], 20), {}),
            ('volatility', Operators.volatility, (data['close'], 20), {}),
            ('atr', Operators.atr, (data['high'], data['low'], data['close'], 20), {}),
            ('breakout', Operators.breakout, (data['close'], 20, 'up'), {}),
            ('persist', Operators.persist, ((data['close'] > data['close'].shift(1)), 20), {}),

            # Cross-Sectional Operators
            ('cs_rank', Operators.cs_rank, (data['close'],), {}),
            ('cs_zscore', Operators.cs_zscore, (data['close'],), {}),
            ('cs_winsor', Operators.cs_winsor, (data['close'], 0.05), {}),
            ('cs_rescale_l2', Operators.cs_rescale, (data['close'],), {'norm': 'l2'}),

            # Elementwise/Algebraic
            ('add', Operators.add, (data['close'], data['volume']), {}),
            ('sub', Operators.sub, (data['close'], data['volume']), {}),
            ('mul', Operators.mul, (data['close'], data['volume']), {}),
            ('div', Operators.div, (data['close'], data['volume']), {}),
            ('abs', Operators.abs, (data['close'] - 100,), {}),
            ('sign', Operators.sign, (data['close'] - 100,), {}),
            ('clip', Operators.clip, (data['close'], 95, 105), {}),
            ('pow', Operators.pow, (data['close'] / 100, 2), {}),
            ('log1p', Operators.log1p, (data['close'] - 100,), {}),
            ('sqrt', Operators.sqrt, (data['close'],), {}),
            ('tanh', Operators.tanh, ((data['close'] - 100) / 10,), {}),
            ('gt', Operators.gt, (data['close'], 100), {}),
            ('lt', Operators.lt, (data['close'], 100), {}),
            ('between', Operators.between, (data['close'], 95, 105), {}),

            # Price/Volume Specific
            ('turnover', Operators.turnover, (data['volume'], 20), {}),
            ('adv', Operators.adv, (data['volume'], 20), {}),
            ('amihud', Operators.amihud, (data['close'].pct_change(), data['volume'], 20), {}),

            # Event/Pattern
            ('crossover', Operators.crossover, (data['close'], data['close'].rolling(10).mean()), {}),
            ('crossunder', Operators.crossunder, (data['close'], data['close'].rolling(10).mean()), {}),

            # Signal Hygiene
            ('demean_ts', Operators.demean_ts, (data['close'], 20), {}),
            ('debias_cs', Operators.debias_cs, (data['close'],), {}),
            ('smooth_ts', Operators.smooth_ts, (data['close'], 20), {'method': 'ema'}),
        ]

        results = {}
        passed_count = 0
        failed_count = 0

        for op_name, op_func, op_args, op_kwargs in test_cases:
            result = self.test_operator(op_func, op_args, op_kwargs, data, op_name)
            results[op_name] = result

            # Print result
            status = "PASS" if result['passed'] else "FAIL"
            print(f"[{status}] {op_name:20s} | {result['reason']}")

            if result['passed']:
                passed_count += 1
            else:
                failed_count += 1

        print()
        print("=" * 80)
        print(f"SUMMARY: {passed_count} passed, {failed_count} failed out of {len(test_cases)} tests")
        print("=" * 80)

        self.results = results
        return results


def main():
    """Run the test suite"""
    tester = LookaheadTester()
    results = tester.run_all_tests()

    # Exit with error code if any tests failed
    failed_tests = [name for name, result in results.items() if not result['passed']]

    if failed_tests:
        print("\n⚠️  WARNING: The following operators have lookahead bias:")
        for name in failed_tests:
            print(f"  - {name}")
        print("\nThese operators must be fixed before use in production!")
        sys.exit(1)
    else:
        print("\n✅ All operators are lookahead-free!")
        sys.exit(0)


if __name__ == "__main__":
    main()
