#!/usr/bin/env python3
"""
Pipeline execution module for Alpha Agent Factor Pipeline

This module contains the main pipeline logic for:
- Loading hypotheses from JSON files
- Processing factors for each hypothesis
- Computing metrics across all tickers
- AST-based iterative refinement
"""

import os
import json
import glob
import logging
import gc
import multiprocessing as mp
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from config import PDF_RESULTS_PATH, OUTPUT_PATH, ANNUALIZE_PERIODS
from factor_agent import FactorAgent
from data_loader import load_real_market_data
from ast_compiler import FactorSpec
from validator import dedupe_factor_names
from factor_evaluation import (
    to_dense_from_prices,
    compute_ic_series_dense,
    compute_sharpe_dense,
    compute_sharpe_multiple_lags,
    save_factor_pnl_visualization,
    _nw_tstat_from_series
)
from future_leak_checker import check_factor_dict_for_leakage

logger = logging.getLogger(__name__)


def load_hypotheses(pdf_results_path: str = PDF_RESULTS_PATH) -> List[Dict]:
    """Load all hypotheses from JSON files in pdf_results directory.

    Args:
        pdf_results_path: Path to directory containing hypothesis JSON files

    Returns:
        List of hypothesis dictionaries
    """
    logger.info("\n" + "="*50)
    logger.info("LOADING HYPOTHESES FROM PDF_RESULTS")
    logger.info("="*50)

    hypothesis_json_files = glob.glob(os.path.join(pdf_results_path, "*hypotheses*.json"))

    if not hypothesis_json_files:
        logger.error(f"No hypothesis JSON files found in {pdf_results_path}")
        raise ValueError("No hypothesis JSON files found")

    logger.info(f"Found {len(hypothesis_json_files)} hypothesis JSON files:")
    for json_file in hypothesis_json_files:
        logger.info(f"  - {os.path.basename(json_file)}")

    # Load all hypotheses from JSON files
    all_hypotheses = []
    for json_file in hypothesis_json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                # Handle different JSON structures
                if isinstance(data, list):
                    for paper in data:
                        if 'hypotheses' in paper:
                            for hyp in paper['hypotheses']:
                                all_hypotheses.append({
                                    'hypothesis_text': hyp.get('hypothesis', ''),
                                    'reasoning': hyp.get('reasoning', ''),
                                    'paper_title': paper.get('paper_title', 'Unknown'),
                                    'source_file': os.path.basename(json_file)
                                })
                elif isinstance(data, dict) and 'hypotheses' in data:
                    for hyp in data['hypotheses']:
                        all_hypotheses.append({
                            'hypothesis_text': hyp.get('hypothesis', ''),
                            'reasoning': hyp.get('reasoning', ''),
                            'paper_title': data.get('paper_title', 'Unknown'),
                            'source_file': os.path.basename(json_file)
                        })
            logger.info(f"  Loaded hypotheses from {os.path.basename(json_file)}")
        except Exception as e:
            logger.error(f"  Error loading {json_file}: {e}")

    logger.info(f"\nTotal hypotheses loaded: {len(all_hypotheses)}")

    if not all_hypotheses:
        logger.error("No valid hypotheses found in JSON files")
        raise ValueError("No valid hypotheses found")

    return all_hypotheses


def process_single_hypothesis(agent: FactorAgent, hypothesis: str, reasoning: str,
                              hypothesis_idx: int, tickers: List[str], ticker_data_dict: Dict,
                              output_path: str = OUTPUT_PATH) -> None:
    """Process a single hypothesis through the pipeline.

    Args:
        agent: Initialized FactorAgent
        hypothesis: Hypothesis text to process
        reasoning: Reasoning for the hypothesis
        hypothesis_idx: Index of hypothesis (for saving ASTs)
        tickers: List of ticker symbols
        ticker_data_dict: Dictionary mapping tickers to DataFrames
        output_path: Path to save output files
    """
    # Define path for saving/loading ASTs (unique per hypothesis)
    ast_save_path = os.path.join(output_path, f"saved_asts_hypothesis_{hypothesis_idx + 1}.json")
    loaded_from_previous = False

    # Check if we should load from previous iteration (file exists)
    if os.path.exists(ast_save_path):
        logger.info(f"Loading ASTs from previous iteration: {ast_save_path}")
        with open(ast_save_path, 'r') as f:
            saved_data = json.load(f)

        # Reconstruct specs from saved ASTs
        specs = []
        for item in saved_data:
            specs.append(FactorSpec(
                name=item['name'],
                description=item['description'],
                reasoning=item['reasoning'],
                ast=item['ast']
            ))
        logger.info(f"Loaded {len(specs)} factor specifications from saved ASTs")

        # Compile the loaded factors
        compiled_factors = agent.compile_factors(specs)
        logger.info(f"Compiled {len(compiled_factors)} factors from saved ASTs")
        loaded_from_previous = True
    else:
        # First iteration: generate new factors
        logger.info(f"Parsing hypothesis: {hypothesis}")
        logger.info(f"Reasoning: {reasoning}")
        specs = agent.parse_hypothesis(hypothesis, reasoning)
        logger.info(f"Parsed {len(specs)} factor specifications")
        compiled_factors = agent.compile_factors(specs)
        logger.info(f"Compiled {len(compiled_factors)} factors")

        # Save ASTs for next iteration
        logger.info(f"Saving ASTs to: {ast_save_path}")
        ast_data = []
        for item in compiled_factors:
            ast_data.append({
                'name': item['name'],
                'description': item['description'],
                'reasoning': item['reasoning'],
                'ast': item['ast']
            })
        with open(ast_save_path, 'w') as f:
            json.dump(ast_data, f, indent=2)
        logger.info(f"Saved {len(ast_data)} ASTs for next iteration")

    # Deduplicate factor names to prevent duplicate columns
    compiled_factors = dedupe_factor_names(compiled_factors)
    logger.info(f"Deduplicated factor names")

    # Validate no duplicate names remain
    names = [f["name"] for f in compiled_factors]
    if len(names) != len(set(names)):
        duplicates = [n for n in names if names.count(n) > 1]
        raise ValueError(f"Duplicate factor names detected after deduplication: {set(duplicates)}")

    # Check for future data leakage before computing metrics
    logger.info("\n" + "="*50)
    logger.info("CHECKING FOR FUTURE DATA LEAKAGE")
    logger.info("="*50)

    # Use first ticker's data for leakage testing
    if tickers and tickers[0] in ticker_data_dict:
        sample_ticker = tickers[0]
        sample_data = ticker_data_dict[sample_ticker]
        logger.info(f"Using {sample_ticker} data for future leakage testing")

        try:
            # Test ALL factors and discard those that fail
            original_count = len(compiled_factors)
            compiled_factors, leakage_results = check_factor_dict_for_leakage(
                compiled_factors,
                sample_data,
                n_test_points=5,
                discard_on_failure=True
            )

            # Report filtering results
            discarded_count = original_count - len(compiled_factors)
            if discarded_count > 0:
                logger.warning(f"🗑️  DISCARDED {discarded_count} factors that failed future leakage check")
                logger.warning(f"   Remaining factors: {len(compiled_factors)}/{original_count}")
            else:
                logger.info(f"✅ All {original_count} factors passed future leakage check")

            # If all factors were discarded, raise error
            if len(compiled_factors) == 0:
                raise ValueError("All factors failed future leakage check! No factors to evaluate.")

        except ValueError as e:
            # Re-raise ValueError (critical errors like all factors discarded)
            raise
        except Exception as e:
            logger.error(f"Error during future leakage check: {e}")
            logger.warning("Continuing with pipeline despite leakage check failure")
    else:
        logger.warning("No ticker data available for future leakage testing - skipping")

    # Compute factors and metrics (only on factors that passed leakage test)
    compute_factors_and_metrics(agent, compiled_factors, specs, tickers, ticker_data_dict,
                               loaded_from_previous, ast_save_path, output_path, hypothesis_idx)


def _compute_factor_for_ticker(args: Tuple) -> Tuple[str, Optional[pd.Series]]:
    """Worker function to compute a single factor for a single ticker.

    Args:
        args: Tuple of (ticker, ticker_data, factor_callable, factor_name)

    Returns:
        Tuple of (ticker, factor_series) where factor_series is None on error
    """
    ticker, ticker_data, factor_callable, factor_name = args

    if ticker_data is None:
        return (ticker, None)

    try:
        # Apply factor to entire time series
        factor_series = factor_callable(ticker_data)

        # Ensure factor_series is a Series with proper index
        if isinstance(factor_series, pd.DataFrame):
            if factor_series.shape[1] == 1:
                factor_series = factor_series.iloc[:, 0]
            else:
                factor_series = factor_series.mean(axis=1)

        if not isinstance(factor_series, pd.Series):
            factor_series = pd.Series(factor_series, index=ticker_data.index)

        return (ticker, factor_series)

    except Exception as e:
        # Log the error for debugging (only log first few to avoid spam)
        import traceback
        logger.debug(f"Error computing {factor_name} for {ticker}: {e}\n{traceback.format_exc()}")
        # Return None to indicate error - will be handled in main process
        return (ticker, None)


def _compute_factor_metrics(compiled_factors: List[Dict], tickers: List[str],
                           ticker_data_dict: Dict, price_data_dict: Dict,
                           n_workers: int = 20) -> List[Dict]:
    """Compute metrics for all factors.

    Args:
        compiled_factors: List of compiled factor dictionaries
        tickers: List of ticker symbols
        ticker_data_dict: Dictionary mapping tickers to DataFrames
        price_data_dict: Dictionary mapping tickers to price Series
        n_workers: Number of workers for parallel computation

    Returns:
        List of factor metric dictionaries
    """
    factor_metrics = []
    n_factors = len(compiled_factors)

    logger.info(f"Computing metrics for {n_factors} factors across {len(tickers)} tickers...")

    for factor_idx, factor_item in enumerate(compiled_factors):
        factor_name = factor_item["name"]
        logger.info(f"\nProcessing factor {factor_idx+1}/{n_factors}: {factor_name}")

        # Compute this factor for all tickers IN PARALLEL
        factor_results_single = {}  # {ticker: Series} - only for current factor

        # Prepare arguments for parallel processing
        args_list = [
            (ticker, ticker_data_dict.get(ticker), factor_item["callable"], factor_name)
            for ticker in tickers
        ]

        logger.info(f"  Computing {factor_name} for {len(tickers)} tickers using {n_workers} workers...")

        # Use multiprocessing Pool to parallelize across tickers
        try:
            with mp.Pool(processes=n_workers) as pool:
                results = pool.map(_compute_factor_for_ticker, args_list)

            # Process results
            error_count = 0
            for ticker, factor_series in results:
                if factor_series is None:
                    # Error occurred, store zeros
                    ticker_data = ticker_data_dict.get(ticker)
                    if ticker_data is not None:
                        factor_results_single[ticker] = pd.Series(0.0, index=ticker_data.index)
                    error_count += 1
                else:
                    factor_results_single[ticker] = factor_series

            if error_count > 0:
                logger.warning(f"  {error_count} tickers had errors computing {factor_name}")

        except (OSError, RuntimeError) as e:
            # Fallback to sequential processing if multiprocessing fails
            logger.warning(f"Multiprocessing failed for {factor_name} ({e}), falling back to sequential processing")

            for ticker_idx, ticker in enumerate(tickers):
                if ticker_idx % 500 == 0:
                    logger.info(f"  Computing {factor_name} for ticker {ticker_idx+1}/{len(tickers)}...")

                ticker_data = ticker_data_dict.get(ticker)
                if ticker_data is None:
                    continue

                try:
                    # Apply factor to entire time series
                    factor_series = factor_item["callable"](ticker_data)

                    # Ensure factor_series is a Series with proper index
                    if isinstance(factor_series, pd.DataFrame):
                        if factor_series.shape[1] == 1:
                            factor_series = factor_series.iloc[:, 0]
                        else:
                            factor_series = factor_series.mean(axis=1)

                    if not isinstance(factor_series, pd.Series):
                        factor_series = pd.Series(factor_series, index=ticker_data.index)

                    # Store the factor series for this ticker
                    factor_results_single[ticker] = factor_series

                except Exception as e:
                    logger.warning(f"  Error computing {factor_name} for {ticker}: {e}")
                    # Store zeros on error
                    factor_results_single[ticker] = pd.Series(0.0, index=ticker_data.index)

                # Periodic garbage collection
                if (ticker_idx + 1) % 100 == 0:
                    gc.collect()

        logger.info(f"  ✅ Computed {factor_name} for {len(factor_results_single)} tickers (parallel execution)")

        # Compute metrics for this factor
        logger.info(f"  Computing metrics for {factor_name}...")

        # Convert to dense matrices for efficient computation with 30-minute resampling
        # Uses provably correct alignment: resample prices -> compute returns -> resample factors -> intersect
        ts_index, tickers_list, F, R = to_dense_from_prices(factor_results_single, price_data_dict, resample_freq='30T')

        # Compute Pearson IC using dense matrix approach
        pearson_ic_ts = compute_ic_series_dense(F, R, ts_index)
        pearson_ic_ts_clean = pearson_ic_ts.replace([np.inf, -np.inf], np.nan).dropna()

        if len(pearson_ic_ts_clean) > 10:
            mean_pearson_ic = float(pearson_ic_ts_clean.mean())
            std_pearson_ic = float(pearson_ic_ts_clean.std(ddof=1))
            pearson_nw_tstat = _nw_tstat_from_series(pearson_ic_ts_clean, maxlags=5)
        else:
            mean_pearson_ic = 0.0
            std_pearson_ic = 0.0
            pearson_nw_tstat = 0.0

        # Compute Sharpe at multiple lags
        minute_lags = [1]
        logger.info(f"  Computing Sharpe at multiple lags: {minute_lags}...")
        try:
            lag_sharpe_results = compute_sharpe_multiple_lags(
                factor_results_single,
                price_data_dict,
                resample_freq='30T',
                minute_lags=minute_lags,
                annualize_periods=ANNUALIZE_PERIODS
            )
            # Extract Sharpe values for each lag
            sharpe_at_lags = {lag: sharpe for lag, (sharpe, _) in lag_sharpe_results.items()}
            # Use the default lag (1 min) as the primary Sharpe
            sharpe_annualized = sharpe_at_lags.get(1, 0.0)
            # Get PnL for visualization from lag 1
            pnl_timeseries = lag_sharpe_results.get(1, (0.0, pd.Series()))[1]
            logger.info(f"  Sharpe at lags: {sharpe_at_lags}")
        except Exception as e:
            logger.warning(f"  Could not compute Sharpe at multiple lags: {e}")
            # Fallback to single Sharpe computation
            sharpe_annualized, pnl_timeseries = compute_sharpe_dense(
                F, R, ts_index,
                annualize_periods=ANNUALIZE_PERIODS
            )
            sharpe_at_lags = {lag: 0.0 for lag in minute_lags}
            sharpe_at_lags[1] = sharpe_annualized

        # Save PnL visualization for this factor
        save_factor_pnl_visualization(
            pnl_timeseries,
            factor_name,
            sharpe_annualized
        )

        n_timestamps = len(pearson_ic_ts_clean)

        # Create metrics dict with Sharpe at all lags
        metrics_dict = {
            'name': factor_name,
            'pearson_ic': mean_pearson_ic,
            'pearson_ic_std': std_pearson_ic,
            'pearson_nw_tstat': pearson_nw_tstat,
            'nw_tstat': pearson_nw_tstat,  # For backward compatibility
            'sharpe': sharpe_annualized,
            'n_observations': n_timestamps
        }

        # Add Sharpe at each lag
        for lag, sharpe_val in sharpe_at_lags.items():
            metrics_dict[f'sharpe_lag_{lag}min'] = sharpe_val

        factor_metrics.append(metrics_dict)

        logger.info(f"  ✅ {factor_name}: Pearson IC={mean_pearson_ic:.4f}±{std_pearson_ic:.4f} (t={pearson_nw_tstat:.2f}), "
                   f"Sharpe={sharpe_annualized:.4f}, n={n_timestamps} timestamps")

        # CRITICAL: Free up factor data immediately after computing metrics
        del factor_results_single, pearson_ic_ts, pearson_ic_ts_clean, pnl_timeseries
        gc.collect()
        logger.info(f"  🗑️  Freed memory for {factor_name}")

    return factor_metrics


def compute_factors_and_metrics(agent: FactorAgent, compiled_factors: List[Dict],
                                specs: List[FactorSpec], tickers: List[str],
                                ticker_data_dict: Dict, loaded_from_previous: bool,
                                ast_save_path: str, output_path: str = OUTPUT_PATH,
                                hypothesis_idx: int = 0) -> None:
    """Compute factors and metrics for all tickers with iterative AST refinement.

    Args:
        agent: Initialized FactorAgent
        compiled_factors: List of compiled factor dictionaries
        specs: List of FactorSpec objects
        tickers: List of ticker symbols
        ticker_data_dict: Dictionary mapping tickers to DataFrames
        loaded_from_previous: Whether ASTs were loaded from previous iteration
        ast_save_path: Path to save ASTs
        output_path: Path to save output files
        hypothesis_idx: Index of hypothesis (for unique file naming)
    """
    logger.info("\n" + "="*50)
    logger.info("ITERATIVE FACTOR REFINEMENT WITH AST")
    logger.info("="*50)

    try:
        n_tickers = len(tickers)
        n_factors = len(compiled_factors)

        logger.info(f"Starting iterative refinement for {n_factors} factors across {n_tickers} tickers")

        # Store price data for all tickers (will compute forward returns during resampling)
        logger.info("Extracting price data for all tickers...")
        price_data_dict = {}  # {ticker: Series}

        for ticker_idx, ticker in enumerate(tickers):
            ticker_data = ticker_data_dict.get(ticker)
            if ticker_data is None:
                continue

            try:
                close_prices = ticker_data['close']
                # Filter out near-zero prices to avoid division issues
                close_prices = close_prices.where(close_prices > 0.01)
                price_data_dict[ticker] = close_prices
            except Exception as e:
                logger.warning(f"  Error extracting prices for {ticker}: {e}")
                continue

        logger.info(f"Extracted prices for {len(price_data_dict)} tickers")

        # Number of workers for parallelization
        n_workers = 20
        n_iterations = 3  # Number of refinement iterations

        # Current state
        current_compiled_factors = compiled_factors
        current_specs = specs

        # Iterative refinement loop
        for iteration in range(n_iterations):
            logger.info("\n" + "="*80)
            logger.info(f"ITERATION {iteration + 1}/{n_iterations}")
            logger.info("="*80)

            # STEP 1: Compute statistics BEFORE refinement
            logger.info(f"\n[Iteration {iteration + 1}] Computing metrics BEFORE refinement...")
            factor_metrics_before = _compute_factor_metrics(
                current_compiled_factors, tickers, ticker_data_dict,
                price_data_dict, n_workers
            )

            # Display summary
            logger.info(f"\n[Iteration {iteration + 1}] Metrics BEFORE refinement:")
            display_summary_metrics(factor_metrics_before, output_path, hypothesis_idx)

            # STEP 2: Perform AST refinement
            logger.info(f"\n[Iteration {iteration + 1}] Performing AST refinement...")
            perform_ast_refinement(agent, current_specs, factor_metrics_before,
                                 loaded_from_previous, ast_save_path)

            # STEP 3: Reload refined factors
            logger.info(f"\n[Iteration {iteration + 1}] Reloading refined factors...")
            try:
                with open(ast_save_path, 'r') as f:
                    saved_data = json.load(f)

                # Reconstruct specs from refined ASTs
                refined_specs = []
                for item in saved_data:
                    refined_specs.append(FactorSpec(
                        name=item['name'],
                        description=item['description'],
                        reasoning=item['reasoning'],
                        ast=item['ast']
                    ))
                logger.info(f"  Loaded {len(refined_specs)} refined factor specifications")

                # Compile the refined factors
                refined_compiled_factors = agent.compile_factors(refined_specs)
                logger.info(f"  Compiled {len(refined_compiled_factors)} refined factors")

                # Deduplicate factor names
                refined_compiled_factors = dedupe_factor_names(refined_compiled_factors)

            except Exception as e:
                logger.error(f"  Error reloading refined factors: {e}")
                logger.warning(f"  Skipping iteration {iteration + 1}")
                break

            # STEP 4: Compute statistics AFTER refinement
            logger.info(f"\n[Iteration {iteration + 1}] Computing metrics AFTER refinement...")
            factor_metrics_after = _compute_factor_metrics(
                refined_compiled_factors, tickers, ticker_data_dict,
                price_data_dict, n_workers
            )

            # Display summary
            logger.info(f"\n[Iteration {iteration + 1}] Metrics AFTER refinement:")
            display_summary_metrics(factor_metrics_after, output_path, hypothesis_idx)

            # STEP 5: Compare before/after metrics for each factor
            logger.info("\n" + "="*80)
            logger.info(f"[Iteration {iteration + 1}] REFINEMENT COMPARISON")
            logger.info("="*80)

            # Create lookup dict for after metrics
            after_metrics_dict = {m['name']: m for m in factor_metrics_after}

            comparison_results = []
            for before_metric in factor_metrics_before:
                factor_name = before_metric['name']
                after_metric = after_metrics_dict.get(factor_name)

                if after_metric:
                    sharpe_before = before_metric['sharpe']
                    sharpe_after = after_metric['sharpe']
                    ic_before = before_metric['pearson_ic']
                    ic_after = after_metric['pearson_ic']

                    sharpe_diff = abs(sharpe_after - sharpe_before)
                    ic_diff = abs(ic_after - ic_before)

                    comparison_results.append({
                        'factor': factor_name,
                        'sharpe_before': sharpe_before,
                        'sharpe_after': sharpe_after,
                        'sharpe_abs_diff': sharpe_diff,
                        'ic_before': ic_before,
                        'ic_after': ic_after,
                        'ic_abs_diff': ic_diff
                    })

            # Log comparison results
            if comparison_results:
                comparison_df = pd.DataFrame(comparison_results)

                logger.info(f"\nTop 10 factors by Sharpe improvement (absolute difference):")
                top_sharpe_improved = comparison_df.nlargest(10, 'sharpe_abs_diff')[
                    ['factor', 'sharpe_before', 'sharpe_after', 'sharpe_abs_diff']
                ]
                logger.info(f"\n{top_sharpe_improved.to_string(index=False)}")

                logger.info(f"\nTop 10 factors by IC improvement (absolute difference):")
                top_ic_improved = comparison_df.nlargest(10, 'ic_abs_diff')[
                    ['factor', 'ic_before', 'ic_after', 'ic_abs_diff']
                ]
                logger.info(f"\n{top_ic_improved.to_string(index=False)}")

                # Overall statistics
                avg_sharpe_improvement = comparison_df['sharpe_abs_diff'].mean()
                avg_ic_improvement = comparison_df['ic_abs_diff'].mean()
                logger.info(f"\nOverall improvements:")
                logger.info(f"  Average Sharpe abs diff: {avg_sharpe_improvement:.6f}")
                logger.info(f"  Average IC abs diff: {avg_ic_improvement:.6f}")

                # Save comparison to CSV
                comparison_csv_path = os.path.join(
                    output_path,
                    f"refinement_comparison_iter{iteration+1}_hypothesis_{hypothesis_idx + 1}.csv"
                )
                comparison_df.to_csv(comparison_csv_path, index=False)
                logger.info(f"\nSaved comparison results to: {comparison_csv_path}")

            # Update for next iteration
            current_compiled_factors = refined_compiled_factors
            current_specs = refined_specs
            loaded_from_previous = True  # Now we're using refined factors

        # Clear price data to free memory
        logger.info("\n🗑️  Freeing price data memory...")
        del price_data_dict
        gc.collect()

        logger.info("\n" + "="*80)
        logger.info("ITERATIVE REFINEMENT COMPLETED")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"Error in pipeline: {e}")
        raise


def display_summary_metrics(factor_metrics: List[Dict], output_path: str = OUTPUT_PATH,
                           hypothesis_idx: int = 0) -> None:
    """Display comprehensive summary metrics.

    Args:
        factor_metrics: List of factor metric dictionaries
        output_path: Path to save output files
        hypothesis_idx: Index of hypothesis (for unique file naming)
    """
    logger.info("\n" + "="*50)
    logger.info("FACTOR PERFORMANCE SUMMARY")
    logger.info("="*50)

    metrics_df = pd.DataFrame(factor_metrics)

    # Top 10 factors by Pearson IC
    logger.info(f"\nTop 10 factors by Pearson IC (with Newey-West t-stats):")
    top_pearson = metrics_df.nlargest(10, 'pearson_ic')[
        ['name', 'pearson_ic', 'nw_tstat', 'sharpe']
    ]
    logger.info(f"\n{top_pearson.to_string(index=False)}")

    # Top 10 factors by Sharpe ratio
    logger.info(f"\nTop 10 factors by Sharpe ratio:")
    top_sharpe = metrics_df.nlargest(10, 'sharpe')[
        ['name', 'sharpe', 'pearson_ic', 'nw_tstat']
    ]
    logger.info(f"\n{top_sharpe.to_string(index=False)}")

    # Overall statistics
    logger.info(f"\nOverall statistics across all {len(metrics_df)} factors:")
    logger.info(f"  Pearson IC:    mean={metrics_df['pearson_ic'].mean():.6f}, std={metrics_df['pearson_ic'].std():.6f}")
    logger.info(f"  NW t-stat:     mean={metrics_df['nw_tstat'].mean():.4f}, std={metrics_df['nw_tstat'].std():.4f}")
    logger.info(f"  Sharpe ratio:  mean={metrics_df['sharpe'].mean():.6f}, std={metrics_df['sharpe'].std():.6f}")

    # Statistical significance summary
    significant_pearson = (metrics_df['nw_tstat'].abs() > 1.96).sum()  # 95% confidence
    highly_significant_pearson = (metrics_df['nw_tstat'].abs() > 2.576).sum()  # 99% confidence
    logger.info(f"\nStatistical Significance (Newey-West HAC t-tests):")
    logger.info(f"  Significant at 95% (|t| > 1.96): {significant_pearson}/{len(metrics_df)} factors")
    logger.info(f"  Significant at 99% (|t| > 2.58): {highly_significant_pearson}/{len(metrics_df)} factors")

    # Save detailed results to CSV (unique per hypothesis)
    csv_output_path = os.path.join(output_path, f"factor_metrics_detailed_hypothesis_{hypothesis_idx + 1}.csv")
    metrics_df.to_csv(csv_output_path, index=False)
    logger.info(f"\nDetailed factor metrics saved to: {csv_output_path}")


def perform_ast_refinement(agent: FactorAgent, specs: List[FactorSpec],
                           factor_metrics: List[Dict], loaded_from_previous: bool,
                           ast_save_path: str) -> None:
    """Perform AST-based iterative refinement.

    Args:
        agent: Initialized FactorAgent
        specs: List of FactorSpec objects
        factor_metrics: List of factor metric dictionaries
        loaded_from_previous: Whether ASTs were loaded from previous iteration
        ast_save_path: Path to save refined ASTs
    """
    logger.info("\n" + "="*50)
    logger.info("AST-BASED ITERATIVE REFINEMENT")
    logger.info("="*50)

    # Perform refinement if we have specs and metrics (regardless of whether loaded from previous)
    if len(specs) > 0 and len(factor_metrics) > 0:
        logger.info(f"Refining worst-performing factors based on computed metrics")
        logger.info(f"  Source: {'Loaded from previous iteration' if loaded_from_previous else 'Newly generated'}")

        metrics_df = pd.DataFrame(factor_metrics)

        # Refine factors with low Sharpe ratio (default threshold: 2.0)
        refined_specs = agent.refine_asts_with_metrics(specs, metrics_df)

        # Save refined ASTs for next iteration
        logger.info(f"Saving refined ASTs to: {ast_save_path}")
        refined_ast_data = []
        for spec in refined_specs:
            refined_ast_data.append({
                'name': spec.name,
                'description': spec.description,
                'reasoning': spec.reasoning,
                'ast': spec.ast
            })
        with open(ast_save_path, 'w') as f:
            json.dump(refined_ast_data, f, indent=2)
        logger.info(f"Saved {len(refined_ast_data)} refined ASTs for next iteration")
        logger.info("✅ Run the script again to test the refined factors!")
    else:
        logger.info("No factors to refine (missing specs or metrics)")


def run_pipeline(agent: FactorAgent, n_tickers: int, data_path: str = "/home/lichenhui/data/1min",
                output_path: str = OUTPUT_PATH) -> None:
    """Run the complete alpha factor pipeline.

    Args:
        agent: Initialized FactorAgent
        n_tickers: Number of top liquid tickers to load
        data_path: Path to market data directory
        output_path: Path to save output files
    """
    # Load real market data - TOP N TICKERS
    logger.info("\n" + "="*50)
    logger.info(f"LOADING MARKET DATA - TOP {n_tickers} TICKERS")
    logger.info("="*50)
    market_data_info = load_real_market_data(data_path=data_path, n_tickers=n_tickers)
    tickers = market_data_info['tickers']
    ticker_data_dict = market_data_info['ticker_data']
    logger.info(f"Prepared {len(tickers)} tickers for one-by-one processing")

    # Load all hypotheses
    all_hypotheses = load_hypotheses()

    # Iterate through all hypotheses
    for current_hypothesis_idx in range(len(all_hypotheses)):
        hypothesis = all_hypotheses[current_hypothesis_idx]['hypothesis_text']
        reasoning = all_hypotheses[current_hypothesis_idx]['reasoning']

        logger.info("\n" + "="*80)
        logger.info(f"PROCESSING HYPOTHESIS {current_hypothesis_idx + 1}/{len(all_hypotheses)}")
        logger.info("="*80)
        logger.info(f"  Paper: {all_hypotheses[current_hypothesis_idx]['paper_title']}")
        logger.info(f"  Hypothesis: {hypothesis[:200]}...")
        logger.info(f"  Reasoning: {reasoning[:200]}...")
        logger.info(f"  Source: {all_hypotheses[current_hypothesis_idx]['source_file']}")

        # Process this hypothesis
        process_single_hypothesis(
            agent, hypothesis, reasoning, current_hypothesis_idx,
            tickers, ticker_data_dict, output_path
        )

    logger.info("\n" + "="*80)
    logger.info("ALPHA AGENT FACTOR PIPELINE FINISHED")
    logger.info("="*80)
