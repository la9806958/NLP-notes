#!/usr/bin/env python3
"""
AlphaAgent Factor Agent - Main CLI Entry Point

This is the main command-line interface for the Alpha Agent Factor Pipeline.
It handles argument parsing and delegates to the pipeline module for execution.
"""

import os
import argparse
import multiprocessing as mp

from config import setup_logging, DEFAULT_N_TICKERS
from data_loader import load_credentials
from factor_agent import FactorAgent
from pipeline import run_pipeline

# Setup logging
logger = setup_logging()


def main():
    """Main entry point for the Alpha Agent Factor Pipeline."""
    # Configure multiprocessing for cross-platform compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Start method already set

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Alpha Agent Factor Pipeline with Model Selection')
    parser.add_argument('--model', type=str, default='lstm_simplified',
                       choices=['lstm_simplified', 'timemixer_simplified', 'dlinear_simplified',
                               'mlp_simplified', 'itransformer_simplified', 'translob_simplified'],
                       help='Specify which simplified model to use for predictions')
    parser.add_argument('--iterations', type=int, default=3,
                       help='Number of iterative improvement iterations')
    parser.add_argument('--n_minutes', type=int, default=300,
                       help='Number of minutes of historical data to use')
    parser.add_argument('--n_tickers', type=int, default=DEFAULT_N_TICKERS,
                       help=f'Number of top liquid tickers to load (default: {DEFAULT_N_TICKERS})')
    parser.add_argument('--data_path', type=str, default="/home/lichenhui/data/1min",
                       help='Path to market data directory')

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("ALPHA AGENT FACTOR PIPELINE STARTING")
    logger.info("="*80)
    logger.info(f"Selected Model: {args.model}")
    logger.info(f"Iterations: {args.iterations}")
    logger.info(f"Historical Data: {args.n_minutes} minutes")
    logger.info(f"Top Tickers to Load: {args.n_tickers}")
    logger.info(f"Data Path: {args.data_path}")

    # Load API credentials
    credentials = load_credentials("credentials.json")
    api_key = credentials.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")

    # Initialize FactorAgent with OpenAI API
    if not api_key:
        logger.error("No OpenAI API key found. Set OPENAI_API_KEY environment variable or add to credentials.json")
        raise ValueError("OpenAI API key required")

    logger.info("Using FactorAgent with OpenAI API")
    agent = FactorAgent(api_key=api_key, data_path=args.data_path)

    # Run the pipeline
    run_pipeline(agent, n_tickers=args.n_tickers, data_path=args.data_path)


if __name__ == "__main__":
    main()
