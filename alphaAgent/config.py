#!/usr/bin/env python3
"""
Configuration and constants for Alpha Agent Factor Pipeline
"""

import logging
from datetime import datetime, timedelta

# Calculate date for last three years of data
TWO_YEARS_AGO = (datetime.now() - timedelta(days=1095)).strftime('%Y-%m-%d')

# Configure logging
def setup_logging():
    """Setup logging configuration for the pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('alpha_agent_factor.log')
        ]
    )
    return logging.getLogger(__name__)

# Default configuration values
DEFAULT_N_TICKERS = 2500
DEFAULT_N_CORES = 20
DEFAULT_FORWARD_HORIZON_MINUTES = 30
DEFAULT_N_FACTORS = 25
ANNUALIZE_PERIODS = 252 * (390 // 30)  # Trading periods per year (30-minute bars: 13 periods/day * 252 days) = 3276
FORWARD_HORIZON_MINUTES = 30   # Forward return horizon for factor evaluation

# File paths
DEFAULT_DATA_PATH = "/home/lichenhui/data/1min"
PDF_RESULTS_PATH = "/home/lichenhui/data/alphaAgent/pdf_results"
OUTPUT_PATH = "/home/lichenhui/data/alphaAgent"
