#!/usr/bin/env python3
"""
AlphaAgent Factor Agent Implementation (AST version)
- Defines a compact Abstract Syntax Tree (AST) for factor expressions
- Asks the LLM for a *precise* JSON AST + human-readable reasoning
- Compiles the AST to an executable function using your Operators library

Key ideas
---------
1) Deterministic interface to the LLM: we request a strict JSON object containing
   factor expressions as AST nodes, plus a free-form `reasoning` string.
2) Sandboxed compilation: only a small, audited subset of operations map to
   alpha_agent_operators.Operators methods (safe DIV, TS_* windows, etc.).
3) Traceability: we persist the raw AST, the compiled Python, and the model's
   reasoning next to each factor for audit and reproducibility.

You can integrate this file drop-in to replace the previous FactorAgent.
"""

from __future__ import annotations
import os
import json
import glob
import logging
import jsonschema
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import multiprocessing as mp
from functools import partial
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, log_loss
from scipy.stats import pearsonr, spearmanr, t as t_dist
from scipy.special import softmax
import torch
import torch.nn as nn
import importlib.util
import sys
from pathlib import Path
from statsmodels.stats.sandwich_covariance import cov_hac
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

# External dependency you already have
from alpha_agent_operators import Operators

# Import evaluation functions from separate module
from factor_evaluation import (
    compute_pearson_ic_timeseries_streaming,
    compute_crosssec_sharpe_streaming,
    save_factor_pnl_visualization,
    compute_factor_correlations,
    compute_factor_correlations_by_ticker,
    _nw_tstat_from_series
)

# Import validation functions from separate module
from validator import (
    normalize_ast_for_comparison,
    hash_ast,
    dedupe_factor_names
)

# Import AST compiler from separate module
from ast_compiler import (
    AST,
    ALLOWED_FUNCS,
    ALLOWED_SYMBOLS,
    MAX_WINDOW,
    MAX_POW,
    FactorSpec,
    ASTValidationError,
    ASTCompiler
)

# Import prompt-building functions from separate module
from prompts import (
    build_main_prompt,
    build_single_factor_prompt,
    build_error_recovery_prompt,
    generate_factor_feedback,
    build_hypothesis_improvement_prompt,
    extract_operator_signatures
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('alpha_agent_factor.log')
    ]
)
logger = logging.getLogger(__name__)

# Calculate date for last two years of data
TWO_YEARS_AGO = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

# ------------------------------
# UTILITY FUNCTIONS
# ------------------------------
# Note: Evaluation functions have been moved to factor_evaluation.py
# Note: Validation functions have been moved to validator.py

# ------------------------------
# PARALLEL PROCESSING HELPERS
# ------------------------------

def _process_csv_for_volume(csv_file: str) -> Tuple[Optional[str], Optional[float]]:
    """Helper function to process a single CSV file for volume calculation.
    
    Returns:
        Tuple of (ticker, avg_volume) or (None, None) if processing fails
    """
    try:
        # Extract ticker from filename
        ticker = os.path.splitext(os.path.basename(csv_file))[0]
        
        # Load and calculate average volume
        df = pd.read_csv(csv_file)
        # Standardize column names to lowercase to handle both 'Volume' and 'volume'
        df.columns = df.columns.str.lower()
        
        if 'volume' not in df.columns:
            return ticker, None  # Will log warning in main process
            
        # Handle timestamp/datetime column and filter to last two years
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'])
        elif 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        else:
            # Skip files without datetime information
            return ticker, None
            
        # Filter to last two years
        df = df[df['datetime'] >= TWO_YEARS_AGO]
        
        if len(df) == 0:
            return ticker, None  # No data from last two years
            
        avg_volume = df['volume'].mean()
        return ticker, avg_volume
    except Exception:
        return None, None

def _process_csv_for_data(csv_file_info: Tuple[str, int]) -> Tuple[Optional[str], Optional[pd.DataFrame]]:
    """Helper function to process a single CSV file for data loading.
    
    Args:
        csv_file_info: Tuple of (csv_file_path, max_samples_to_scan)
    
    Returns:
        Tuple of (ticker, dataframe) or (None, None) if processing fails
    """
    csv_file, max_samples = csv_file_info
    try:
        ticker = os.path.splitext(os.path.basename(csv_file))[0]
        df = pd.read_csv(csv_file)
        
        # Standardize column names to lowercase
        df.columns = df.columns.str.lower()
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        
        if not all(col in df.columns for col in required_cols):
            return ticker, None  # Will log warning in main process
        
        # Handle timestamp/datetime column
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'])
        elif 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        else:
            # Skip files without datetime information for last two years filtering
            return ticker, None
        
        # Filter to last two years BEFORE setting index
        df = df[df['datetime'] >= TWO_YEARS_AGO]
        
        if len(df) == 0:
            return ticker, None  # No data from last two years
        
        df.set_index('datetime', inplace=True)
        df = df[required_cols].sort_index()
        
        # Filter out invalid data
        df = df[(df > 0).all(axis=1)]  # Remove rows with non-positive values
        df = df.dropna()
        
        if len(df) < 1000:  # Need sufficient data
            return ticker, None

        # Return filepath and avg_volume only (not the dataframe!)
        # Loading dataframes in parallel causes BrokenPipeError
        avg_volume = df['volume'].mean()
        return ticker, (csv_file, avg_volume)
        
    except Exception:
        return None, None

# ------------------------------
# AST DEFINITIONS
# ------------------------------
# Note: AST definitions have been moved to ast_compiler.py

# ------------------------------
# FACTOR AGENT
# ------------------------------

class FactorAgent:
    """Parses market hypotheses into executable factor expressions using an AST.

    The LLM is asked to return *strict JSON* following a schema; we then
    validate, compile, and execute those factors.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini", data_path: str = "/home/lichenhui/data/1min"):
        logger.info(f"Initializing FactorAgent with model={model}, data_path={data_path}")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.error("No OpenAI API key provided")
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        self.model = model
        self.data_path = data_path
        # Lazy import to keep this module importable without openai installed
        import openai  # type: ignore
        self._openai = openai.OpenAI(api_key=self.api_key)
        self.ops = Operators()
        self.compiler = ASTCompiler(self.ops)
        
        # Precompute and cache top liquid tickers
        logger.info("Loading top liquid tickers...")
        self.top_liquid_tickers = self._load_top_liquid_tickers()
        logger.info(f"Initialized FactorAgent with {len(self.top_liquid_tickers)} liquid tickers")

    # ---------- LLM IO ----------

    FACTOR_JSON_SCHEMA = {
        "type": "object",
        "properties": {
            "factors": {
                "type": "array",
                "minItems": 1,
                "maxItems": 25,
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "reasoning": {"type": "string"},
                        "expr": {"type": "object"}
                    },
                    "required": ["name", "description", "reasoning", "expr"]
                }
            }
        },
        "required": ["factors"]
    }

    SINGLE_FACTOR_JSON_SCHEMA = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "reasoning": {"type": "string"},
            "expr": {"type": "object"}
        },
        "required": ["name", "description", "reasoning", "expr"]
    }

    def _load_top_liquid_tickers(self, n_tickers: int = 2000) -> List[str]:
        """Load top 2000 most liquid tickers by volume from 1min folder.
        
        Returns list of ticker symbols sorted by average volume descending.
        """
        logger.info(f"Loading top {n_tickers} liquid tickers from {self.data_path}")
        if not os.path.exists(self.data_path):
            logger.warning(f"Data path {self.data_path} does not exist. Using mock tickers.")
            # Return mock tickers for development
            return [f"TICK_{i:04d}" for i in range(n_tickers)]
        
        # Look for CSV files in the 1min folder
        csv_files = glob.glob(os.path.join(self.data_path, "*.csv"))
        
        if not csv_files:
            logger.warning(f"No CSV files found in {self.data_path}. Using mock tickers.")
            return [f"TICK_{i:04d}" for i in range(n_tickers)]
        
        # Use limited CPU cores for parallel processing to avoid overloading system
        n_cores = min(20, mp.cpu_count())
        logger.info(f"Processing {len(csv_files)} CSV files using {n_cores} CPU cores")
        
        ticker_volumes = {}
        
        # Process CSV files in parallel
        try:
            with mp.Pool(processes=n_cores) as pool:
                results = pool.map(_process_csv_for_volume, csv_files)
        except (OSError, RuntimeError) as e:
            # Fallback to sequential processing if multiprocessing fails
            logger.warning(f"Multiprocessing failed ({e}), falling back to sequential processing")
            results = [_process_csv_for_volume(csv_file) for csv_file in csv_files]
        
        # Collect results and log warnings
        for ticker, avg_volume in results:
            if ticker is not None and avg_volume is not None:
                ticker_volumes[ticker] = avg_volume
            elif ticker is not None and avg_volume is None:
                logger.warning(f"No volume column in {ticker}.csv")
            # If both are None, there was an error (already handled silently)
        
        # Sort by volume descending and take top n_tickers
        sorted_tickers = sorted(ticker_volumes.items(), key=lambda x: x[1], reverse=True)
        top_tickers = [ticker for ticker, _ in sorted_tickers[:n_tickers]]
        
        logger.info(f"Loaded {len(top_tickers)} liquid tickers from {len(csv_files)} files")
        return top_tickers
    
    def _load_ticker_data(self, ticker: str, lookback_minutes: int = 300) -> Optional[pd.DataFrame]:
        """Load OHLCV data for a specific ticker with specified lookback.
        
        Returns DataFrame with columns: open, high, low, close, volume
        """
        csv_path = os.path.join(self.data_path, f"{ticker}.csv")
        
        if not os.path.exists(csv_path):
            return None
            
        try:
            df = pd.read_csv(csv_path)
            
            # Standardize column names to lowercase
            df.columns = df.columns.str.lower()
            
            # Ensure we have required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                print(f"Warning: Missing required columns in {csv_path}")
                return None
                
            # Convert timestamp if present
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                # Filter to last two years
                df = df[df['timestamp'] >= TWO_YEARS_AGO]
                df.set_index('timestamp', inplace=True)
            elif 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                # Filter to last two years
                df = df[df['datetime'] >= TWO_YEARS_AGO]
                df.set_index('datetime', inplace=True)
            else:
                print(f"Warning: No datetime column found in {csv_path}")
                return None
                
            if len(df) == 0:
                print(f"Warning: No data from last two years in {csv_path}")
                return None

            # Return all available data (no longer limiting to lookback_minutes)
            return df[required_cols]
            
        except Exception as e:
            print(f"Error loading data for {ticker}: {e}")
            return None

    def _build_prompt(self, hypothesis: str) -> str:
        return build_main_prompt(hypothesis)

    def _call_llm(self, prompt: str, schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Use JSON mode for reliability where available
        # schema parameter allows specifying SINGLE_FACTOR_JSON_SCHEMA or FACTOR_JSON_SCHEMA
        if schema is None:
            schema = self.FACTOR_JSON_SCHEMA

        logger.info("🔍 LLM DEBUG: Starting LLM call...")
        logger.info(f"🔍 LLM DEBUG: Full prompt being sent:")
        logger.info("="*100)
        logger.info(prompt)
        logger.info("="*100)

        try:
            resp = self._openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise machine that outputs strict JSON only. Your response must be valid JSON with no markdown formatting, no code blocks, and no additional text."},
                    {"role": "user", "content": prompt + "\n\nIMPORTANT: Respond with ONLY valid JSON, no markdown formatting or code blocks."}
                ],
                temperature=0,
                response_format={"type": "json_object"}  # Force JSON mode
            )
            content = resp.choices[0].message.content.strip()

            logger.info(f"🔍 LLM DEBUG: Raw LLM response received ({len(content)} chars):")
            logger.info("="*100)
            logger.info(content)
            logger.info("="*100)

            # Clean up common JSON formatting issues
            if content.startswith('```json'):
                content = content.replace('```json', '').replace('```', '').strip()
                logger.info("🔍 LLM DEBUG: Removed ```json formatting")
            elif content.startswith('```'):
                content = content.replace('```', '').strip()
                logger.info("🔍 LLM DEBUG: Removed ``` formatting")

            logger.info("🔍 LLM DEBUG: Attempting to parse JSON...")
            parsed_json = json.loads(content)
            logger.info("🔍 LLM DEBUG: JSON parsing successful!")

            # Validate against JSON schema to ensure contract compliance
            try:
                jsonschema.validate(instance=parsed_json, schema=schema)
                logger.info("🔍 LLM DEBUG: JSON schema validation passed!")
            except jsonschema.ValidationError as schema_e:
                logger.error("❌ LLM DEBUG: JSON schema validation failed!")
                logger.error(f"❌ Schema Error: {schema_e.message}")
                logger.error(f"❌ Failed at path: {'.'.join(str(p) for p in schema_e.path) if schema_e.path else 'root'}")
                raise ASTValidationError(f"JSON schema validation failed: {schema_e.message}")
            except jsonschema.SchemaError as schema_e:
                logger.error("❌ LLM DEBUG: Invalid schema definition!")
                logger.error(f"❌ Schema Definition Error: {schema_e.message}")
                raise ASTValidationError(f"Invalid schema definition: {schema_e.message}")

            return parsed_json
            
        except json.JSONDecodeError as json_e:
            logger.error("❌ LLM DEBUG: JSON parsing failed!")
            logger.error(f"❌ JSON Error: {json_e}")
            logger.error(f"❌ Error position: {json_e.pos if hasattr(json_e, 'pos') else 'unknown'}")
            
            # Show context around the error
            if hasattr(json_e, 'pos') and content:
                error_pos = json_e.pos
                start = max(0, error_pos - 200)
                end = min(len(content), error_pos + 200)
                logger.error(f"❌ Error context (±200 chars around position {error_pos}):")
                logger.error("="*100)
                logger.error(content[start:end])
                logger.error("="*100)
                
            # Show the problematic content in smaller chunks
            logger.error(f"❌ Full content to debug (length: {len(content)}):")
            chunk_size = 1000
            for i in range(0, len(content), chunk_size):
                chunk = content[i:i+chunk_size]
                logger.error(f"❌ Chunk {i//chunk_size + 1}: {chunk}")
                
            raise RuntimeError(f"LLM JSON parse failed: {json_e}")
            
        except Exception as e:
            logger.error(f"❌ LLM DEBUG: Unexpected error: {e}")
            logger.error(f"❌ Error type: {type(e).__name__}")
            raise RuntimeError(f"LLM call failed: {e}")

    def _extract_operator_signatures(self) -> str:
        """Extract operator signatures from the operators file."""
        return extract_operator_signatures()
    
    def _build_single_factor_prompt(self, hypothesis: str, factor_num: int, existing_factors: List[FactorSpec]) -> str:
        """Build prompt for generating a single factor."""
        return build_single_factor_prompt(hypothesis, factor_num, existing_factors)

    def _build_error_recovery_prompt(self, hypothesis: str, factor_num: int, error_msg: str, failed_payload: dict = None) -> str:
        """Build error recovery prompt with specific feedback."""
        return build_error_recovery_prompt(hypothesis, factor_num, error_msg, failed_payload)

    def _extract_single_factor(self, payload: dict) -> dict:
        """Extract and validate a single factor from LLM response."""
        
        # Check if payload has the required structure
        if not isinstance(payload, dict):
            raise ValueError("LLM response is not a JSON object")
            
        required_fields = ["name", "description", "reasoning", "expr"]
        for field in required_fields:
            if field not in payload:
                raise ValueError(f"Missing required field: {field}")
                
        # Validate that expr is a proper AST object
        expr = payload["expr"]
        if not isinstance(expr, dict) or "type" not in expr:
            raise ValueError("Invalid AST expression structure")
            
        return payload

    def _generate_fallback_factor(self, factor_num: int) -> FactorSpec:
        """Generate a simple fallback factor when LLM fails."""
        
        # Simple fallback factors
        fallback_options = [
            {
                "name": f"close_momentum_{factor_num}",
                "description": "Simple close price momentum",
                "reasoning": "Price momentum as basic factor",
                "expr": {"type": "call", "fn": "diff", "args": [
                    {"type": "symbol", "name": "close"},
                    {"type": "const", "value": 1}
                ]}
            },
            {
                "name": f"volume_mean_{factor_num}",
                "description": "Rolling volume average", 
                "reasoning": "Volume trend as basic factor",
                "expr": {"type": "call", "fn": "ts_mean", "args": [
                    {"type": "symbol", "name": "volume"},
                    {"type": "const", "value": 10}
                ]}
            },
            {
                "name": f"high_low_range_{factor_num}",
                "description": "High-low price range",
                "reasoning": "Volatility measure as basic factor", 
                "expr": {"type": "call", "fn": "sub", "args": [
                    {"type": "symbol", "name": "high"},
                    {"type": "symbol", "name": "low"}
                ]}
            }
        ]
        
        # Choose fallback based on factor number
        fallback = fallback_options[factor_num % len(fallback_options)]
        fallback["name"] = f"{fallback['name']}_{factor_num}"  # Ensure uniqueness
        
        return FactorSpec(
            name=fallback["name"],
            description=fallback["description"], 
            reasoning=fallback["reasoning"],
            ast=fallback["expr"]
        )

    # ---------- Public API ----------

    def parse_hypothesis(self, hypothesis: str) -> List[FactorSpec]:
        """Parse hypothesis using sequential LLM calls with error recovery and AST deduplication."""
        logger.info("Starting sequential factor generation with error recovery and AST deduplication")

        out: List[FactorSpec] = []
        seen_ast_hashes: set = set()  # Track AST hashes to prevent duplicates

        for i in range(25):  # Generate exactly 25 factors
            logger.info(f"Generating factor {i+1}/25")

            # Build prompt for single factor
            factor_prompt = self._build_single_factor_prompt(hypothesis, i+1, out)

            max_attempts = 5  # Increased from 3 to allow more retries for duplicates
            for attempt in range(max_attempts):
                try:
                    # Get single factor from LLM using the single factor schema
                    payload = self._call_llm(factor_prompt, schema=self.SINGLE_FACTOR_JSON_SCHEMA)

                    # Log the raw response for debugging
                    logger.debug(f"Factor {i+1} attempt {attempt+1} raw response: {payload}")

                    # Parse the factor - payload should be a single factor dict with keys: name, description, reasoning, expr
                    factor_data = self._extract_single_factor(payload)

                    # Create FactorSpec
                    spec = FactorSpec(
                        name=factor_data["name"],
                        description=factor_data["description"],
                        reasoning=factor_data["reasoning"],
                        ast=factor_data["expr"],
                    )

                    # Validate the AST
                    self.compiler._validate(spec.ast)

                    # Check for duplicate AST
                    ast_hash = hash_ast(spec.ast)
                    if ast_hash in seen_ast_hashes:
                        # Find which factor has this AST
                        duplicate_factor = next((f.name for f in out if hash_ast(f.ast) == ast_hash), "unknown")
                        raise ValueError(f"Duplicate AST detected - identical to factor '{duplicate_factor}'")

                    # Success - add to list and track hash
                    seen_ast_hashes.add(ast_hash)
                    out.append(spec)
                    logger.info(f"✅ Factor {i+1} generated successfully: {spec.name} (unique AST hash: {ast_hash[:8]}...)")
                    break

                except ASTValidationError as e:
                    logger.warning(f"❌ Factor {i+1} attempt {attempt+1} failed validation: {e}")
                    if attempt < max_attempts - 1:
                        # Try error recovery with feedback
                        try:
                            factor_prompt = self._build_error_recovery_prompt(hypothesis, i+1, str(e), payload if 'payload' in locals() else None)
                            logger.info(f"🔄 Attempting error recovery for factor {i+1}")
                        except:
                            logger.error(f"Failed to build error recovery prompt for factor {i+1}")
                    else:
                        # Final attempt failed - skip this factor instead of using fallback
                        logger.error(f"⚠️  All attempts failed for factor {i+1} due to validation errors, SKIPPING (no fallback)")
                        # Don't add anything to out, just continue to next factor

                except ValueError as e:
                    if "Duplicate AST" in str(e):
                        logger.warning(f"❌ Factor {i+1} attempt {attempt+1} failed: {e}")
                        if attempt < max_attempts - 1:
                            # Modify prompt to explicitly reject this duplicate
                            factor_prompt = self._build_single_factor_prompt(hypothesis, i+1, out)
                            factor_prompt += f"\n\nIMPORTANT: Your previous attempt was rejected because it generated a duplicate factor expression. You MUST create a factor with a COMPLETELY DIFFERENT structure and operators. Use different combinations of operators, different window sizes, or different data fields."
                            logger.info(f"🔄 Retrying factor {i+1} with anti-duplicate prompt")
                        else:
                            # Final attempt failed - generate fallback factor
                            logger.error(f"All attempts failed for factor {i+1}, generating fallback")
                            fallback_spec = self._generate_fallback_factor(i+1)
                            # Ensure fallback is also unique
                            fallback_hash = hash_ast(fallback_spec.ast)
                            if fallback_hash not in seen_ast_hashes:
                                seen_ast_hashes.add(fallback_hash)
                                out.append(fallback_spec)
                            else:
                                # Modify fallback to make it unique
                                fallback_spec.ast["args"][1]["value"] = 10 + i  # Change window size
                                fallback_hash = hash_ast(fallback_spec.ast)
                                seen_ast_hashes.add(fallback_hash)
                                out.append(fallback_spec)
                    else:
                        raise  # Re-raise other ValueErrors

                except Exception as e:
                    logger.warning(f"❌ Factor {i+1} attempt {attempt+1} failed: {e}")

                    if attempt < max_attempts - 1:
                        # Try error recovery with feedback
                        try:
                            factor_prompt = self._build_error_recovery_prompt(hypothesis, i+1, str(e), payload if 'payload' in locals() else None)
                            logger.info(f"🔄 Attempting error recovery for factor {i+1}")
                        except:
                            logger.error(f"Failed to build error recovery prompt for factor {i+1}")
                    else:
                        # Final attempt failed - generate fallback factor
                        logger.error(f"All attempts failed for factor {i+1}, generating fallback")
                        fallback_spec = self._generate_fallback_factor(i+1)
                        # Ensure fallback is also unique
                        fallback_hash = hash_ast(fallback_spec.ast)
                        if fallback_hash not in seen_ast_hashes:
                            seen_ast_hashes.add(fallback_hash)
                            out.append(fallback_spec)
                        else:
                            # Modify fallback to make it unique by changing window size
                            if fallback_spec.ast.get("type") == "call" and fallback_spec.ast.get("args"):
                                for arg in fallback_spec.ast["args"]:
                                    if isinstance(arg, dict) and arg.get("type") == "const":
                                        arg["value"] = arg["value"] + i  # Make unique
                                        break
                            fallback_hash = hash_ast(fallback_spec.ast)
                            seen_ast_hashes.add(fallback_hash)
                            out.append(fallback_spec)

        logger.info(f"Sequential parsing completed: {len(out)}/25 factors generated ({len(seen_ast_hashes)} unique ASTs)")
        return out

    def compile_factors(self, specs: List[FactorSpec]) -> List[Dict[str, Any]]:
        compiled: List[Dict[str, Any]] = []
        for spec in specs:
            fn = self.compiler.compile(spec.ast)
            compiled.append({
                "name": spec.name,
                "description": spec.description,
                "reasoning": spec.reasoning,
                "ast": spec.ast,
                "callable": fn,
            })
        return compiled

    def generate_factor_matrix(self, hypothesis: str, n_minutes: int = None) -> np.ndarray:
        """Generate factor matrix for time-series prediction using all available data.

        Args:
            n_minutes: Deprecated parameter (kept for backward compatibility but ignored)

        Returns:
            np.ndarray: Factor values matrix with shape (n_timesteps, 25 factors)
        """
        logger.info(f"Generating factor matrix with all available data from last two years")
        specs = self.parse_hypothesis(hypothesis)
        logger.info(f"Parsed {len(specs)} factor specifications from hypothesis")
        compiled = self.compile_factors(specs)

        # Ensure we have exactly 25 factors
        if len(compiled) != 25:
            logger.error(f"Expected 25 factors, got {len(compiled)}")
            raise ValueError(f"Expected 25 factors, got {len(compiled)}")

        # Use first available ticker from top liquid list
        ticker_data = None
        for ticker in self.top_liquid_tickers[:10]:  # Try first 10 tickers
            ticker_data = self._load_ticker_data(ticker)
            if ticker_data is not None:
                logger.info(f"Loaded {len(ticker_data)} timesteps from {ticker}")
                break

        if ticker_data is None:
            # Hard fallback: raise error if no data available
            logger.error("No ticker data available")
            raise ValueError("No ticker data available. Cannot generate factor matrix.")

        # Use all available data instead of limiting to n_minutes
        n_timesteps = len(ticker_data)
        factor_matrix = np.zeros((n_timesteps, 25))

        for i, item in enumerate(compiled):
            try:
                factor_series = item["callable"](ticker_data)
                factor_series = self._sanitize_series(factor_series)
                factor_series = self._zscore(factor_series)

                # Use all available factor values
                if len(factor_series) == n_timesteps:
                    factor_matrix[:, i] = factor_series.values
                elif len(factor_series) < n_timesteps:
                    # Pad with zeros if not enough data
                    factor_matrix[:len(factor_series), i] = factor_series.values
                else:
                    # Truncate if somehow we have more data
                    factor_matrix[:, i] = factor_series.iloc[:n_timesteps].values

            except Exception as e:
                logger.error(f"Error computing factor {item['name']}: {e}")
                # Fill with zeros on error
                factor_matrix[:, i] = 0

        logger.info(f"Generated factor matrix with shape {factor_matrix.shape}")
        return factor_matrix

    def generate_alpha_from_hypothesis(self, hypothesis: str, data: pd.DataFrame,
                                       combine: str = "equal_weight") -> pd.Series:
        specs = self.parse_hypothesis(hypothesis)
        compiled = self.compile_factors(specs)

        signals: List[pd.Series] = []
        meta: List[Tuple[str, str]] = []  # (name, reasoning)
        for item in compiled:
            s = item["callable"](data)
            s = self._sanitize_series(s)
            s = self._zscore(s)
            signals.append(s)
            meta.append((item["name"], item["reasoning"]))

        if not signals:
            raise ValueError("No factor signals produced")

        stacked = pd.concat(signals, axis=1)
        stacked.columns = [m[0] for m in meta]

        if combine == "equal_weight":
            alpha = stacked.mean(axis=1)
        elif combine == "median":
            alpha = stacked.median(axis=1)
        else:
            raise ValueError(f"Unknown combine='{combine}'")

        alpha = self._zscore(alpha)
        # Optionally return meta alongside alpha for audit
        self.last_factor_table_ = stacked
        self.last_reasoning_ = {name: rsn for name, rsn in meta}
        return alpha

    # ---------- utils ----------

    @staticmethod
    def _sanitize_series(s: Union[pd.Series, pd.DataFrame, np.ndarray]) -> pd.Series:
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0] if s.shape[1] == 1 else s.mean(axis=1)
        elif isinstance(s, np.ndarray):
            s = pd.Series(np.asarray(s).squeeze())
        s = s.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return s

    @staticmethod
    def _zscore(s: pd.Series) -> pd.Series:
        mu, sd = s.mean(), s.std()
        return (s - mu) / sd if sd > 0 else s - mu

    def calculate_forward_returns(self, data: pd.DataFrame, horizon_minutes: int = 30) -> np.ndarray:
        """Calculate forward looking returns for a given horizon.

        Args:
            data: DataFrame with OHLCV data indexed by timestamp
            horizon_minutes: Number of minutes to look forward (default 30)

        Returns:
            np.ndarray: Forward returns for each timestamp
        """
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")

        close_prices = data['close'].values

        # Calculate forward returns: (price[t+horizon] - price[t]) / price[t]
        forward_returns = np.zeros(len(close_prices))

        for i in range(len(close_prices) - horizon_minutes):
            if close_prices[i] != 0:
                forward_returns[i] = (close_prices[i + horizon_minutes] - close_prices[i]) / close_prices[i]

        # Last horizon_minutes entries will be NaN since we can't compute forward returns
        forward_returns[-horizon_minutes:] = np.nan

        return forward_returns

    def calculate_alpha_correlation(self, alpha: np.ndarray, forward_returns: np.ndarray) -> float:
        """Calculate Pearson correlation between alpha and forward returns.

        Args:
            alpha: Alpha values (factor signals)
            forward_returns: Forward looking returns

        Returns:
            float: Pearson correlation coefficient
        """
        # Align lengths and remove NaN values
        min_len = min(len(alpha), len(forward_returns))
        alpha_aligned = alpha[:min_len]
        returns_aligned = forward_returns[:min_len]

        # Remove NaN values
        valid_mask = ~(np.isnan(alpha_aligned) | np.isnan(returns_aligned))
        alpha_clean = alpha_aligned[valid_mask]
        returns_clean = returns_aligned[valid_mask]

        if len(alpha_clean) < 2 or np.std(alpha_clean) == 0 or np.std(returns_clean) == 0:
            return 0.0

        correlation, _ = pearsonr(alpha_clean, returns_clean)
        return correlation

    def calculate_alpha_sharpe(self, alpha: np.ndarray, forward_returns: np.ndarray) -> float:
        """Calculate Sharpe ratio of PNL from alpha using dot product.

        The PNL at each timestep is the dot product of alpha across the universe
        with the realized returns.

        Args:
            alpha: Alpha values (factor signals) - shape (n_timesteps,) or (n_timesteps, n_assets)
            forward_returns: Forward looking returns - shape (n_timesteps,) or (n_timesteps, n_assets)

        Returns:
            float: Sharpe ratio (mean PNL / std PNL)
        """
        # Align lengths
        min_len = min(len(alpha), len(forward_returns))
        alpha_aligned = alpha[:min_len]
        returns_aligned = forward_returns[:min_len]

        # Remove NaN values
        valid_mask = ~(np.isnan(alpha_aligned) | np.isnan(returns_aligned))
        alpha_clean = alpha_aligned[valid_mask]
        returns_clean = returns_aligned[valid_mask]

        if len(alpha_clean) < 2:
            return 0.0

        # Calculate PNL as element-wise product (dot product across universe at each timestep)
        # For single asset case, this is just alpha * returns
        pnl = alpha_clean * returns_clean

        # Calculate Sharpe ratio
        mean_pnl = np.mean(pnl)
        std_pnl = np.std(pnl)

        if std_pnl == 0 or np.isnan(std_pnl):
            return 0.0

        sharpe = mean_pnl / std_pnl

        # Annualize assuming 1-minute data and 252 trading days
        # Sharpe = (mean / std) * sqrt(252 * 390 minutes per day)
        sharpe_annualized = sharpe * np.sqrt(252 * 390)

        return sharpe_annualized

    def iterative_factor_improvement(self, initial_hypothesis: str, n_iterations: int = 3,
                                   n_minutes: int = None, horizon_minutes: int = 30) -> Dict[str, Any]:
        """Iteratively improve factors using performance feedback with real forward returns.

        This method evaluates each factor individually on:
        1. Correlation with 30-min forward returns
        2. Sharpe ratio using dot product PNL

        These metrics are fed into the next iteration for improvement.

        Args:
            initial_hypothesis: Starting market hypothesis
            n_iterations: Number of improvement iterations
            n_minutes: Deprecated parameter (kept for backward compatibility but ignored)
            horizon_minutes: Forward return horizon (default 30 minutes)

        Returns:
            Dictionary with improvement history and final factors
        """
        improvement_history = []
        current_hypothesis = initial_hypothesis

        # Load real market data for evaluation (all available data from last 2 years)
        ticker_data = None
        for ticker in self.top_liquid_tickers[:10]:  # Try first 10 tickers
            ticker_data = self._load_ticker_data(ticker)
            if ticker_data is not None:
                logger.info(f"Using ticker {ticker} for factor evaluation with {len(ticker_data)} timesteps")
                break

        if ticker_data is None:
            # Hard fallback: raise error if no data available
            logger.error("No ticker data available for iterative improvement")
            raise ValueError("No ticker data available. Cannot run iterative factor improvement.")

        # Calculate forward returns for the entire dataset
        forward_returns = self.calculate_forward_returns(ticker_data, horizon_minutes)

        for iteration in range(n_iterations):
            print(f"\n=== ITERATION {iteration + 1} ===")
            print(f"Current hypothesis: {current_hypothesis}")

            # Parse and compile factors
            specs = self.parse_hypothesis(current_hypothesis)
            compiled = self.compile_factors(specs)
            factor_names = [spec.name for spec in specs]

            # Evaluate each factor individually
            factor_metrics = []

            for i, item in enumerate(compiled):
                try:
                    # Generate factor signal
                    factor_series = item["callable"](ticker_data)
                    factor_series = self._sanitize_series(factor_series)
                    factor_series = self._zscore(factor_series)

                    # Convert to numpy array
                    factor_values = factor_series.values if hasattr(factor_series, 'values') else np.array(factor_series)

                    # Calculate correlation with forward returns
                    correlation = self.calculate_alpha_correlation(factor_values, forward_returns)

                    # Calculate Sharpe ratio
                    sharpe = self.calculate_alpha_sharpe(factor_values, forward_returns)

                    factor_metrics.append({
                        'name': item['name'],
                        'correlation': correlation,
                        'sharpe': sharpe,
                        'reasoning': item['reasoning']
                    })

                    print(f"  Factor {i+1}/{len(compiled)}: {item['name']}")
                    print(f"    Correlation: {correlation:.4f}")
                    print(f"    Sharpe: {sharpe:.4f}")

                except Exception as e:
                    logger.error(f"Error evaluating factor {item['name']}: {e}")
                    factor_metrics.append({
                        'name': item['name'],
                        'correlation': 0.0,
                        'sharpe': 0.0,
                        'reasoning': item['reasoning'],
                        'error': str(e)
                    })

            # Calculate aggregate metrics
            avg_correlation = np.mean([m['correlation'] for m in factor_metrics])
            avg_sharpe = np.mean([m['sharpe'] for m in factor_metrics])

            aggregate_metrics = {
                'avg_correlation': avg_correlation,
                'avg_sharpe': avg_sharpe,
                'factor_metrics': factor_metrics
            }

            print(f"\nAggregate metrics:")
            print(f"  Average Correlation: {avg_correlation:.4f}")
            print(f"  Average Sharpe: {avg_sharpe:.4f}")

            # Generate feedback for LLM
            feedback = self._generate_factor_feedback(factor_metrics, avg_correlation, avg_sharpe)

            improvement_history.append({
                'iteration': iteration + 1,
                'hypothesis': current_hypothesis,
                'metrics': aggregate_metrics,
                'feedback': feedback,
                'factor_names': factor_names
            })

            # Update hypothesis for next iteration (if not last iteration)
            if iteration < n_iterations - 1:
                current_hypothesis = self._improve_hypothesis_with_feedback(
                    current_hypothesis, feedback, aggregate_metrics
                )

        return {
            'history': improvement_history,
            'final_hypothesis': current_hypothesis,
            'final_metrics': improvement_history[-1]['metrics']
        }

    def _generate_factor_feedback(self, factor_metrics: List[Dict], avg_correlation: float,
                                  avg_sharpe: float) -> str:
        """Generate detailed feedback on factor performance."""
        return generate_factor_feedback(factor_metrics, avg_correlation, avg_sharpe)
    
    def refine_asts_with_metrics(self, current_specs: List[FactorSpec],
                                  metrics_df: pd.DataFrame,
                                  n_to_refine: int = 10) -> List[FactorSpec]:
        """Refine the worst-performing ASTs based on metrics.

        Args:
            current_specs: Current list of FactorSpecs
            metrics_df: DataFrame with factor metrics (name, pearson_ic, sharpe, etc.)
            n_to_refine: Number of worst-performing factors to refine

        Returns:
            Updated list of FactorSpecs with refined factors
        """
        logger.info(f"Refining worst {n_to_refine} factors based on performance metrics")

        # Sort by absolute Pearson IC (lowest first)
        worst_factors = metrics_df.nsmallest(n_to_refine, 'pearson_ic')
        worst_factor_names = set(worst_factors['name'].tolist())

        logger.info(f"Factors to refine: {worst_factor_names}")

        # Build feedback for worst factors
        feedback_parts = []
        for _, row in worst_factors.iterrows():
            feedback_parts.append(
                f"- {row['name']}: Pearson IC={row['pearson_ic']:.4f}, "
                f"Sharpe={row['sharpe']:.4f}"
            )
        feedback = "\n".join(feedback_parts)

        # Get best factors for reference
        best_factors = metrics_df.nlargest(5, 'pearson_ic')
        best_feedback_parts = []
        for _, row in best_factors.iterrows():
            best_feedback_parts.append(
                f"- {row['name']}: Pearson IC={row['pearson_ic']:.4f}, "
                f"Sharpe={row['sharpe']:.4f}"
            )
        best_feedback = "\n".join(best_feedback_parts)

        refined_specs = []
        for spec in current_specs:
            if spec.name in worst_factor_names:
                logger.info(f"Refining factor: {spec.name}")

                # Build refinement prompt
                refinement_prompt = f"""
You are refining an underperforming alpha factor.

WORST PERFORMING FACTORS:
{feedback}

BEST PERFORMING FACTORS (for inspiration):
{best_feedback}

CURRENT FACTOR TO REFINE:
Name: {spec.name}
Description: {spec.description}
Reasoning: {spec.reasoning}
Current AST: {json.dumps(spec.ast, indent=2)}

This factor performed poorly with Pearson IC near zero or negative.

Generate an IMPROVED version of this factor that:
1. Uses different operators or combinations
2. Uses different window sizes
3. Learns from the best-performing factors' structure
4. Maintains the general intent but improves execution

Return a JSON object with the same structure as before (name, description, reasoning, expr).
Make sure the factor name is slightly different to indicate it's refined (e.g., add "_v2" suffix).
"""

                # Try to get refined factor from LLM
                max_attempts = 3
                refined = False
                for attempt in range(max_attempts):
                    try:
                        payload = self._call_llm(refinement_prompt, schema=self.SINGLE_FACTOR_JSON_SCHEMA)
                        factor_data = self._extract_single_factor(payload)

                        # Create refined FactorSpec
                        refined_spec = FactorSpec(
                            name=factor_data["name"],
                            description=factor_data["description"],
                            reasoning=factor_data["reasoning"],
                            ast=factor_data["expr"],
                        )

                        # Validate the AST
                        self.compiler._validate(refined_spec.ast)

                        refined_specs.append(refined_spec)
                        logger.info(f"✅ Successfully refined factor: {spec.name} -> {refined_spec.name}")
                        refined = True
                        break

                    except Exception as e:
                        logger.warning(f"❌ Refinement attempt {attempt+1} failed for {spec.name}: {e}")

                if not refined:
                    logger.warning(f"Failed to refine {spec.name}, keeping original")
                    refined_specs.append(spec)
            else:
                # Keep well-performing factors as-is
                refined_specs.append(spec)

        logger.info(f"Refinement complete: {len(refined_specs)} factors")
        return refined_specs

    def _improve_hypothesis_with_feedback(self, hypothesis: str, feedback: str,
                                        metrics: Dict[str, Any]) -> str:
        """Improve hypothesis based on performance feedback."""
        # Extract aggregate metrics
        avg_correlation = metrics.get('avg_correlation', 0.0)
        avg_sharpe = metrics.get('avg_sharpe', 0.0)

        # Build improvement prompt
        improvement_prompt = build_hypothesis_improvement_prompt(hypothesis, feedback, avg_correlation, avg_sharpe)

        try:
            # Call LLM to improve hypothesis
            response = self._openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a quantitative researcher improving market hypotheses based on factor performance feedback. Your goal is to iteratively improve factors using correlation and Sharpe ratio metrics."},
                    {"role": "user", "content": improvement_prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )

            improved_hypothesis = response.choices[0].message.content.strip()
            return improved_hypothesis

        except Exception as e:
            logger.error(f"Error improving hypothesis: {e}")
            # Fallback: append feedback insights to original hypothesis
            if avg_correlation < 0.05:
                return hypothesis + " Focus on short-term momentum, volume imbalances, and volatility patterns."
            else:
                return hypothesis + " Enhance with microstructure dynamics and order flow information."


# ------------------------------
# FACTOR CORRELATION UTILITIES
# ------------------------------

# ------------------------------
# USAGE EXAMPLE
# ------------------------------
# Note: Factor correlation functions have been moved to factor_evaluation.py
def load_credentials(credentials_path: str = "credentials.json") -> Dict[str, str]:
    """Load API credentials from JSON file."""
    try:
        with open(credentials_path, 'r') as f:
            credentials = json.load(f)
        return credentials
    except FileNotFoundError:
        print(f"Warning: {credentials_path} not found. Using environment variables.")
        return {}
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON in {credentials_path}. Using environment variables.")
        return {}

def load_real_market_data(data_path: str = "/home/lichenhui/data/1min", n_tickers: int = 5) -> pd.DataFrame:
    """Load real market data from CSV files in the data directory.
    
    Args:
        data_path: Path to directory containing ticker CSV files
        n_tickers: Number of top liquid tickers to load
        
    Returns:
        Combined DataFrame with multi-level columns (ticker, field)
    """
    csv_files = glob.glob(os.path.join(data_path, "*.csv"))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_path}. Cannot load market data.")
    
    # Load and rank tickers by average volume using parallel processing
    ticker_volumes = {}
    ticker_data = {}

    # Process enough files to get the requested number of tickers
    # Assuming ~80% success rate, process 1.5x the requested amount for safety
    files_needed = min(len(csv_files), int(n_tickers * 1.5))
    files_to_process = csv_files[:files_needed]

    # Use limited CPU cores for parallel processing to avoid overloading system
    n_cores = min(20, mp.cpu_count())
    print(f"Processing {len(files_to_process)} CSV files using {n_cores} CPU cores to get {n_tickers} tickers")
    
    # Create input data for parallel processing
    csv_file_info = [(csv_file, 20) for csv_file in files_to_process]  # 20 is max_samples parameter
    
    # Process CSV files SERIALLY (multiprocessing causes BrokenPipeError with large DataFrames)
    print("Processing files serially to avoid memory issues...")
    results = [_process_csv_for_data(info) for info in csv_file_info]
    
    # Collect results and log warnings
    ticker_files = {}  # Store file paths instead of dataframes
    for ticker, result in results:
        if ticker is not None and result is not None:
            csv_file, avg_volume = result
            ticker_volumes[ticker] = avg_volume
            ticker_files[ticker] = csv_file
        elif ticker is not None:
            print(f"Skipping {ticker}: missing required columns or insufficient data")

    if not ticker_volumes:
        raise ValueError("No valid ticker data found. Cannot load market data.")

    # Select top n_tickers by volume
    top_tickers = sorted(ticker_volumes.items(), key=lambda x: x[1], reverse=True)[:n_tickers]

    print(f"Loaded {len(top_tickers)} tickers: {[ticker for ticker, _ in top_tickers]}")

    # Now load the actual data SERIALLY for the top tickers only
    print(f"Loading data for top {len(top_tickers)} tickers (serial)...")
    for ticker, _ in top_tickers:
        try:
            csv_file = ticker_files[ticker]
            df = pd.read_csv(csv_file)
            df.columns = df.columns.str.lower()

            # Process same as in _process_csv_for_data
            if 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'])
            elif 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
            else:
                continue

            df = df[df['datetime'] >= TWO_YEARS_AGO]
            df.set_index('datetime', inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volume']].sort_index()
            df = df[(df > 0).all(axis=1)].dropna()

            ticker_data[ticker] = df
        except Exception as e:
            print(f"Error loading {ticker}: {e}")

    # Return ticker metadata and file paths for one-by-one processing
    # This avoids loading all tickers into memory at once
    print(f"Prepared {len(top_tickers)} tickers for one-by-one processing")

    return {
        'tickers': [ticker for ticker, _ in top_tickers],
        'ticker_files': ticker_files,
        'ticker_data': ticker_data  # Keep loaded data for reference
    }

if __name__ == "__main__":
    import argparse
    
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
    parser.add_argument('--n_tickers', type=int, default=2000,
                       help='Number of top liquid tickers to load (default: 2000)')

    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("ALPHA AGENT FACTOR PIPELINE STARTING")
    logger.info("="*80)
    logger.info(f"🤖 Selected Model: {args.model}")
    logger.info(f"🔄 Iterations: {args.iterations}")
    logger.info(f"⏱️  Historical Data: {args.n_minutes} minutes")
    logger.info(f"📊 Top Tickers to Load: {args.n_tickers}")
    
    # 1. Load API credentials
    credentials = load_credentials("credentials.json")
    api_key = credentials.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")

    # Initialize FactorAgent with OpenAI API
    if not api_key:
        logger.error("No OpenAI API key found. Set OPENAI_API_KEY environment variable or add to credentials.json")
        raise ValueError("OpenAI API key required")

    logger.info("Using FactorAgent with OpenAI API")
    agent = FactorAgent(api_key=api_key, data_path="/home/lichenhui/data/1min")
    
    
    # 2. Load real market data - TOP N TICKERS
    logger.info("\n" + "="*50)
    logger.info(f"LOADING MARKET DATA - TOP {args.n_tickers} TICKERS")
    logger.info("="*50)
    # Load top N most liquid tickers (processed one-by-one to save memory)
    market_data_info = load_real_market_data(data_path="/home/lichenhui/data/1min", n_tickers=args.n_tickers)
    tickers = market_data_info['tickers']
    ticker_data_dict = market_data_info['ticker_data']
    logger.info(f"Prepared {len(tickers)} tickers for one-by-one processing")

    # 3. Load all hypothesis JSON files from pdf_results
    logger.info("\n" + "="*50)
    logger.info("LOADING HYPOTHESES FROM PDF_RESULTS")
    logger.info("="*50)

    pdf_results_path = "/home/lichenhui/data/alphaAgent/pdf_results"
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

    # Generate hypothesis text for processing
    if not all_hypotheses:
        logger.error("No valid hypotheses found in JSON files")
        raise ValueError("No valid hypotheses found")

    # Iterate through all hypotheses
    for current_hypothesis_idx in range(len(all_hypotheses)):
        hypothesis = all_hypotheses[current_hypothesis_idx]['hypothesis_text']

        logger.info("\n" + "="*80)
        logger.info(f"PROCESSING HYPOTHESIS {current_hypothesis_idx + 1}/{len(all_hypotheses)}")
        logger.info("="*80)
        logger.info(f"  Paper: {all_hypotheses[current_hypothesis_idx]['paper_title']}")
        logger.info(f"  Hypothesis: {hypothesis[:200]}...")
        logger.info(f"  Source: {all_hypotheses[current_hypothesis_idx]['source_file']}")

        # Define path for saving/loading ASTs (unique per hypothesis)
        ast_save_path = f"/home/lichenhui/data/alphaAgent/saved_asts_hypothesis_{current_hypothesis_idx + 1}.json"
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
            specs = agent.parse_hypothesis(hypothesis)
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

        # 4. Compute factors and metrics ONE FACTOR AT A TIME to minimize memory usage
        logger.info("\n" + "="*50)
        logger.info("COMPUTING FACTORS AND METRICS (ONE FACTOR AT A TIME)")
        logger.info("="*50)
        try:
            n_tickers = len(tickers)
            n_factors = len(compiled_factors)

            logger.info(f"Computing {n_factors} factors for {n_tickers} tickers (one factor at a time to save memory)")

            # Compute forward returns once for all tickers
            logger.info("Computing forward returns for all tickers...")
            forward_returns_dict = {}  # {ticker: Series}

            for ticker_idx, ticker in enumerate(tickers):
                ticker_data = ticker_data_dict.get(ticker)
                if ticker_data is None:
                    continue

                try:
                    forward_horizon = 30
                    close_prices = ticker_data['close']
                    forward_return = close_prices.shift(-forward_horizon) / close_prices - 1
                    forward_returns_dict[ticker] = forward_return
                except Exception as e:
                    logger.warning(f"  Error computing forward returns for {ticker}: {e}")
                    forward_returns_dict[ticker] = pd.Series(0.0, index=ticker_data.index)

            logger.info(f"Computed forward returns for {len(forward_returns_dict)} tickers")

            # Process each factor individually: compute factor values, then metrics, then garbage collect
            factor_metrics = []

            for factor_idx, factor_item in enumerate(compiled_factors):
                factor_name = factor_item["name"]
                logger.info(f"\nProcessing factor {factor_idx+1}/{n_factors}: {factor_name}")

                # Compute this factor for all tickers
                factor_results_single = {}  # {ticker: Series} - only for current factor

                for ticker_idx, ticker in enumerate(tickers):
                    if ticker_idx % 500 == 0:
                        logger.info(f"  Computing {factor_name} for ticker {ticker_idx+1}/{n_tickers}...")

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

                logger.info(f"  Computed {factor_name} for {len(factor_results_single)} tickers")

                # Compute metrics for this factor
                logger.info(f"  Computing metrics for {factor_name}...")

                # Compute streaming Pearson IC
                pearson_ic_ts = compute_pearson_ic_timeseries_streaming(
                    factor_results_single,
                    forward_returns_dict
                )
                pearson_ic_ts_clean = pearson_ic_ts.replace([np.inf, -np.inf], np.nan).dropna()

                if len(pearson_ic_ts_clean) > 10:
                    mean_pearson_ic = float(pearson_ic_ts_clean.mean())
                    std_pearson_ic = float(pearson_ic_ts_clean.std(ddof=1))
                    pearson_nw_tstat = _nw_tstat_from_series(pearson_ic_ts_clean, maxlags=5)
                else:
                    mean_pearson_ic = 0.0
                    std_pearson_ic = 0.0
                    pearson_nw_tstat = 0.0

                # Compute streaming Sharpe and get PnL timeseries
                sharpe_annualized, pnl_timeseries = compute_crosssec_sharpe_streaming(
                    factor_results_single,
                    forward_returns_dict,
                    annualize_minutes=252*390
                )

                # Save PnL visualization for this factor
                save_factor_pnl_visualization(
                    pnl_timeseries,
                    factor_name,
                    sharpe_annualized
                )

                n_timestamps = len(pearson_ic_ts_clean)

                factor_metrics.append({
                    'name': factor_name,
                    'pearson_ic': mean_pearson_ic,
                    'pearson_ic_std': std_pearson_ic,
                    'pearson_nw_tstat': pearson_nw_tstat,
                    'nw_tstat': pearson_nw_tstat,  # For backward compatibility
                    'sharpe': sharpe_annualized,
                    'n_observations': n_timestamps
                })

                logger.info(f"  ✅ {factor_name}: Pearson IC={mean_pearson_ic:.4f}±{std_pearson_ic:.4f} (t={pearson_nw_tstat:.2f}), "
                           f"Sharpe={sharpe_annualized:.4f}, n={n_timestamps} timestamps")

                # CRITICAL: Free up factor data immediately after computing metrics
                del factor_results_single, pearson_ic_ts, pearson_ic_ts_clean, pnl_timeseries
                gc.collect()
                logger.info(f"  🗑️  Freed memory for {factor_name}")

            # Clear forward returns to free memory
            logger.info("\n🗑️  Freeing forward returns memory...")
            del forward_returns_dict
            gc.collect()

            # 7. Display comprehensive summary metrics
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

            # Save detailed results to CSV
            output_path = "/home/lichenhui/data/alphaAgent/factor_metrics_detailed.csv"
            metrics_df.to_csv(output_path, index=False)
            logger.info(f"\nDetailed factor metrics saved to: {output_path}")

            # 7b. AST-based iterative refinement
            logger.info("\n" + "="*50)
            logger.info("AST-BASED ITERATIVE REFINEMENT")
            logger.info("="*50)

            # Check if ASTs were loaded (i.e., this is a subsequent iteration)
            if loaded_from_previous and len(specs) > 0:
                logger.info("Refining worst-performing factors from loaded ASTs")

                # Refine the worst 10 factors
                refined_specs = agent.refine_asts_with_metrics(specs, metrics_df, n_to_refine=10)

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
                logger.info("First iteration - ASTs saved. Run again to refine based on metrics.")

            # 8. Model-driven iterative factor improvement - DISABLED (self-referential noise)
            # This loop also optimizes against model predictions without real out-of-sample targets
            # causing drift and meaningless optimization. Disabled per user request.
            logger.info(f"\n" + "="*50)
            logger.info("MODEL-DRIVEN ITERATIVE IMPROVEMENT - SKIPPED")
            logger.info("="*50)
            logger.info("Model-driven improvement loop disabled - it optimizes without real targets")
        
            # try:
            #     # List available models for iterative improvement
            #     available_models = agent.list_available_models()
            #     logger.info(f"Available models for improvement: {available_models}")
            #     
            #     # Use the model specified via command-line argument
            #     selected_model = args.model
            #     logger.info(f"Using specified simplified model for this run: {selected_model}")
            #     
            #     if available_models:
            #         # Run model-driven iterative improvement with single model
            #         improvement_results = agent.model_driven_factor_improvement(
            #             initial_hypothesis=hypothesis,
            #             market_data=market_data,
            #             model_names=[selected_model],
            #             n_iterations=args.iterations,
            #             n_minutes=args.n_minutes
            #         )
            #         
            #         # Report results from each iteration
            #         logger.info(f"\n🔄 MODEL-DRIVEN IMPROVEMENT RESULTS:")
            #         for iteration_data in improvement_results['history']:
            #             logger.info(f"\nIteration {iteration_data['iteration']}:")
            #             logger.info(f"  Hypothesis: {iteration_data['hypothesis'][:100]}...")
            #             
            #             if 'model_predictions' in iteration_data:
            #                 for model_name, pred in iteration_data['model_predictions'].items():
            #                     if isinstance(pred, (int, float)):
            #                         logger.info(f"  {model_name} prediction: {pred:.6f}")
            #             
            #             if 'consensus_mean' in iteration_data and iteration_data['consensus_mean'] is not None:
            #                 logger.info(f"  Model consensus: mean={iteration_data['consensus_mean']:.6f}, std={iteration_data['consensus_std']:.6f}")
            #             
            #             logger.info(f"  Feedback: {iteration_data['feedback'][:150]}...")
            #         
            #         # Report final results
            #         logger.info(f"\n🎯 FINAL MODEL-DRIVEN RESULTS:")
            #         logger.info(f"Final improved hypothesis: {improvement_results['final_hypothesis']}")
            #         
            #         for model_name, pred in improvement_results['final_predictions'].items():
            #             logger.info(f"Final {model_name} prediction: {pred:.6f}")
            #         
            #         # Analyze improvement trajectory
            #         consensus_means = [iter_data.get('consensus_mean') for iter_data in improvement_results['history'] 
            #                          if iter_data.get('consensus_mean') is not None]
            #         if len(consensus_means) > 1:
            #             improvement_trend = consensus_means[-1] - consensus_means[0]
            #             logger.info(f"📈 Prediction trend: {improvement_trend:+.6f} (final - initial)")
            #         
            #         # Save final factor matrix for potential further analysis
            #         logger.info(f"Final factor matrix shape: {improvement_results['final_factor_matrix'].shape}")
            #         
            #     else:
            #         logger.warning("No models available for iterative improvement. Running basic factor improvement instead.")
            #         
            #         # Fallback to basic iterative improvement without models
            #         basic_improvement_results = agent.iterative_factor_improvement(
            #             initial_hypothesis=hypothesis,
            #             n_iterations=args.iterations,
            #             n_minutes=args.n_minutes
            #         )
            #         logger.info(f"Basic improvement final hypothesis: {basic_improvement_results['final_hypothesis']}")
            #         logger.info(f"Basic improvement final metrics: {basic_improvement_results['final_metrics']}")
            #         
            # except Exception as model_error:
            #     logger.error(f"Error in model-driven improvement: {model_error}")
            #     logger.info("Falling back to basic factor improvement...")
            #     
            #     try:
            #         # Fallback to basic improvement
            #         fallback_results = agent.iterative_factor_improvement(
            #             initial_hypothesis=hypothesis,
            #             n_iterations=args.iterations,
            #             n_minutes=args.n_minutes
            #         )
            #         logger.info(f"Fallback improvement completed successfully")
            #         logger.info(f"Final metrics: {fallback_results['final_metrics']}")
            #     except Exception as fallback_error:
            #         logger.error(f"Even fallback improvement failed: {fallback_error}")
        
            logger.info("Pipeline completed successfully - all iterative improvement loops disabled")

        except Exception as e:
            logger.error(f"Error in pipeline: {e}")
            raise
    
    logger.info("\n" + "="*80)
    logger.info("ALPHA AGENT FACTOR PIPELINE FINISHED")
    logger.info("="*80)
