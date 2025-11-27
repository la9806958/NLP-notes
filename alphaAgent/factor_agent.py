#!/usr/bin/env python3
"""
FactorAgent class for Alpha Agent Factor Pipeline

This module contains the main FactorAgent class that:
- Interfaces with LLM (OpenAI) to generate factor specifications
- Parses market hypotheses into executable factor expressions using AST
- Compiles and evaluates factors
- Performs iterative factor improvement based on metrics
"""

import os
import json
import glob
import logging
import jsonschema
import multiprocessing as mp
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from alpha_agent_operators import Operators
from config import TWO_YEARS_AGO, DEFAULT_N_TICKERS, DEFAULT_N_CORES
from data_loader import _process_csv_for_volume
from ast_compiler import FactorSpec, ASTValidationError, ASTCompiler
from validator import hash_ast
from prompts import (
    build_single_factor_prompt,
    build_error_recovery_prompt,
    generate_factor_feedback,
    extract_operator_signatures,
    build_refinement_prompt,
    build_analysis_prompt,
    build_refinement_from_analysis_prompt
)

logger = logging.getLogger(__name__)


class FactorAgent:
    """Parses market hypotheses into executable factor expressions using an AST.

    The LLM is asked to return *strict JSON* following a schema; we then
    validate, compile, and execute those factors.
    """

    # JSON schemas for LLM responses
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

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o",
                 data_path: str = "/home/lichenhui/data/1min"):
        """Initialize FactorAgent.

        Args:
            api_key: OpenAI API key
            model: Model name to use
            data_path: Path to market data directory
        """
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

    def _load_top_liquid_tickers(self, n_tickers: int = DEFAULT_N_TICKERS) -> List[str]:
        """Load top N most liquid tickers by volume from data path.

        Returns list of ticker symbols sorted by average volume descending.
        """
        logger.info(f"Loading top {n_tickers} liquid tickers from {self.data_path}")
        if not os.path.exists(self.data_path):
            logger.warning(f"Data path {self.data_path} does not exist. Using mock tickers.")
            return [f"TICK_{i:04d}" for i in range(n_tickers)]

        # Look for CSV files in the data folder
        csv_files = glob.glob(os.path.join(self.data_path, "*.csv"))

        if not csv_files:
            logger.warning(f"No CSV files found in {self.data_path}. Using mock tickers.")
            return [f"TICK_{i:04d}" for i in range(n_tickers)]

        # Use limited CPU cores for parallel processing
        n_cores = min(DEFAULT_N_CORES, mp.cpu_count())
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

        # In _load_top_liquid_tickers(self, n_tickers: int = DEFAULT_N_TICKERS),
        # save down list of tickers, if mapping exists, reuse

        # Collect results and log warnings
        for ticker, avg_volume in results:
            if ticker is not None and avg_volume is not None:
                ticker_volumes[ticker] = avg_volume
            elif ticker is not None and avg_volume is None:
                logger.warning(f"No volume column in {ticker}.csv")

        # Sort by volume descending and take top n_tickers
        sorted_tickers = sorted(ticker_volumes.items(), key=lambda x: x[1], reverse=True)
        top_tickers = [ticker for ticker, _ in sorted_tickers[:n_tickers]]

        logger.info(f"Loaded {len(top_tickers)} liquid tickers from {len(csv_files)} files")
        return top_tickers

    def _load_ticker_data(self, ticker: str, lookback_minutes: int = 300) -> Optional[pd.DataFrame]:
        """Load OHLCV data for a specific ticker.

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
                logger.warning(f"Missing required columns in {csv_path}")
                return None

            # Convert timestamp if present
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df[df['timestamp'] >= TWO_YEARS_AGO]
                df.set_index('timestamp', inplace=True)
            elif 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df[df['datetime'] >= TWO_YEARS_AGO]
                df.set_index('datetime', inplace=True)
            else:
                logger.warning(f"No datetime column found in {csv_path}")
                return None

            if len(df) == 0:
                logger.warning(f"No data from last two years in {csv_path}")
                return None

            # Return all available data
            return df[required_cols]

        except Exception as e:
            logger.error(f"Error loading data for {ticker}: {e}")
            return None

    def _call_llm(self, prompt: str, schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Call LLM with prompt and validate response against schema.

        Args:
            prompt: Prompt to send to LLM
            schema: JSON schema to validate response against

        Returns:
            Parsed and validated JSON response
        """
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

            # Validate against JSON schema
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

    def _build_single_factor_prompt(self, hypothesis: str, reasoning: str, factor_num: int,
                                    existing_factors: List[FactorSpec]) -> str:
        """Build prompt for generating a single factor."""
        return build_single_factor_prompt(hypothesis, reasoning, factor_num, existing_factors)

    def _build_error_recovery_prompt(self, hypothesis: str, reasoning: str, factor_num: int, error_msg: str,
                                     failed_payload: dict = None) -> str:
        """Build error recovery prompt with specific feedback."""
        return build_error_recovery_prompt(hypothesis, reasoning, factor_num, error_msg, failed_payload)

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

    def parse_hypothesis(self, hypothesis: str, reasoning: str = "") -> List[FactorSpec]:
        """Parse hypothesis using sequential LLM calls with error recovery and AST deduplication.

        Args:
            hypothesis: The hypothesis text
            reasoning: The reasoning for the hypothesis

        Returns:
            List of FactorSpec objects
        """
        logger.info("Starting sequential factor generation with error recovery and AST deduplication")

        out: List[FactorSpec] = []
        seen_ast_hashes: set = set()  # Track AST hashes to prevent duplicates

        for i in range(25):  # Generate exactly 25 factors
            logger.info(f"Generating factor {i+1}/25")

            # Build prompt for single factor
            factor_prompt = self._build_single_factor_prompt(hypothesis, reasoning, i+1, out)

            max_attempts = 5
            for attempt in range(max_attempts):
                try:
                    # Get single factor from LLM
                    payload = self._call_llm(factor_prompt, schema=self.SINGLE_FACTOR_JSON_SCHEMA)

                    # Log the raw response for debugging
                    logger.debug(f"Factor {i+1} attempt {attempt+1} raw response: {payload}")

                    # Parse the factor
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
                        try:
                            factor_prompt = self._build_error_recovery_prompt(hypothesis, reasoning, i+1, str(e), payload if 'payload' in locals() else None)
                            logger.info(f"🔄 Attempting error recovery for factor {i+1}")
                        except:
                            logger.error(f"Failed to build error recovery prompt for factor {i+1}")
                    else:
                        logger.error(f"⚠️  All attempts failed for factor {i+1} due to validation errors, SKIPPING")

                except ValueError as e:
                    if "Duplicate AST" in str(e):
                        logger.warning(f"❌ Factor {i+1} attempt {attempt+1} failed: {e}")
                        if attempt < max_attempts - 1:
                            factor_prompt = self._build_single_factor_prompt(hypothesis, reasoning, i+1, out)
                            factor_prompt += f"\n\nIMPORTANT: Your previous attempt was rejected because it generated a duplicate factor expression. You MUST create a factor with a COMPLETELY DIFFERENT structure and operators."
                            logger.info(f"🔄 Retrying factor {i+1} with anti-duplicate prompt")
                        else:
                            logger.error(f"All attempts failed for factor {i+1}, generating fallback")
                            fallback_spec = self._generate_fallback_factor(i+1)
                            fallback_hash = hash_ast(fallback_spec.ast)
                            if fallback_hash not in seen_ast_hashes:
                                seen_ast_hashes.add(fallback_hash)
                                out.append(fallback_spec)
                            else:
                                fallback_spec.ast["args"][1]["value"] = 10 + i
                                fallback_hash = hash_ast(fallback_spec.ast)
                                seen_ast_hashes.add(fallback_hash)
                                out.append(fallback_spec)
                    else:
                        raise

                except Exception as e:
                    logger.warning(f"❌ Factor {i+1} attempt {attempt+1} failed: {e}")

                    if attempt < max_attempts - 1:
                        try:
                            factor_prompt = self._build_error_recovery_prompt(hypothesis, reasoning, i+1, str(e), payload if 'payload' in locals() else None)
                            logger.info(f"🔄 Attempting error recovery for factor {i+1}")
                        except:
                            logger.error(f"Failed to build error recovery prompt for factor {i+1}")
                    else:
                        logger.error(f"All attempts failed for factor {i+1}, generating fallback")
                        fallback_spec = self._generate_fallback_factor(i+1)
                        fallback_hash = hash_ast(fallback_spec.ast)
                        if fallback_hash not in seen_ast_hashes:
                            seen_ast_hashes.add(fallback_hash)
                            out.append(fallback_spec)
                        else:
                            if fallback_spec.ast.get("type") == "call" and fallback_spec.ast.get("args"):
                                for arg in fallback_spec.ast["args"]:
                                    if isinstance(arg, dict) and arg.get("type") == "const":
                                        arg["value"] = arg["value"] + i
                                        break
                            fallback_hash = hash_ast(fallback_spec.ast)
                            seen_ast_hashes.add(fallback_hash)
                            out.append(fallback_spec)

        logger.info(f"Sequential parsing completed: {len(out)}/25 factors generated ({len(seen_ast_hashes)} unique ASTs)")
        return out

    def compile_factors(self, specs: List[FactorSpec]) -> List[Dict[str, Any]]:
        """Compile factor specifications into callable functions.

        Returns:
            List of dictionaries containing compiled factor information
        """
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

    @staticmethod
    def _sanitize_series(s: Union[pd.Series, pd.DataFrame, np.ndarray]) -> pd.Series:
        """Sanitize series by converting to Series and handling inf/nan."""
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0] if s.shape[1] == 1 else s.mean(axis=1)
        elif isinstance(s, np.ndarray):
            s = pd.Series(np.asarray(s).squeeze())
        s = s.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return s

    @staticmethod
    def _zscore(s: pd.Series, window: int = 60) -> pd.Series:
        """Rolling z-score normalize a series.

        Args:
            s: Input series
            window: Rolling window size in periods (default: 60 for 1-hour lookback)

        Returns:
            Rolling z-score normalized series
        """
        rolling_mean = s.rolling(window=window, min_periods=max(1, window // 2)).mean()
        rolling_std = s.rolling(window=window, min_periods=max(1, window // 2)).std()
        # Avoid division by zero
        rolling_std = rolling_std.replace(0, np.nan)
        zscore = (s - rolling_mean) / rolling_std
        # Fill NaN with 0 for stability
        return zscore.fillna(0.0)

    def _generate_factor_feedback(self, factor_metrics: List[Dict], avg_correlation: float,
                                  avg_sharpe: float) -> str:
        """Generate detailed feedback on factor performance."""
        return generate_factor_feedback(factor_metrics, avg_correlation, avg_sharpe)

    def refine_asts_with_metrics(self, current_specs: List[FactorSpec],
                                  metrics_df: pd.DataFrame,
                                  sharpe_threshold: float = 2.0,
                                  ticker_data: Optional[pd.DataFrame] = None) -> List[FactorSpec]:
        """Refine all factors with Sharpe ratio ≤ threshold using two-step LLM approach.

        Step 1: Analyze why best 10 factors outperform worst 10 factors
        Step 2: Use that analysis to construct improved ASTs for ALL factors with Sharpe ≤ threshold

        Args:
            current_specs: Current factor specifications
            metrics_df: DataFrame with factor metrics (must include 'sharpe' and 'pearson_ic' columns)
            sharpe_threshold: Sharpe ratio threshold - refine all factors with Sharpe ≤ this value (default: 2.0)
            ticker_data: Optional ticker data (unused in current implementation)

        Returns:
            Updated list of FactorSpecs
        """
        logger.info(f"Refining all factors with Sharpe ratio ≤ {sharpe_threshold}")
        logger.info("Using two-step refinement: (1) Analyze patterns, (2) Construct improved factors")

        # Get top 10 best factors by Pearson IC
        best_factors_df = metrics_df.nlargest(10, 'pearson_ic')

        # Get all factors with Sharpe <= threshold for refinement
        factors_to_refine_df = metrics_df[metrics_df['sharpe'] <= sharpe_threshold].copy()
        worst_factor_names = set(factors_to_refine_df['name'].tolist())

        # For analysis step, get the 10 worst factors by Sharpe (to compare with best 10)
        worst_for_analysis_df = metrics_df.nsmallest(10, 'sharpe')

        logger.info(f"Best factors (top 10 by Pearson IC): {list(best_factors_df['name'])}")
        logger.info(f"Total factors to refine (Sharpe ≤ {sharpe_threshold}): {len(worst_factor_names)}")
        logger.info(f"Factors to refine: {worst_factor_names}")

        # Prepare data structures for best factors
        best_factors_data = []
        spec_lookup = {spec.name: spec for spec in current_specs}

        for _, row in best_factors_df.iterrows():
            if row['name'] in spec_lookup:
                best_factors_data.append({
                    'name': row['name'],
                    'ast': spec_lookup[row['name']].ast,
                    'pearson_ic': row['pearson_ic'],
                    'sharpe': row['sharpe']
                })

        # Prepare data structures for worst factors (for analysis - use 10 worst by Sharpe)
        worst_factors_data = []
        for _, row in worst_for_analysis_df.iterrows():
            if row['name'] in spec_lookup:
                worst_factors_data.append({
                    'name': row['name'],
                    'ast': spec_lookup[row['name']].ast,
                    'pearson_ic': row['pearson_ic'],
                    'sharpe': row['sharpe']
                })

        # Also prepare data for ALL factors that need refinement (Sharpe <= threshold)
        all_factors_to_refine_data = []
        for _, row in factors_to_refine_df.iterrows():
            if row['name'] in spec_lookup:
                all_factors_to_refine_data.append({
                    'name': row['name'],
                    'ast': spec_lookup[row['name']].ast,
                    'pearson_ic': row['pearson_ic'],
                    'sharpe': row['sharpe']
                })

        # ═══════════════════════════════════════════════════════════════════
        # STEP 1: Ask LLM to analyze why best factors outperform worst factors
        # ═══════════════════════════════════════════════════════════════════
        logger.info("📊 STEP 1: Requesting analysis of best vs worst factors...")
        analysis_prompt = build_analysis_prompt(best_factors_data, worst_factors_data)

        try:
            # Use a simpler response format for analysis (not strict JSON)
            analysis_response = self._openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert quantitative researcher analyzing alpha factor performance."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.3
            ).choices[0].message.content.strip()

            logger.info("✅ Analysis complete!")
            logger.info(f"Analysis response preview (first 500 chars):\n{analysis_response[:500]}...")

        except Exception as e:
            logger.error(f"❌ Failed to get analysis from LLM: {e}")
            logger.warning("Falling back to simple refinement without analysis")
            analysis_response = "Analysis unavailable. Proceeding with standard refinement."

        # ═══════════════════════════════════════════════════════════════════
        # STEP 2: Use analysis to construct improved factors
        # ═══════════════════════════════════════════════════════════════════
        logger.info(f"🔧 STEP 2: Constructing improved factors based on analysis...")
        logger.info(f"Will refine {len(worst_factor_names)} factors with Sharpe ≤ {sharpe_threshold}")

        refined_specs = []
        for spec in current_specs:
            if spec.name in worst_factor_names:
                logger.info(f"Refining factor: {spec.name}")

                # Build refinement prompt using analysis
                # Use all_factors_to_refine_data instead of worst_factors_data for context
                refinement_prompt = build_refinement_from_analysis_prompt(
                    spec, analysis_response, best_factors_data, all_factors_to_refine_data
                )

                max_attempts = 3
                refined = False
                for attempt in range(max_attempts):
                    try:
                        payload = self._call_llm(refinement_prompt, schema=self.SINGLE_FACTOR_JSON_SCHEMA)
                        factor_data = self._extract_single_factor(payload)

                        refined_spec = FactorSpec(
                            name=factor_data["name"],
                            description=factor_data["description"],
                            reasoning=factor_data["reasoning"],
                            ast=factor_data["expr"],
                        )

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
                refined_specs.append(spec)

        num_refined = len([s for s in refined_specs if s.name not in [orig.name for orig in current_specs]])
        num_attempted = len(worst_factor_names)
        logger.info(f"Refinement complete: {num_refined}/{num_attempted} factors successfully refined (Sharpe ≤ {sharpe_threshold})")
        logger.info(f"Total factors returned: {len(refined_specs)}")
        return refined_specs
