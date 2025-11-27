#!/usr/bin/env python3
"""
AST Compiler Module

This module provides AST compilation functionality for converting JSON AST
expressions into executable Python callables using the Operators library.

Components:
- AST type definitions
- Allowed functions and symbols configuration
- ASTValidationError exception
- FactorSpec dataclass
- ASTCompiler class for compiling ASTs to Python functions

Extracted from alpha_agent_factor.py for better modularity.
"""

import logging
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd

from alpha_agent_operators import Operators

logger = logging.getLogger(__name__)


# ------------------------------
# AST DEFINITIONS
# ------------------------------

# We keep the AST intentionally small and typed. Each node is a dict-serializable
# structure with a `type` field and specific attributes.

AST = Dict[str, Any]

# Allowed operator names map 1:1 to Operators methods.
# Normalized function mapping - all lowercase keys to prevent guard bypass
ALLOWED_FUNCS = {
    # time-series operators
    "delay": "delay",
    "ts_mean": "ts_mean",
    "ts_std": "ts_std",
    "ts_zscore": "ts_zscore",
    "ts_min": "ts_min",
    "ts_max": "ts_max",
    "ts_rank": "ts_rank",
    "ts_sum": "ts_sum",
    "ts_pctile": "ts_pctile",
    "ts_argmin": "ts_argmin",
    "ts_argmax": "ts_argmax",
    "ts_median": "ts_median",
    "ts_prod": "ts_prod",
    "corr_ts": "corr_ts",
    "cov_ts": "cov_ts",
    "beta_ts": "beta_ts",
    # NOTE: 'ret' removed - requires string 'mode' parameter
    "volatility": "volatility",
    "ema": "ema",
    "wma": "wma",
    "diff": "diff",
    # cross-sectional operators
    "cs_rank": "cs_rank",
    "cs_zscore": "cs_zscore",
    "cs_winsor": "cs_winsor",
    # NOTE: 'cs_rescale' removed - requires string 'norm' parameter
    "cs_neutralize": "cs_neutralize",
    "cs_neutralize_multi": "cs_neutralize_multi",
    "debias_cs": "debias_cs",
    "demean_ts": "demean_ts",
    # NOTE: 'smooth_ts' removed - requires string 'method' parameter
    # NOTE: 'fillna_ts' removed - requires string 'method' parameter
    # algebraic operators
    "add": "add",
    "sub": "sub",
    "mul": "mul",
    "div": "div",
    "abs": "abs",
    "sign": "sign",
    "clip": "clip",
    "pow": "pow",
    "log1p": "log1p",
    "sqrt": "sqrt",
    "tanh": "tanh",
    "neg": "neg",
    "inv": "inv",
    "signed_pow": "signed_pow",
    "safe_log": "safe_log",
    "pct_change": "pct_change",
    # comparisons & conditionals
    "gt": "gt",
    "lt": "lt",
    "eq": "eq",
    "ne": "ne",
    "ge": "ge",
    "le": "le",
    "between": "between",
    "cond": "cond",
    "emin": "emin",
    "emax": "emax",
    "land": "land",
    "lor": "lor",
    "lnot": "lnot",
    "coalesce": "coalesce",
    # volume/range/microstructure helpers
    "turnover": "turnover",
    "adv": "adv",
    "hl_range": "hl_range",
    "atr": "atr",
    "true_range": "true_range",
    # NOTE: 'breakout' removed - requires string 'direction' parameter
    "persist": "persist",
    "crossover": "crossover",
    "crossunder": "crossunder",
    "spread": "spread",
    "rel_spread": "rel_spread",
    "order_imb": "order_imb",
    "amihud": "amihud",
}

# Symbols we allow the LLM to reference directly from `data`
ALLOWED_SYMBOLS = {"open", "high", "low", "close", "volume"}

# Windows / constants guardrails
MAX_WINDOW = 300  # as per spec (300 minutes history)
MAX_POW = 8       # avoid explosive exponents


@dataclass
class FactorSpec:
    name: str
    description: str
    reasoning: str
    ast: AST


class ASTValidationError(ValueError):
    pass


class CompiledFactorFunction:
    """Picklable wrapper for compiled factor functions.

    This class is designed to be picklable for use with multiprocessing.
    Instead of using a closure (which can't be pickled), we store the AST
    and operators as instance attributes.
    """

    def __init__(self, ast_obj: AST, operators: Operators):
        self.ast_obj = ast_obj
        self.ops = operators

    def _get_field(self, data: pd.DataFrame, field: str):
        """Get a field from data, handling both flat and MultiIndex columns."""
        if field in data.columns:
            return data[field]
        # MultiIndex case: (ticker, field) or (time, field)
        try:
            return data.xs(field, level=1, axis=1)
        except Exception:
            raise KeyError(f"Field '{field}' not found in data columns")

    def _make_env(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create evaluation environment with data and operators."""
        env: Dict[str, Any] = {"pd": pd, "np": np}

        # Expose the raw data fields safely
        for sym in ALLOWED_SYMBOLS:
            env[sym] = self._get_field(data, sym)

        # Bind Operators methods with short names
        for k, v in ALLOWED_FUNCS.items():
            try:
                env[k] = getattr(self.ops, v)
            except AttributeError as e:
                # Better error message showing which function is missing
                raise AttributeError(f"Operators object missing method '{v}' (referenced by '{k}'). Available methods: {[m for m in dir(self.ops) if not m.startswith('_')][:10]}...") from e

        return env

    def _eval(self, node: AST, env: Dict[str, Any]):
        """Evaluate AST node recursively."""
        if not isinstance(node, dict):
            if isinstance(node, (int, float, bool)):
                logger.warning(f"Raw constant {node} found in AST - should be wrapped as {{'type': 'const', 'value': {node}}}")
                return node
            raise ASTValidationError(f"AST node must be a dictionary, got {type(node).__name__}: {node}")

        ntype = node.get("type")
        if ntype == "symbol":
            name = node["name"]
            if name not in ALLOWED_SYMBOLS:
                raise ASTValidationError(f"Symbol '{name}' not allowed")
            return env[name]
        elif ntype == "const":
            return node["value"]
        elif ntype == "call":
            func = node.get("func") or node.get("fn")
            if func is None:
                raise ASTValidationError(f"Call node missing function name. Node: {node}")
            func = func.lower()
            if func not in ALLOWED_FUNCS:
                available_funcs = list(ALLOWED_FUNCS.keys())[:10]
                raise ASTValidationError(f"Function '{func}' not allowed. Available functions include: {available_funcs}...")
            args = [self._eval(arg, env) for arg in node.get("args", [])]
            if "kwargs" in node:
                raise ASTValidationError("kwargs field not allowed in AST expressions per prompt contract")
            return env[func](*args)
        else:
            raise ASTValidationError(f"Unknown node type: {ntype}")

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        """Execute the compiled factor function."""
        env = self._make_env(data)
        out = self._eval(self.ast_obj, env)
        # Force 1-D output
        out = np.asarray(out).squeeze() if isinstance(out, (list, np.ndarray)) else out
        if isinstance(out, pd.DataFrame):
            out = out.iloc[:, 0] if out.shape[1] == 1 else out.mean(axis=1)
        if not isinstance(out, pd.Series):
            out = pd.Series(out, index=data.index)
        return out


class ASTCompiler:
    """Compiles JSON AST into a Python callable using Operators.

    The compiled callable has the signature: fn(data: pd.DataFrame) -> pd.Series
    where `data` columns can be a flat Index like ['open','high',...] or a
    MultiIndex (datetime, field) – we use Operators helpers that should already
    support both styles in your codebase.
    """

    def __init__(self, operators: Operators):
        self.ops = operators

    # --------- Public API ---------

    def compile(self, ast_obj: AST) -> Callable[[pd.DataFrame], pd.Series]:
        self._validate(ast_obj)
        # Return a picklable function wrapper instead of a local function
        return CompiledFactorFunction(ast_obj, self.ops)

    # --------- Evaluation ---------

    def _make_env(self, data: pd.DataFrame) -> Dict[str, Any]:
        # Accept either flat columns or MultiIndex with level of fields
        env: Dict[str, Any] = {"pd": pd, "np": np}
        # Expose the raw data fields safely
        def _get(field: str):
            if field in data.columns:
                return data[field]
            # MultiIndex case: (ticker, field) or (time, field)
            try:
                return data.xs(field, level=1, axis=1)
            except Exception:
                raise KeyError(f"Field '{field}' not found in data columns")
        for sym in ALLOWED_SYMBOLS:
            env[sym] = _get(sym)
        # Bind Operators methods with short names
        for k, v in ALLOWED_FUNCS.items():
            env[k] = getattr(self.ops, v)
        return env

    def _eval(self, node: AST, env: Dict[str, Any]):
        # Handle case where node is not a dictionary (e.g., raw integer/string)
        # Allow raw numbers as a fallback for LLM-generated ASTs that don't wrap constants properly
        if not isinstance(node, dict):
            if isinstance(node, (int, float, bool)):
                logger.warning(f"Raw constant {node} found in AST - should be wrapped as {{'type': 'const', 'value': {node}}}")
                return node
            raise ASTValidationError(f"AST node must be a dictionary, got {type(node).__name__}: {node}. Numbers must be wrapped as {{'type': 'const', 'value': NUMBER}}")

        ntype = node.get("type")
        if ntype == "symbol":
            name = node["name"]
            if name not in ALLOWED_SYMBOLS:
                raise ASTValidationError(f"Symbol '{name}' not allowed")
            return env[name]
        elif ntype == "const":
            return node["value"]
        elif ntype == "call":
            # Handle both "func" and "fn" field names
            func = node.get("func") or node.get("fn")
            if func is None:
                raise ASTValidationError(f"Call node missing function name. Node: {node}")
            # Normalize function name to lowercase to prevent guard bypass
            func = func.lower()
            if func not in ALLOWED_FUNCS:
                available_funcs = list(ALLOWED_FUNCS.keys())[:10]  # Show first 10
                raise ASTValidationError(f"Function '{func}' not allowed. Available functions include: {available_funcs}...")
            args = [self._eval(arg, env) for arg in node.get("args", [])]
            # Enforce no-kwargs contract from prompt
            if "kwargs" in node:
                raise ASTValidationError("kwargs field not allowed in AST expressions per prompt contract")
            # Extra guards for windows and powers
            self._guard_call(func, args, {})
            return env[func](*args)
        else:
            raise ASTValidationError(f"Unknown node type: {ntype}")

    # --------- Validation ---------

    def _validate(self, node: AST):
        # Handle case where node is not a dictionary (e.g., raw integer/string)
        # Allow raw numbers as a fallback for LLM-generated ASTs that don't wrap constants properly
        if not isinstance(node, dict):
            if isinstance(node, (int, float, bool)):
                logger.warning(f"Raw constant {node} found in AST during validation - should be wrapped as {{'type': 'const', 'value': {node}}}")
                return
            raise ASTValidationError(f"AST node must be a dictionary, got {type(node).__name__}: {node}. Numbers must be wrapped as {{'type': 'const', 'value': NUMBER}}")

        ntype = node.get("type")
        if ntype == "symbol":
            name = node.get("name")
            if name not in ALLOWED_SYMBOLS:
                raise ASTValidationError(f"Unknown symbol '{name}'")
            return
        if ntype == "const":
            # Must be JSON-serializable primitive (no strings - those require kwargs)
            if not isinstance(node.get("value"), (int, float, bool)):
                raise ASTValidationError("const must be int/float/bool (string parameters not supported in AST - use operators with numeric args only)")
            return
        if ntype == "call":
            # Handle both "func" and "fn" field names
            func = node.get("func") or node.get("fn")
            if func is None:
                raise ASTValidationError(f"Call node missing function name. Expected 'func' or 'fn' field. Node: {node}")
            # Normalize function name to lowercase to prevent guard bypass
            func = func.lower()
            if func not in ALLOWED_FUNCS:
                available_funcs = list(ALLOWED_FUNCS.keys())[:10]  # Show first 10
                raise ASTValidationError(f"Unknown function '{func}'. Available functions include: {available_funcs}... (and {len(ALLOWED_FUNCS)-10} more)")
            for arg in node.get("args", []):
                self._validate(arg)
            # Enforce no-kwargs contract from prompt
            if "kwargs" in node:
                raise ASTValidationError("kwargs field not allowed in AST expressions per prompt contract")
            # Post-check windows
            self._guard_call(func, node.get("args", []), {})
            return
        raise ASTValidationError(f"Invalid node type '{ntype}'. Expected 'symbol', 'const', or 'call'. Full node: {node}")

    def _guard_call(self, func: str, args: List[Any], kwargs: Dict[str, Any]):
        # Guard against windows > MAX_WINDOW and extreme exponents, etc.
        # func is already normalized to lowercase by callers
        # args might be AST nodes during validation or actual values during evaluation

        # Load function signatures from Operators class using introspection
        if func in ALLOWED_FUNCS:
            method_name = ALLOWED_FUNCS[func]
            if hasattr(Operators, method_name):
                method = getattr(Operators, method_name)
                try:
                    sig = inspect.signature(method)
                    # Get parameters excluding 'self' (if static method, there's no self)
                    params = [p for p in sig.parameters.values() if p.name != 'self']

                    # Count required parameters (those without defaults)
                    required_params = [p for p in params if p.default == inspect.Parameter.empty]
                    optional_params = [p for p in params if p.default != inspect.Parameter.empty]

                    min_args = len(required_params)
                    max_args = len(params) if not any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params) else float('inf')

                    actual_count = len(args)

                    # Validate argument count
                    if actual_count < min_args:
                        param_names = ', '.join([p.name for p in required_params])
                        raise ASTValidationError(
                            f"Function '{func}' expects at least {min_args} arguments ({param_names}) but got {actual_count}. "
                            f"Check function signature in Operators.{method_name}"
                        )
                    elif max_args != float('inf') and actual_count > max_args:
                        param_names = ', '.join([p.name for p in params])
                        raise ASTValidationError(
                            f"Function '{func}' expects at most {max_args} arguments ({param_names}) but got {actual_count}. "
                            f"Check function signature in Operators.{method_name}"
                        )
                except Exception as e:
                    # If introspection fails, log warning and continue
                    logger.warning(f"Could not introspect function signature for '{func}': {e}")

        window_like_funcs = {"delay", "ts_mean", "ts_std", "ts_zscore", "ts_min", "ts_max",
                             "ts_rank", "ts_sum", "ts_median", "ts_prod", "corr_ts", "ret",
                             "volatility", "pct_change"}

        def extract_value(arg):
            """Extract numeric value from AST node or return as-is if already a number"""
            if isinstance(arg, dict) and arg.get("type") == "const":
                return arg.get("value")
            elif isinstance(arg, (int, float)):
                return arg
            return None

        if func in window_like_funcs:
            # window is usually last positional argument (except RET/DELAY)
            # We check all integer args/kwargs named 'w', 'k'
            for a in args:
                val = extract_value(a)
                if isinstance(val, int) and val > MAX_WINDOW:
                    raise ASTValidationError(f"Window {val} exceeds MAX_WINDOW={MAX_WINDOW}")
            for k, v in kwargs.items():
                val = extract_value(v)
                if k in {"w", "k"} and isinstance(val, int) and val > MAX_WINDOW:
                    raise ASTValidationError(f"Window {val} exceeds MAX_WINDOW={MAX_WINDOW}")

        if func == "pow":  # lowercase now
            # Usually POW(x, p)
            if len(args) >= 2:
                exp_val = extract_value(args[1])
                if isinstance(exp_val, (int, float)) and abs(exp_val) > MAX_POW:
                    raise ASTValidationError(f"Exponent {exp_val} too large (>{MAX_POW})")
