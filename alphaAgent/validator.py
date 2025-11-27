#!/usr/bin/env python3
"""
Validator Module

This module contains validation and deduplication functions for factor ASTs:
- AST normalization for semantic comparison
- AST hashing for duplicate detection
- Factor name deduplication

Extracted from alpha_agent_factor.py for better modularity.
"""

import json
import hashlib
import copy
import re
from typing import Any, Dict, List


def normalize_ast_for_comparison(ast: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize AST to canonical form for comparison.

    This handles cases where semantically equivalent operators are used
    (e.g., 'adv' vs 'ts_mean' for volume averaging).

    Args:
        ast: AST dictionary

    Returns:
        Normalized AST dictionary
    """
    ast = copy.deepcopy(ast)

    # Mapping of equivalent operators to canonical forms
    OPERATOR_EQUIVALENTS = {
        'adv': 'ts_mean',  # adv(volume, w) is just ts_mean(volume, w)
    }

    def normalize_node(node):
        if not isinstance(node, dict):
            return node

        if node.get('type') == 'call':
            # Normalize function name to canonical form
            fn = node.get('fn') or node.get('func')
            if fn and fn in OPERATOR_EQUIVALENTS:
                node['fn'] = OPERATOR_EQUIVALENTS[fn]
                if 'func' in node:
                    node['func'] = OPERATOR_EQUIVALENTS[fn]

            # Recursively normalize arguments
            if 'args' in node:
                node['args'] = [normalize_node(arg) for arg in node['args']]

        return node

    return normalize_node(ast)


def hash_ast(ast: Dict[str, Any]) -> str:
    """Create a deterministic hash of an AST expression.

    Args:
        ast: AST dictionary

    Returns:
        SHA256 hash string of the normalized AST
    """
    # Normalize before hashing to catch semantic equivalents
    normalized = normalize_ast_for_comparison(ast)

    # Convert to deterministic JSON string (sorted keys)
    ast_str = json.dumps(normalized, sort_keys=True, separators=(',', ':'))

    # Hash it
    return hashlib.sha256(ast_str.encode()).hexdigest()


def dedupe_factor_names(compiled_factors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate factor names to prevent duplicate columns in factor_matrix.

    Ensures each factor has a unique name by appending version suffixes when duplicates
    are detected. This prevents the broadcast error from xs() when extracting factors.

    Args:
        compiled_factors: List of compiled factor dictionaries with 'name' key

    Returns:
        List of compiled factors with unique names (modified in-place)
    """
    seen = set()
    for i, item in enumerate(compiled_factors):
        # Sanitize the name: replace non-word chars with underscore
        base = re.sub(r'\W+', '_', item["name"]).strip('_') or f"factor_{i+1}"
        name = base.lower()
        k = 2
        # If duplicate, append version suffix
        while name in seen:
            name = f"{base.lower()}__v{k}"
            k += 1
        item["name"] = name
        seen.add(name)
    return compiled_factors
