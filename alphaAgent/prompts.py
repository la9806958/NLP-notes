#!/usr/bin/env python3
"""
Prompts Module

This module contains all prompt-building functions for the FactorAgent:
- Main factor generation prompts
- Single factor prompts
- Error recovery prompts
- Factor feedback generation
- Hypothesis improvement prompts

Extracted from alpha_agent_factor.py for better modularity.
"""

from typing import List, Dict
from ast_compiler import ALLOWED_FUNCS


def extract_operator_signatures() -> str:
    """Extract operator signatures from the operators file."""
    import inspect
    from alpha_agent_operators import Operators

    signatures = []
    for name in dir(Operators):
        if not name.startswith('_'):
            method = getattr(Operators, name)
            if callable(method):
                try:
                    sig = inspect.signature(method)
                    signatures.append(f"{name}{sig}")
                except:
                    signatures.append(f"{name}(...)")

    return "\n".join(signatures)


def build_main_prompt(hypothesis: str) -> str:
    """Build the main prompt for generating 25 factors from a hypothesis.

    Args:
        hypothesis: Market hypothesis string

    Returns:
        Formatted prompt string for LLM
    """
    # Read complete function signatures from the operators file
    operators_signatures = ""
    try:
        with open("/home/lichenhui/data/alphaAgent/alpha_agent_operators.py", "r") as f:
            operators_content = f.read()
            # Extract function signatures with docstrings
            import re

            # Find all function definitions with their signatures and docstrings
            pattern = r'def (\w+)\([^)]*\)[^:]*:\s*"""([^"]+)"""'
            matches = re.findall(pattern, operators_content, re.DOTALL)

            sig_lines = []
            for func_name, docstring in matches:
                # Get the full function signature
                func_pattern = rf'def ({func_name}\([^)]+\))[^:]*:'
                sig_match = re.search(func_pattern, operators_content)
                if sig_match:
                    signature = sig_match.group(1)
                    first_line = docstring.split('\\n')[0].strip()
                    sig_lines.append(f"{signature} - {first_line}")

            operators_signatures = "\\n".join(sig_lines[:25])  # First 25 functions

    except Exception as e:
        operators_signatures = f"Could not read operators file: {e}"

    return f"""
You are a quantitative researcher. You are using historical OHLC data of 1 min frequency to generate alpha factors for forward 30 min price prediction.
You can use window up to 300 minutes for time series functions.
Generate exactly 25 alpha factors from the hypothesis.

COMPLETE OPERATOR SIGNATURES:
{operators_signatures}

ALLOWED FUNCTION NAMES (use exactly): {sorted(ALLOWED_FUNCS.keys())}

CRITICAL FUNCTION SIGNATURES (MUST match exactly):
- hl_range(high, low, close_prev) - requires 3 args: high, low, and PREVIOUS close (use delay(close, 1))
- cond(mask, a, b) - requires EXACTLY 3 args: condition/mask, value_if_true, value_if_false
- true_range(high, low, close_prev) - requires 3 args
- atr(high, low, close, w) - requires 4 args: high, low, close, window
- corr_ts(x, y, w) - requires 3 args: two series and window
- cov_ts(x, y, w) - requires 3 args
- beta_ts(y, x, w) - requires 3 args

CRITICAL RULES:
1. Use ONLY the function names from ALLOWED_FUNCTION_NAMES above
2. Match the exact signatures shown above for argument count and types
3. NO "kwargs" field in AST expressions
4. Wrap all numbers as {{"type": "const", "value": NUMBER}}
5. DATA_COLUMNS: open, high, low, close, volume
6. For close_prev in hl_range/true_range, use: {{"type": "call", "func": "delay", "args": [{{"type": "symbol", "name": "close"}}, {{"type": "const", "value": 1}}]}}

AST EXAMPLES:
- Symbol: {{"type": "symbol", "name": "close"}}
- Constant: {{"type": "const", "value": 10}}
- Simple call: {{"type": "call", "func": "ts_mean", "args": [{{"type": "symbol", "name": "close"}}, {{"type": "const", "value": 20}}]}}
- Nested call: {{"type": "call", "func": "div", "args": [{{"type": "symbol", "name": "volume"}}, {{"type": "call", "func": "ts_mean", "args": [{{"type": "symbol", "name": "volume"}}, {{"type": "const", "value": 10}}]}}]}}
- hl_range usage: {{"type": "call", "func": "hl_range", "args": [{{"type": "symbol", "name": "high"}}, {{"type": "symbol", "name": "low"}}, {{"type": "call", "func": "delay", "args": [{{"type": "symbol", "name": "close"}}, {{"type": "const", "value": 1}}]}}]}}
- cond usage: {{"type": "call", "func": "cond", "args": [MASK_EXPR, TRUE_EXPR, FALSE_EXPR]}}

REQUIRED JSON FORMAT:
{{
  "factors": [
    {{
      "name": "factor_name_1",
      "description": "brief description",
      "reasoning": "why relevant to hypothesis",
      "expr": AST_EXPRESSION
    }},
    ... exactly 25 factors total ...
  ]
}}

HYPOTHESIS: {hypothesis}

Generate exactly 25 diverse factors focusing on volume, price momentum, volatility patterns.
Return ONLY valid JSON with complete expressions:"""


def build_single_factor_prompt(hypothesis: str, reasoning: str, factor_num: int, existing_factors: List) -> str:
    """Build prompt for generating a single factor.

    Args:
        hypothesis: Market hypothesis string
        reasoning: Reasoning for the hypothesis
        factor_num: Current factor number (1-25)
        existing_factors: List of FactorSpec objects already generated

    Returns:
        Formatted prompt string for LLM
    """
    # Get operator signatures
    operator_sigs = extract_operator_signatures()

    # List existing factor names AND ASTs to avoid duplicates
    existing_names = [spec.name for spec in existing_factors] if existing_factors else []

    # Build detailed existing factors string with ASTs
    if existing_factors:
        import json
        existing_factors_details = []
        for spec in existing_factors:
            # Format AST as compact JSON
            ast_str = json.dumps(spec.ast, separators=(',', ':'))
            existing_factors_details.append(f"  - {spec.name}: {ast_str}")
        existing_names_str = f"Existing factors ({len(existing_factors)}):\n" + "\n".join(existing_factors_details)
    else:
        existing_names_str = "No existing factors yet."

    # Include reasoning if provided
    reasoning_section = f"\n\nHYPOTHESIS REASONING: {reasoning}" if reasoning else ""

    prompt = f"""You are tasked with generating EXACTLY ONE factor expression for quantitative trading.

    You have available OHLCV data at 1 min frequency to predict forward 30 min returns.
    You can use window sizes up to 300 minutes for time series functions.

MARKET HYPOTHESIS: {hypothesis}{reasoning_section}

TASK: Generate factor #{factor_num} of 25 total factors.
{existing_names_str}

REQUIREMENTS:
1. Generate a factor that's unique and different from existing ones
2. Factor should relate to the market hypothesis
3. Use only the available operators and data fields listed below
4. Return VALID JSON in exactly this format:

{{
  "name": "descriptive_factor_name",
  "description": "what this factor measures",
  "reasoning": "why this factor is relevant to the hypothesis",
  "expr": {{AST_JSON_HERE}}
}}

AVAILABLE DATA FIELDS: open, high, low, close, volume

AVAILABLE OPERATORS:
{operator_sigs}

EXPRESSION FORMAT (AST JSON):
- Symbol: {{"type": "symbol", "name": "close"}}
- Constant: {{"type": "const", "value": 20}}
- Function call: {{"type": "call", "fn": "ts_mean", "args": [{{"type": "symbol", "name": "close"}}, {{"type": "const", "value": 20}}]}}

CRITICAL FUNCTION SIGNATURES (MUST match exactly):
- hl_range(high, low, close_prev) - 3 args required, use delay(close, 1) for close_prev
- cond(mask, a, b) - EXACTLY 3 args: mask, true_value, false_value
- true_range(high, low, close_prev) - 3 args required
- atr(high, low, close, w) - 4 args required
- corr_ts(x, y, w) / cov_ts(x, y, w) / beta_ts(y, x, w) - 3 args each

CRITICAL RULES FOR ARGS ARRAY:
- ALL arguments must be properly wrapped AST nodes
- NEVER put raw numbers or strings directly in args array
- WRONG: "args": [20, "up"]
- CORRECT: "args": [{{"type": "const", "value": 20}}]
- Match exact argument counts for each function

ADDITIONAL REQUIREMENTS:
- Return ONLY valid JSON, no additional text
- Ensure the AST uses proper JSON structure
- Factor name should be unique and descriptive
- Keep window sizes ≤ 300
- Only use operators from ALLOWED_FUNCS (no breakout, ret, cs_rescale, smooth_ts, fillna_ts)"""

    return prompt


def build_error_recovery_prompt(hypothesis: str, reasoning: str, factor_num: int, error_msg: str,
                                 failed_payload: dict = None) -> str:
    """Build error recovery prompt with specific feedback.

    Args:
        hypothesis: Market hypothesis string
        reasoning: Reasoning for the hypothesis
        factor_num: Current factor number (1-25)
        error_msg: Error message from failed attempt
        failed_payload: Failed JSON payload (optional)

    Returns:
        Formatted prompt string for LLM with error recovery guidance
    """
    operator_sigs = extract_operator_signatures()

    # Include reasoning if provided
    reasoning_section = f"\n\nHYPOTHESIS REASONING: {reasoning}" if reasoning else ""

    failed_json = ""
    if failed_payload:
        try:
            import json
            failed_json = f"FAILED ATTEMPT JSON:\n{json.dumps(failed_payload, indent=2)}\n\n"
        except:
            failed_json = f"FAILED ATTEMPT (unparseable): {failed_payload}\n\n"

    # Parse error message to provide specific guidance
    error_guidance = ""
    if "Unknown function" in error_msg:
        # Extract the unknown function name
        import re
        match = re.search(r"Unknown function '(\w+)'", error_msg)
        if match:
            unknown_func = match.group(1)
            error_guidance = f"""
❌ THE ERROR: You used function '{unknown_func}' which is NOT allowed.

✅ HOW TO FIX:
- Do NOT use '{unknown_func}' - it does not exist in the allowed operators
- Choose ONLY from the available operators listed below
- Common mistakes:
  * 'ret' → does not exist, use 'diff(close, 1)' for returns
  * 'cs_rescale' → does not exist, use 'ts_zscore' for normalization
  * 'breakout' → does not exist, use comparison operators like 'gt'
  * 'smooth_ts' → does not exist, use 'ts_mean' for smoothing
  * 'fillna_ts' → does not exist, data is auto-cleaned

REMEMBER: Check your function name '{unknown_func}' against the AVAILABLE OPERATORS list below!
"""
    elif "Invalid AST" in error_msg or "validation" in error_msg.lower():
        error_guidance = f"""
❌ THE ERROR: Your AST structure is invalid.

✅ HOW TO FIX:
- Every argument must be a proper AST node with "type" field
- Numbers must be wrapped: {{"type": "const", "value": 20}}
- Data fields must be wrapped: {{"type": "symbol", "name": "close"}}
- Function calls: {{"type": "call", "fn": "function_name", "args": [...]}}
- The "args" array must contain ONLY valid AST nodes, not raw values

REVIEW YOUR FAILED JSON BELOW AND FIX THE STRUCTURE!
"""
    elif "Duplicate AST" in error_msg:
        error_guidance = f"""
❌ THE ERROR: This factor expression is identical to a previously generated factor.

✅ HOW TO FIX:
- Use different operators
- Use different window sizes
- Combine operators in a different way
- Use different data fields (volume vs close, etc.)

CREATE A COMPLETELY DIFFERENT FACTOR!
"""
    else:
        error_guidance = f"""
❌ THE ERROR: {error_msg}

✅ HOW TO FIX:
- Read the error message carefully
- Check your JSON structure
- Verify operator names against the available operators list
- Ensure all arguments are properly wrapped AST nodes
"""

    prompt = f"""🔧 ERROR RECOVERY: Your previous attempt failed. Read the error carefully and fix it.

MARKET HYPOTHESIS: {hypothesis}{reasoning_section}
FACTOR NUMBER: {factor_num}/25

{error_guidance}

{failed_json}
════════════════════════════════════════════════════════════════

TASK: Generate a CORRECTED factor that FIXES the error above.

REQUIRED JSON FORMAT:
{{
  "name": "descriptive_factor_name",
  "description": "what this factor measures",
  "reasoning": "why this factor is relevant to the hypothesis",
  "expr": {{AST_JSON_HERE}}
}}

AVAILABLE DATA FIELDS (use ONLY these): open, high, low, close, volume

AVAILABLE OPERATORS (use ONLY these function names):
{operator_sigs}

AST STRUCTURE RULES:
✅ Symbol: {{"type": "symbol", "name": "close"}}
✅ Constant: {{"type": "const", "value": 20}}
✅ Function: {{"type": "call", "fn": "ts_mean", "args": [{{"type": "symbol", "name": "close"}}, {{"type": "const", "value": 20}}]}}

❌ NEVER use raw values in args: "args": [20] ← WRONG
✅ ALWAYS wrap in AST nodes: "args": [{{"type": "const", "value": 20}}] ← CORRECT

CRITICAL FUNCTION SIGNATURES:
- hl_range(high, low, close_prev) → 3 args, use delay(close, 1) for close_prev
- cond(mask, a, b) → EXACTLY 3 args: condition, true_value, false_value
- true_range(high, low, close_prev) → 3 args
- atr(high, low, close, w) → 4 args
- corr_ts(x, y, w) / cov_ts(x, y, w) / beta_ts(y, x, w) → 3 args each

⚠️  DOUBLE CHECK: Does your function name exist in the AVAILABLE OPERATORS list above?

Return ONLY valid JSON, no additional text."""

    return prompt


def generate_factor_feedback(factor_metrics: List[Dict], avg_correlation: float,
                             avg_sharpe: float) -> str:
    """Generate detailed feedback on factor performance.

    Args:
        factor_metrics: List of metrics for each factor
        avg_correlation: Average correlation across factors
        avg_sharpe: Average Sharpe ratio across factors

    Returns:
        Feedback string for LLM
    """
    feedback_parts = []

    feedback_parts.append(f"=== FACTOR-BY-FACTOR PERFORMANCE ===")
    feedback_parts.append(f"Average Correlation: {avg_correlation:.4f}")
    feedback_parts.append(f"Average Sharpe Ratio: {avg_sharpe:.4f}")
    feedback_parts.append("")

    # Sort factors by correlation
    sorted_by_corr = sorted(factor_metrics, key=lambda x: abs(x['correlation']), reverse=True)

    feedback_parts.append("Top 5 factors by correlation:")
    for i, m in enumerate(sorted_by_corr[:5]):
        feedback_parts.append(f"{i+1}. {m['name']}: corr={m['correlation']:.4f}, sharpe={m['sharpe']:.4f}")

    feedback_parts.append("")
    feedback_parts.append("Bottom 5 factors by correlation:")
    for i, m in enumerate(sorted_by_corr[-5:]):
        feedback_parts.append(f"{i+1}. {m['name']}: corr={m['correlation']:.4f}, sharpe={m['sharpe']:.4f}")

    # Sort factors by Sharpe
    sorted_by_sharpe = sorted(factor_metrics, key=lambda x: x['sharpe'], reverse=True)

    feedback_parts.append("")
    feedback_parts.append("Top 5 factors by Sharpe ratio:")
    for i, m in enumerate(sorted_by_sharpe[:5]):
        feedback_parts.append(f"{i+1}. {m['name']}: sharpe={m['sharpe']:.4f}, corr={m['correlation']:.4f}")

    feedback_parts.append("")
    feedback_parts.append("Bottom 5 factors by Sharpe ratio:")
    for i, m in enumerate(sorted_by_sharpe[-5:]):
        feedback_parts.append(f"{i+1}. {m['name']}: sharpe={m['sharpe']:.4f}, corr={m['correlation']:.4f}")

    return "\n".join(feedback_parts)


def build_analysis_prompt(best_factors_data: List[Dict], worst_factors_data: List[Dict]) -> str:
    """Build prompt for analyzing why best factors outperform worst factors.

    Args:
        best_factors_data: List of dicts with keys: name, ast, pearson_ic, sharpe
        worst_factors_data: List of dicts with keys: name, ast, pearson_ic, sharpe

    Returns:
        Formatted prompt string for LLM analysis
    """
    import json

    # Format best factors
    best_factors_str = []
    for factor in best_factors_data:
        best_factors_str.append(f"""
Factor Name: {factor['name']}
Pearson IC: {factor['pearson_ic']:.4f}
Sharpe Ratio: {factor['sharpe']:.4f}
AST Expression:
{json.dumps(factor['ast'], indent=2)}
""")

    # Format worst factors
    worst_factors_str = []
    for factor in worst_factors_data:
        worst_factors_str.append(f"""
Factor Name: {factor['name']}
Pearson IC: {factor['pearson_ic']:.4f}
Sharpe Ratio: {factor['sharpe']:.4f}
AST Expression:
{json.dumps(factor['ast'], indent=2)}
""")

    prompt = f"""You are analyzing the performance of alpha factors for quantitative trading.

You have been given two groups of factors:
1. TOP 10 BEST PERFORMING FACTORS (highest Pearson IC and Sharpe ratios)
2. BOTTOM 10 WORST PERFORMING FACTORS (lowest Pearson IC and Sharpe ratios)

Your task is to analyze WHY the best performing factors are successful while the worst performing factors fail.

═══════════════════════════════════════════════════════════════════
TOP 10 BEST PERFORMING FACTORS:
═══════════════════════════════════════════════════════════════════
{"".join(best_factors_str)}

═══════════════════════════════════════════════════════════════════
BOTTOM 10 WORST PERFORMING FACTORS:
═══════════════════════════════════════════════════════════════════
{"".join(worst_factors_str)}

═══════════════════════════════════════════════════════════════════

Please provide a detailed analysis covering:

1. **Operator Patterns**: What operators or operator combinations appear more frequently in successful factors vs unsuccessful ones?

2. **Window Sizes**: What window sizes or time horizons work better? Are there patterns in the lookback periods?

3. **Data Field Usage**: How do successful factors use OHLCV data differently? (e.g., volume-based vs price-based)

4. **Complexity**: Is there an optimal complexity level? Are successful factors simpler or more complex?

5. **Mathematical Structure**: What mathematical transformations or combinations lead to better performance? (e.g., ratios, differences, correlations, z-scores)

6. **Specific Insights**: What specific structural patterns in the AST expressions correlate with success?

Provide your analysis in clear, structured prose. Focus on actionable insights that can guide the construction of improved factors.
"""

    return prompt


def build_refinement_from_analysis_prompt(spec, analysis_response: str, best_factors_data: List[Dict],
                                          worst_factors_data: List[Dict]) -> str:
    """Build prompt for refining a factor using the analysis from step 1.

    Args:
        spec: FactorSpec to refine
        analysis_response: The LLM's analysis from step 1
        best_factors_data: List of dicts with best factor info
        worst_factors_data: List of dicts with worst factor info

    Returns:
        Formatted prompt string for LLM to construct new AST
    """
    import json

    # Get operator signatures
    operators_signatures = extract_operator_signatures()

    # Format current factor
    current_factor_str = f"""
Factor Name: {spec.name}
Description: {spec.description}
Reasoning: {spec.reasoning}
Current AST:
{json.dumps(spec.ast, indent=2)}
"""

    # Find metrics for current factor if it's in worst_factors_data
    current_metrics = ""
    current_sharpe = None
    for factor in worst_factors_data:
        if factor['name'] == spec.name:
            current_metrics = f"Pearson IC: {factor['pearson_ic']:.4f}, Sharpe: {factor['sharpe']:.4f}"
            current_sharpe = factor['sharpe']
            break

    # Add sign inversion note if Sharpe is very negative
    sign_inversion_note = ""
    if current_sharpe is not None and current_sharpe < -0.3:
        sign_inversion_note = f"""

⚠️  IMPORTANT NOTE ON SIGN INVERSION:
This factor has a NEGATIVE Sharpe ratio ({current_sharpe:.4f}), which suggests it may have predictive power
but with the WRONG SIGN. Consider whether the improved factor should:

1. **Invert the signal entirely**: Wrap the entire expression in neg() to flip the sign
   Example: {{"type": "call", "fn": "neg", "args": [ORIGINAL_EXPRESSION]}}

2. **Redesign with opposite logic**: If the current factor goes up when it should go down, redesign
   the logic to capture the inverse relationship

A strongly negative Sharpe often means the factor is capturing a real signal but needs sign inversion.
Evaluate whether simple inversion or redesign is more appropriate based on the factor's logic.
"""

    prompt = f"""You are now tasked with constructing an IMPROVED alpha factor based on the analysis you just completed.

═══════════════════════════════════════════════════════════════════
PREVIOUS ANALYSIS (YOUR INSIGHTS):
═══════════════════════════════════════════════════════════════════
{analysis_response}

═══════════════════════════════════════════════════════════════════
CURRENT FACTOR TO REFINE:
═══════════════════════════════════════════════════════════════════
{current_factor_str}
Current Performance: {current_metrics if current_metrics else "Poor performance"}

This factor performed poorly and needs to be improved based on your analysis above.{sign_inversion_note}

═══════════════════════════════════════════════════════════════════
AVAILABLE OPERATORS AND CONSTRAINTS:
═══════════════════════════════════════════════════════════════════

COMPLETE OPERATOR SIGNATURES:
{operators_signatures}

ALLOWED FUNCTION NAMES (use exactly): {sorted(ALLOWED_FUNCS.keys())}

AVAILABLE DATA FIELDS: open, high, low, close, volume
MAXIMUM WINDOW SIZE: 300 minutes

═══════════════════════════════════════════════════════════════════
YOUR TASK:
═══════════════════════════════════════════════════════════════════

Using the insights from your analysis above, construct a NEW and IMPROVED factor that:

1. **Applies the successful patterns** you identified in the best performing factors
2. **Avoids the failure patterns** you identified in the worst performing factors
3. **Maintains diversification** - don't just copy a top factor, create something orthogonal
4. **Uses appropriate operators and window sizes** based on your analysis
5. **Has a clear logical structure** that aligns with successful factor characteristics

Return a JSON object with this EXACT structure:
{{
  "name": "improved_factor_name_v2",
  "description": "clear description of what this factor measures",
  "reasoning": "explanation of how this factor applies insights from the analysis",
  "expr": {{AST_JSON_EXPRESSION}}
}}

CRITICAL AST STRUCTURE RULES:
- Symbol: {{"type": "symbol", "name": "close"}}
- Constant: {{"type": "const", "value": 20}}
- Function call: {{"type": "call", "fn": "function_name", "args": [...]}}
- ALL arguments must be properly wrapped AST nodes
- NEVER use raw numbers or strings in args array

Return ONLY valid JSON, no additional text.
"""

    return prompt


def build_refinement_prompt(spec, feedback: str, best_feedback: str, best_asts_string: str) -> str:
    """Build prompt for refining an underperforming factor.

    Args:
        spec: FactorSpec to refine
        feedback: Feedback on worst performing factors
        best_feedback: Feedback on best performing factors
        best_asts_string: String representation of best factor ASTs

    Returns:
        Formatted prompt string for LLM
    """
    import json

    # Get operator signatures
    operators_signatures = extract_operator_signatures()

    prompt = f"""You are refining an underperforming alpha factor.

WORST PERFORMING FACTORS:
{feedback}

BEST PERFORMING FACTORS (for inspiration):
{best_feedback}

BEST PERFORMING FACTORS - AST EXPRESSIONS:
{best_asts_string}

CURRENT FACTOR TO REFINE:
Name: {spec.name}
Description: {spec.description}
Reasoning: {spec.reasoning}
Current AST: {json.dumps(spec.ast, indent=2)}

This factor performed poorly with Pearson IC near zero or negative.

COMPLETE OPERATOR SIGNATURES:
{operators_signatures}

ALLOWED FUNCTION NAMES (use exactly): {sorted(ALLOWED_FUNCS.keys())}

Generate an IMPROVED version of this factor that:
1. Uses different operators or combinations than the current factor
2. Uses different window sizes
3. Learns from the best-performing factors' structure (see their AST expressions above)
4. Maintains the general intent but improves execution
5. CRITICAL: Generate a factor that is LESS CORRELATED with the best performing factors to provide diversification
   - Avoid directly copying the structure of best performing factors
   - Use different operator combinations and transformations
   - Focus on capturing orthogonal market signals (different time scales, different data combinations, etc.)

Return a JSON object with the same structure as before (name, description, reasoning, expr).
Make sure the factor name is slightly different to indicate it's refined (e.g., add "_v2" suffix).
"""

    return prompt


def build_hypothesis_improvement_prompt(hypothesis: str, feedback: str,
                                       avg_correlation: float, avg_sharpe: float) -> str:
    """Build prompt for improving hypothesis based on performance feedback.

    Args:
        hypothesis: Current market hypothesis
        feedback: Performance feedback text
        avg_correlation: Average correlation metric
        avg_sharpe: Average Sharpe ratio metric

    Returns:
        Formatted prompt string for LLM
    """
    improvement_prompt = f"""
The current market hypothesis was: "{hypothesis}"

The generated factors showed the following performance:
- Average Correlation with 30-min forward returns: {avg_correlation:.4f}
- Average Sharpe Ratio: {avg_sharpe:.4f}

Detailed factor-by-factor feedback:
{feedback}

AVAILABLE OPERATORS AND DATA CONSTRAINTS:

ALLOWED FUNCTION NAMES (use exactly): {sorted(ALLOWED_FUNCS.keys())}

CRITICAL FUNCTION SIGNATURES (MUST match exactly):
- hl_range(high, low, close_prev) - requires 3 args: high, low, and PREVIOUS close (use delay(close, 1))
- cond(mask, a, b) - requires EXACTLY 3 args: condition/mask, value_if_true, value_if_false
- true_range(high, low, close_prev) - requires 3 args
- atr(high, low, close, w) - requires 4 args: high, low, close, window
- corr_ts(x, y, w) - requires 3 args: two series and window
- cov_ts(x, y, w) - requires 3 args
- beta_ts(y, x, w) - requires 3 args

AVAILABLE DATA FIELDS: open, high, low, close, volume
MAXIMUM WINDOW SIZE: 300 minutes
DATA FREQUENCY: 1-minute OHLCV
FORWARD PREDICTION HORIZON: 30 minutes

Based on this feedback, provide an improved market hypothesis that:
1. Amplifies factors with high correlation and Sharpe ratio
2. Removes or modifies factors with poor performance
3. Adds new factors that might improve predictive power
4. Focuses on actionable market microstructure patterns in 1-minute OHLCV data
5. Uses ONLY the allowed operators listed above
6. Stays within the window size constraint of 300 minutes

Provide the improved hypothesis:"""

    return improvement_prompt
