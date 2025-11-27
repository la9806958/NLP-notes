#!/usr/bin/env python3
"""
Test script to verify the updated hypothesis improvement prompt includes allowed expressions
"""

from prompts import build_hypothesis_improvement_prompt

# Sample data
test_hypothesis = "Volume spikes after price increases predict future returns"
test_feedback = """=== FACTOR-BY-FACTOR PERFORMANCE ===
Average Correlation: 0.0005
Average Sharpe Ratio: 0.5

Top 5 factors by correlation:
1. volume_price_spike_persistence: corr=0.0021, sharpe=4.69
2. volume_price_spike_ratio: corr=0.0020, sharpe=4.50
3. volume_price_volatility_correlation: corr=0.0032, sharpe=4.52
"""

avg_corr = 0.0005
avg_sharpe = 0.5

# Generate the prompt
prompt = build_hypothesis_improvement_prompt(test_hypothesis, test_feedback, avg_corr, avg_sharpe)

print("="*80)
print("HYPOTHESIS IMPROVEMENT PROMPT TEST")
print("="*80)
print("\nGenerated prompt:\n")
print(prompt)
print("\n" + "="*80)

# Verify key components are present
checks = {
    "ALLOWED_FUNCS list": "ALLOWED FUNCTION NAMES" in prompt,
    "Critical signatures": "hl_range(high, low, close_prev)" in prompt,
    "Data fields": "AVAILABLE DATA FIELDS" in prompt,
    "Window constraint": "300 minutes" in prompt,
    "Operator constraint": "Uses ONLY the allowed operators" in prompt,
}

print("\n✅ VERIFICATION CHECKS:")
for check_name, passed in checks.items():
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}: {check_name}")

all_passed = all(checks.values())
if all_passed:
    print("\n✅ All checks passed! The prompt now includes the allowed expressions list.")
else:
    print("\n❌ Some checks failed. Please review the prompt structure.")
