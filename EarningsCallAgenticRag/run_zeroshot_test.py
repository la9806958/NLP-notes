#!/usr/bin/env python3
"""
Test script to run zero-shot GPT classifier on a small sample
"""

import sys
sys.path.append('baseline')

from zeroshot_earnings_classifier import ZeroShotEarningsClassifier

def run_test_sample():
    """Run zero-shot on just 5 samples for testing"""

    print("🧠 Testing Zero-Shot GPT Classifier")
    print("=" * 50)

    # Initialize classifier
    classifier = ZeroShotEarningsClassifier('future_3bday_cum_return')

    # Process just 5 samples for testing
    classifier.process_earnings_calls(max_samples=5)

    # Evaluate performance
    results = classifier.evaluate_performance()

    if results:
        print(f"\nTest Results:")
        print(f"Samples: {results['n_samples']}")
        print(f"F1 Score: {results['f1_score']:.4f}")
        print(f"MCC: {results['mcc']:.4f}")
        print(f"\n✅ Test completed! Check ./zeroshot_results/ for detailed results.")

if __name__ == "__main__":
    run_test_sample()