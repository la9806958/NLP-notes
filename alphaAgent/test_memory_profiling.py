#!/usr/bin/env python3
"""
Quick test to verify memory profiling functionality
"""

import sys
import psutil
import gc
import pandas as pd
import numpy as np

# Add current directory to path
sys.path.insert(0, '/home/lichenhui/data/alphaAgent')

from alpha_agent_backtest import MemoryProfiler


def test_memory_profiler():
    """Test the MemoryProfiler class"""

    print("Testing MemoryProfiler...")
    print("="*70)

    # Create profiler
    profiler = MemoryProfiler()
    profiler.start()

    # Simulate data loading
    print("\nSimulating data operations...")
    data1 = pd.DataFrame(np.random.randn(10000, 100))
    profiler.checkpoint("after_small_dataframe")

    # Simulate larger data
    data2 = pd.DataFrame(np.random.randn(100000, 50))
    profiler.checkpoint("after_medium_dataframe")

    # Simulate computation
    result = data1.T @ data1
    profiler.checkpoint("after_computation")

    # Clean up
    del data1
    profiler.checkpoint("after_delete_data1", force_gc=True)

    # Generate report
    profiler.report()
    profiler.get_object_sizes(locals(), top_n=5)

    print("\n" + "="*70)
    print("Memory profiling test completed successfully!")
    print("="*70)


if __name__ == "__main__":
    test_memory_profiler()
