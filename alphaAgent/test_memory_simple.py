#!/usr/bin/env python3
"""
Quick standalone test of memory profiling code
"""

import sys
import psutil
import gc
import pandas as pd
import numpy as np


class MemoryProfiler:
    """Track memory usage throughout backtest execution"""

    def __init__(self):
        self.process = psutil.Process()
        self.checkpoints = []
        self.start_memory = None

    def start(self):
        """Record initial memory state"""
        gc.collect()  # Force garbage collection for accurate baseline
        mem_info = self.process.memory_info()
        self.start_memory = mem_info.rss / (1024 ** 3)  # Convert to GB
        self.checkpoints.append({
            'stage': 'start',
            'rss_gb': self.start_memory,
            'vms_gb': mem_info.vms / (1024 ** 3),
            'delta_gb': 0.0
        })
        print(f"Memory Profiling Started - Initial: {self.start_memory:.3f} GB")

    def checkpoint(self, stage: str, force_gc: bool = False):
        """Record memory at a specific stage"""
        if force_gc:
            gc.collect()

        mem_info = self.process.memory_info()
        rss_gb = mem_info.rss / (1024 ** 3)
        vms_gb = mem_info.vms / (1024 ** 3)
        delta_gb = rss_gb - self.start_memory

        self.checkpoints.append({
            'stage': stage,
            'rss_gb': rss_gb,
            'vms_gb': vms_gb,
            'delta_gb': delta_gb
        })

        print(f"  [{stage}] Memory: {rss_gb:.3f} GB (Δ {delta_gb:+.3f} GB)")

    def report(self) -> pd.DataFrame:
        """Generate memory usage report"""
        df = pd.DataFrame(self.checkpoints)
        print("\n" + "="*70)
        print("MEMORY USAGE REPORT")
        print("="*70)
        print(df.to_string(index=False))

        if len(self.checkpoints) > 1:
            peak_memory = max(cp['rss_gb'] for cp in self.checkpoints)
            total_increase = self.checkpoints[-1]['rss_gb'] - self.start_memory
            print(f"\nPeak Memory: {peak_memory:.3f} GB")
            print(f"Total Increase: {total_increase:+.3f} GB")

        return df

    def get_object_sizes(self, frame_locals: dict, top_n: int = 10):
        """Get sizes of top N largest objects in memory"""
        sizes = []
        for name, obj in frame_locals.items():
            try:
                size_mb = sys.getsizeof(obj) / (1024 ** 2)
                if isinstance(obj, pd.DataFrame):
                    size_mb = obj.memory_usage(deep=True).sum() / (1024 ** 2)
                elif isinstance(obj, np.ndarray):
                    size_mb = obj.nbytes / (1024 ** 2)
                sizes.append({'name': name, 'type': type(obj).__name__, 'size_mb': size_mb})
            except:
                pass

        df_sizes = pd.DataFrame(sizes).sort_values('size_mb', ascending=False).head(top_n)
        print("\n" + "="*70)
        print(f"TOP {top_n} LARGEST OBJECTS IN MEMORY")
        print("="*70)
        print(df_sizes.to_string(index=False))
        return df_sizes


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
