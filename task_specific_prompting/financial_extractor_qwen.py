#!/usr/bin/env python3
"""
Financial Report Data Extractor - Qwen Edition
Uses Qwen 7B language model to extract structured data from financial report text files.
"""

import os
import glob
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import re
import json
from typing import Dict, List, Optional
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm
import time
import psutil
import gc
import subprocess

class FinancialExtractor:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        """Initialize the financial data extractor with a specified model."""
        print(f"Loading model: {model_name}")
        
        # Initialize GPU profiling
        self.gpu_stats = []
        self.profiling_enabled = torch.cuda.is_available()
        
        # Initialize CSV logging
        self.csv_output_path = None
        self.csv_lock = threading.Lock()
        self.csv_initialized = False
        
        # Initialize performance tracking
        self.current_model_name = model_name
        self.processing_times = []
        self.performance_lock = threading.Lock()
        
        if self.profiling_enabled:
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Use smaller model optimized for L4 GPU (24GB VRAM)
        # Qwen2.5-3B-Instruct is ideal for L4 - good performance with manageable memory usage
        model_name = "Qwen/Qwen2.5-3B-Instruct"
        
        try:
            # GPU profiling start
            self._start_gpu_profiling("model_loading")
            
            # Check available GPU memory
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"Available GPU memory: {gpu_memory_gb:.1f} GB")
                
                # Clear any existing GPU cache
                torch.cuda.empty_cache()
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True,
                cache_dir="./model_cache"
            )
            
            # Optimized configuration for L4 GPU (24GB VRAM) with 3B model
            device_map = "auto"
            load_in_8bit = False  # L4 has enough memory for fp16
            load_in_4bit = False
            
            # Configure based on available GPU memory
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"Detected GPU memory: {gpu_memory_gb:.1f} GB")
                
                if gpu_memory_gb >= 20:  # L4 or similar high-memory GPU
                    print(f"High-memory GPU detected, using fp16 without quantization")
                    load_in_8bit = False
                    load_in_4bit = False
                elif gpu_memory_gb >= 12:  # Mid-range GPU
                    print(f"Mid-range GPU, using 8-bit quantization for safety")
                    load_in_8bit = True
                    load_in_4bit = False
                else:  # Lower memory GPU
                    print(f"Low-memory GPU, using 4-bit quantization")
                    load_in_4bit = True
                    load_in_8bit = False
            
            try:
                # Optimized loading for different memory configurations
                if load_in_4bit:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name, 
                        trust_remote_code=True,
                        torch_dtype=torch.float16,
                        device_map=device_map,
                        low_cpu_mem_usage=True,
                        cache_dir="./model_cache",
                        quantization_config=quantization_config
                    )
                    print("✅ Loaded with 4-bit quantization")
                    
                elif load_in_8bit:
                    # 8-bit quantization for mid-range GPUs
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name, 
                        trust_remote_code=True,
                        torch_dtype=torch.float16,
                        device_map=device_map,
                        low_cpu_mem_usage=True,
                        cache_dir="./model_cache",
                        load_in_8bit=True,
                        llm_int8_enable_fp32_cpu_offload=True
                    )
                    print("✅ Loaded with 8-bit quantization")
                    
                else:
                    # Full precision fp16 for high-memory GPUs (L4)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name, 
                        trust_remote_code=True,
                        torch_dtype=torch.float16,
                        device_map=device_map,
                        low_cpu_mem_usage=True,
                        cache_dir="./model_cache",
                        attn_implementation="flash_attention_2",  # Use flash attention for efficiency
                    )
                    print("✅ Loaded with fp16 precision (optimal for L4)")
                    
            except Exception as memory_error:
                print(f"Memory error with initial config: {memory_error}")
                print("Trying with CPU offload and custom device mapping...")
                
                # Fallback: Force CPU offload with custom device map
                custom_device_map = {
                    "model.embed_tokens": "cpu",
                    "model.norm": "cpu",
                    "lm_head": "cpu",
                }
                # Put some layers on GPU, rest on CPU (Qwen2.5-3B has 36 layers: 0-35)
                for i in range(0, 16):  # First 16 layers on GPU
                    custom_device_map[f"model.layers.{i}"] = 0
                for i in range(16, 36):  # Rest on CPU (includes layers 32-35)
                    custom_device_map[f"model.layers.{i}"] = "cpu"
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map=custom_device_map,
                    low_cpu_mem_usage=True,
                    cache_dir="./model_cache",
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True
                )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.float16,
                batch_size=1,  # Single batch for stability
                max_length=8192  # 3B model can handle longer context efficiently
            )
            
            self._end_gpu_profiling("model_loading")
            
        except Exception as e:
            print(f"Error loading Qwen 7B {model_name}: {e}")
            print("Trying fallback models in order of preference...")
            
            # List of fallback models optimized for L4 GPU in order of preference
            fallback_models = [
                ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen2.5-1.5B-Instruct (Smaller Qwen)"),
                ("microsoft/DialoGPT-medium", "DialoGPT Medium (355M)"),
                ("distilgpt2", "DistilGPT-2 (82M)"),
                ("gpt2", "GPT-2 (124M)"),
                ("microsoft/DialoGPT-small", "DialoGPT Small (117M)")
            ]
            
            model_loaded = False
            for fallback_model, model_description in fallback_models:
                try:
                    print(f"Attempting to load {model_description} ({fallback_model})...")
                    self._start_gpu_profiling(f"fallback_model_loading_{fallback_model}")
                    
                    self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
                    
                    # Use conservative settings for fallback models
                    if torch.cuda.is_available():
                        self.model = AutoModelForCausalLM.from_pretrained(
                            fallback_model,
                            torch_dtype=torch.float16,
                            device_map="auto",
                            low_cpu_mem_usage=True
                        )
                    else:
                        # CPU-only fallback
                        self.model = AutoModelForCausalLM.from_pretrained(
                            fallback_model,
                            torch_dtype=torch.float32,
                            low_cpu_mem_usage=True
                        )
                    
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                        
                    self.generator = pipeline(
                        "text-generation",
                        model=self.model,
                        tokenizer=self.tokenizer,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        batch_size=1
                    )
                    
                    self._end_gpu_profiling(f"fallback_model_loading_{fallback_model}")
                    model_name = fallback_model
                    model_loaded = True
                    print(f"✅ Successfully loaded {model_description}")
                    break
                    
                except Exception as fallback_error:
                    print(f"❌ Failed to load {model_description}: {fallback_error}")
                    continue
            
            if not model_loaded:
                raise RuntimeError("Failed to load any model, including all fallbacks!")
        
        # Thread lock for GPU access
        self.gpu_lock = threading.Lock()
        print(f"✅ Model loaded successfully: {model_name}")
        
        # Clear GPU cache after model loading and show final memory usage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            self._log_gpu_status("post_model_loading")
            
            # Show model memory usage
            model_memory_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024**2
            print(f"📊 Model parameters memory: {model_memory_mb:.1f} MB")
            
            # Estimate memory needed for inference
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            allocated_gb = torch.cuda.memory_allocated() / 1024**3
            print(f"🔋 GPU Memory Status: {allocated_gb:.1f}GB / {gpu_memory_gb:.1f}GB used")
        
        # Store the final model name
        self.current_model_name = model_name
        
        # Display performance estimates
        self._display_performance_estimates()
            
    def _get_model_specs(self, model_name: str) -> Dict:
        """Get model specifications for performance estimation."""
        model_specs = {
            "Qwen/Qwen2.5-3B-Instruct": {
                "parameters": "3B",
                "size_gb": 6.0,
                "context_length": 32768,
                "base_latency_sec": 1.8,  # Faster than 7B model
                "tokens_per_sec": 40,     # Better generation speed
                "memory_intensive": False  # More manageable memory usage
            },
            "Qwen/Qwen2.5-7B-Instruct": {
                "parameters": "7B",
                "size_gb": 14.0,
                "context_length": 32768,
                "base_latency_sec": 3.5,  # Base processing time per file
                "tokens_per_sec": 25,      # Approximate generation speed
                "memory_intensive": True
            },
            "Qwen/Qwen2.5-1.5B-Instruct": {
                "parameters": "1.5B",
                "size_gb": 3.0,
                "context_length": 32768,
                "base_latency_sec": 1.2,  # Very fast
                "tokens_per_sec": 50,     # Good generation speed
                "memory_intensive": False
            },
            "microsoft/DialoGPT-medium": {
                "parameters": "355M", 
                "size_gb": 1.4,
                "context_length": 1024,
                "base_latency_sec": 0.8,
                "tokens_per_sec": 45,
                "memory_intensive": False
            },
            "distilgpt2": {
                "parameters": "82M",
                "size_gb": 0.3,
                "context_length": 1024, 
                "base_latency_sec": 0.4,
                "tokens_per_sec": 60,
                "memory_intensive": False
            },
            "gpt2": {
                "parameters": "124M",
                "size_gb": 0.5,
                "context_length": 1024,
                "base_latency_sec": 0.6,
                "tokens_per_sec": 50,
                "memory_intensive": False
            },
            "microsoft/DialoGPT-small": {
                "parameters": "117M",
                "size_gb": 0.5,
                "context_length": 1024,
                "base_latency_sec": 0.5,
                "tokens_per_sec": 55,
                "memory_intensive": False
            }
        }
        
        return model_specs.get(model_name, {
            "parameters": "Unknown",
            "size_gb": 1.0,
            "context_length": 1024,
            "base_latency_sec": 1.0,
            "tokens_per_sec": 30,
            "memory_intensive": False
        })
    
    def _estimate_processing_time(self, model_name: str, num_files: int) -> Dict:
        """Estimate processing time based on model and system specs."""
        specs = self._get_model_specs(model_name)
        
        # Base processing time per file
        base_time = specs["base_latency_sec"]
        
        # Adjust for quantization (makes it slower)
        if "load_in_8bit" in str(self.model.config) or hasattr(self.model, 'quantization_config'):
            base_time *= 1.4  # 8-bit quantization overhead
        
        if hasattr(self.model, 'quantization_config') and getattr(self.model.quantization_config, 'load_in_4bit', False):
            base_time *= 1.8  # 4-bit quantization overhead
        
        # Adjust for GPU vs CPU
        if torch.cuda.is_available():
            # Check if model is actually on GPU
            model_device = next(self.model.parameters()).device
            if model_device.type == 'cpu':
                base_time *= 3.5  # CPU is much slower
                gpu_factor = "CPU (no GPU acceleration)"
            else:
                gpu_factor = f"GPU ({torch.cuda.get_device_name(0)})"
        else:
            base_time *= 3.5  # CPU processing
            gpu_factor = "CPU only"
        
        # Adjust for system memory constraints
        if specs["memory_intensive"] and torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory_gb < 12:
                base_time *= 1.3  # Memory pressure slowdown
        
        # Estimate total time with parallelization
        total_sequential = base_time * num_files
        
        # Account for parallel processing (but with some overhead)
        parallel_efficiency = 0.75  # Not perfectly parallel due to GPU contention
        estimated_parallel = (total_sequential * parallel_efficiency) / 2  # Using 2 workers
        
        return {
            "base_time_per_file": base_time,
            "total_sequential": total_sequential,
            "estimated_parallel": estimated_parallel,
            "gpu_factor": gpu_factor,
            "specs": specs
        }
    
    def _display_performance_estimates(self):
        """Display performance estimates for the current model."""
        print(f"\n⚡ PERFORMANCE ESTIMATES for {self.current_model_name}")
        print(f"{'='*80}")
        
        specs = self._get_model_specs(self.current_model_name)
        
        # Model information
        print(f"📊 Model Specifications:")
        print(f"   Parameters: {specs['parameters']}")
        print(f"   Model Size: ~{specs['size_gb']:.1f} GB")
        print(f"   Context Length: {specs['context_length']:,} tokens")
        print(f"   Generation Speed: ~{specs['tokens_per_sec']} tokens/sec")
        
        # Device information
        if torch.cuda.is_available():
            model_device = next(self.model.parameters()).device
            device_info = f"GPU ({torch.cuda.get_device_name(0)})" if model_device.type != 'cpu' else "CPU"
        else:
            device_info = "CPU only"
        print(f"   Processing Device: {device_info}")
        
        # Time estimates for different file counts
        file_counts = [1, 10, 50, 100, 500]
        print(f"\n⏱️  Estimated Processing Times:")
        print(f"{'Files':<8} {'Per File':<12} {'Sequential':<12} {'Parallel (2x)':<15}")
        print(f"{'-'*8} {'-'*12} {'-'*12} {'-'*15}")
        
        for count in file_counts:
            estimates = self._estimate_processing_time(self.current_model_name, count)
            per_file = f"{estimates['base_time_per_file']:.1f}s"
            sequential = f"{estimates['total_sequential']/60:.1f}m" if estimates['total_sequential'] > 60 else f"{estimates['total_sequential']:.1f}s"
            parallel = f"{estimates['estimated_parallel']/60:.1f}m" if estimates['estimated_parallel'] > 60 else f"{estimates['estimated_parallel']:.1f}s"
            
            print(f"{count:<8} {per_file:<12} {sequential:<12} {parallel:<15}")
        
        print(f"\n💡 Notes:")
        print(f"   • Times include LLM inference + regex extraction + CSV logging")
        print(f"   • Parallel estimates assume 2 workers with 75% efficiency")
        print(f"   • Actual times may vary based on text complexity and system load")
        print(f"   • First file may take longer due to model warmup")
        print(f"{'='*80}")

    def _print_performance_summary(self, total_files: int):
        """Print actual vs estimated performance summary."""
        if not self.processing_times:
            return
            
        print(f"\n📊 PERFORMANCE SUMMARY")
        print(f"{'='*80}")
        
        # Calculate actual statistics
        avg_time = sum(self.processing_times) / len(self.processing_times)
        min_time = min(self.processing_times)
        max_time = max(self.processing_times)
        total_time = sum(self.processing_times)
        
        # Get estimated time for comparison
        estimates = self._estimate_processing_time(self.current_model_name, total_files)
        
        print(f"🎯 Actual Performance:")
        print(f"   Files processed: {len(self.processing_times)}")
        print(f"   Average time per file: {avg_time:.2f}s")
        print(f"   Fastest file: {min_time:.2f}s")
        print(f"   Slowest file: {max_time:.2f}s")
        print(f"   Total processing time: {total_time/60:.1f} minutes")
        
        print(f"\n📈 Estimated vs Actual:")
        print(f"   Estimated per file: {estimates['base_time_per_file']:.2f}s")
        print(f"   Actual per file: {avg_time:.2f}s")
        
        # Calculate accuracy
        if estimates['base_time_per_file'] > 0:
            accuracy = abs(avg_time - estimates['base_time_per_file']) / estimates['base_time_per_file'] * 100
            if avg_time < estimates['base_time_per_file']:
                performance = f"⚡ {accuracy:.1f}% faster than estimated"
            else:
                performance = f"🐌 {accuracy:.1f}% slower than estimated"
            print(f"   Performance: {performance}")
        
        # Throughput statistics  
        if total_time > 0:
            throughput = len(self.processing_times) / (total_time / 60)  # files per minute
            print(f"   Throughput: {throughput:.1f} files/minute")
        
        print(f"{'='*80}")

    def _start_gpu_profiling(self, operation: str):
        """Start GPU profiling for an operation."""
        if not self.profiling_enabled:
            return
            
        torch.cuda.synchronize()
        gpu_memory_before = torch.cuda.memory_allocated()
        gpu_memory_cached_before = torch.cuda.memory_reserved()
        cpu_percent_before = psutil.cpu_percent()
        
        profile_data = {
            'operation': operation,
            'start_time': time.time(),
            'gpu_memory_before_mb': gpu_memory_before / 1024**2,
            'gpu_memory_cached_before_mb': gpu_memory_cached_before / 1024**2,
            'cpu_percent_before': cpu_percent_before
        }
        
        self.gpu_stats.append(profile_data)
        print(f"\n[GPU PROFILE START] {operation}")
        print(f"  GPU Memory: {gpu_memory_before / 1024**2:.2f} MB allocated, {gpu_memory_cached_before / 1024**2:.2f} MB cached")
        print(f"  CPU Usage: {cpu_percent_before:.1f}%")
    
    def _end_gpu_profiling(self, operation: str):
        """End GPU profiling for an operation."""
        if not self.profiling_enabled:
            return
            
        torch.cuda.synchronize()
        
        # Find the matching start record
        for profile_data in reversed(self.gpu_stats):
            if profile_data['operation'] == operation and 'end_time' not in profile_data:
                end_time = time.time()
                gpu_memory_after = torch.cuda.memory_allocated()
                gpu_memory_cached_after = torch.cuda.memory_reserved()
                cpu_percent_after = psutil.cpu_percent()
                
                profile_data.update({
                    'end_time': end_time,
                    'duration_sec': end_time - profile_data['start_time'],
                    'gpu_memory_after_mb': gpu_memory_after / 1024**2,
                    'gpu_memory_cached_after_mb': gpu_memory_cached_after / 1024**2,
                    'gpu_memory_delta_mb': (gpu_memory_after - profile_data['gpu_memory_before_mb'] * 1024**2) / 1024**2,
                    'cpu_percent_after': cpu_percent_after
                })
                
                print(f"\n[GPU PROFILE END] {operation}")
                print(f"  Duration: {profile_data['duration_sec']:.3f} seconds")
                print(f"  GPU Memory Delta: {profile_data['gpu_memory_delta_mb']:+.2f} MB")
                print(f"  GPU Memory After: {gpu_memory_after / 1024**2:.2f} MB allocated, {gpu_memory_cached_after / 1024**2:.2f} MB cached")
                print(f"  CPU Usage After: {cpu_percent_after:.1f}%")
                break
                
    def _get_nvidia_smi_stats(self):
        """Get GPU stats using nvidia-smi command."""
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw,memory.used,memory.total',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                stats = result.stdout.strip().split(', ')
                return {
                    'gpu_util': int(stats[0]),
                    'mem_util': int(stats[1]),
                    'temperature': int(stats[2]),
                    'power_draw': float(stats[3]),
                    'memory_used': int(stats[4]),
                    'memory_total': int(stats[5])
                }
        except Exception as e:
            print(f"nvidia-smi error: {e}")
            return None
        return None

    def _log_gpu_status(self, context: str):
        """Log current GPU status."""
        if not self.profiling_enabled:
            return
            
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        cached = torch.cuda.memory_reserved()
        max_allocated = torch.cuda.max_memory_allocated()
        total_memory = torch.cuda.get_device_properties(0).total_memory
        
        print(f"\n[GPU STATUS] {context}")
        print(f"  PyTorch Memory:")
        print(f"    Allocated: {allocated / 1024**3:.2f} GB ({allocated / total_memory * 100:.1f}%)")
        print(f"    Cached: {cached / 1024**3:.2f} GB ({cached / total_memory * 100:.1f}%)")
        print(f"    Max Allocated: {max_allocated / 1024**3:.2f} GB")
        print(f"    Free Memory: {(total_memory - cached) / 1024**3:.2f} GB")
        
        # Try nvidia-smi first (more accurate)
        nvidia_stats = self._get_nvidia_smi_stats()
        if nvidia_stats:
            print(f"  nvidia-smi Stats:")
            print(f"    GPU Utilization: {nvidia_stats['gpu_util']}%")
            print(f"    Memory Utilization: {nvidia_stats['mem_util']}%")
            print(f"    Memory Used: {nvidia_stats['memory_used']} MB / {nvidia_stats['memory_total']} MB")
            print(f"    Temperature: {nvidia_stats['temperature']}°C")
            print(f"    Power Draw: {nvidia_stats['power_draw']:.1f}W")
        else:
            # Fallback to nvidia-ml-py3
            try:
                import nvidia_ml_py3 as nvml
                nvml.nvmlInit()
                handle = nvml.nvmlDeviceGetHandleByIndex(0)
                info = nvml.nvmlDeviceGetUtilizationRates(handle)
                temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                power = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
                print(f"  nvidia-ml-py3 Stats:")
                print(f"    GPU Utilization: {info.gpu}%")
                print(f"    Memory Utilization: {info.memory}%")
                print(f"    Temperature: {temp}°C")
                print(f"    Power Usage: {power:.1f}W")
            except ImportError:
                print(f"  GPU Utilization: nvidia monitoring libraries not available")
            except Exception as e:
                print(f"  GPU Utilization: Error - {e}")
            
    def print_profiling_summary(self):
        """Print a summary of all profiling data."""
        if not self.gpu_stats:
            print("No profiling data available")
            return
            
        print("\n" + "="*60)
        print("GPU PROFILING SUMMARY")
        print("="*60)
        
        for i, stats in enumerate(self.gpu_stats, 1):
            if 'duration_sec' in stats:
                print(f"\n{i}. Operation: {stats['operation']}")
                print(f"   Duration: {stats['duration_sec']:.3f}s")
                print(f"   GPU Memory Delta: {stats['gpu_memory_delta_mb']:+.2f} MB")
                print(f"   Memory Before: {stats['gpu_memory_before_mb']:.2f} MB")
                print(f"   Memory After: {stats['gpu_memory_after_mb']:.2f} MB")

    def sanitize_text(self, text: str) -> str:
        """Sanitize text to prevent CUDA tokenization errors."""
        # Remove or replace problematic characters
        import unicodedata
        
        # Normalize unicode
        text = unicodedata.normalize('NFKD', text)
        
        # Remove control characters except common ones
        control_chars = ''.join(chr(i) for i in range(32) if i not in [9, 10, 13])  # Keep tab, newline, carriage return
        text = text.translate(str.maketrans('', '', control_chars))
        
        # Replace problematic characters that might cause tokenization issues
        replacements = {
            '\u2028': ' ',  # Line separator
            '\u2029': ' ',  # Paragraph separator
            '\ufeff': '',   # Byte order mark
            '\u200b': '',   # Zero width space
            '\u200c': '',   # Zero width non-joiner
            '\u200d': '',   # Zero width joiner
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Ensure text is properly encoded
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        
        return text

    def initialize_csv(self, output_path: str):
        """Initialize CSV file with headers."""
        self.csv_output_path = output_path
        
        # Create CSV with headers if it doesn't exist
        if not os.path.exists(output_path):
            df_empty = pd.DataFrame(columns=[
                'source_file', 'ticker', 'analysts', 'date', 
                'price_target_mid', 'price_target_bull', 'price_target_bear',
                'eps_target_mid', 'eps_target_bull', 'eps_target_bear',
                'financial_estimates', 'processing_timestamp'
            ])
            df_empty.to_csv(output_path, index=False)
            print(f"✅ Initialized CSV file: {output_path}")
        else:
            print(f"📄 Using existing CSV file: {output_path}")
        
        self.csv_initialized = True

    def log_result_to_csv(self, result: Dict):
        """Immediately log a single result to CSV."""
        if not self.csv_initialized or not self.csv_output_path:
            print("❌ CSV not initialized, cannot log result")
            return
            
        # Add timestamp
        result['processing_timestamp'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Convert arrays and objects to JSON strings for CSV storage
        csv_result = result.copy()
        
        # Convert ticker array to JSON string
        if isinstance(csv_result.get('ticker'), list):
            csv_result['ticker'] = json.dumps(csv_result['ticker'])
        
        # Convert analysts array to JSON string
        if isinstance(csv_result.get('analysts'), list):
            csv_result['analysts'] = json.dumps(csv_result['analysts'])
            
        # Convert financial_estimates array to JSON string
        if isinstance(csv_result.get('financial_estimates'), list):
            csv_result['financial_estimates'] = json.dumps(csv_result['financial_estimates'])
        
        with self.csv_lock:
            try:
                # Create DataFrame from single result
                df_new = pd.DataFrame([csv_result])
                
                # Append to CSV
                df_new.to_csv(self.csv_output_path, mode='a', header=False, index=False)
                
                print(f"💾 LOGGED TO CSV: {result['source_file']}")
                print(f"   📊 Ticker: {result.get('ticker', [])}")
                print(f"   👤 Analysts: {result.get('analysts', [])}")
                print(f"   📅 Date: {result.get('date', 'N/A')}")
                print(f"   💰 Price Target: {result.get('price_target_mid', 'N/A')}")
                print(f"   📊 Financial Estimates: {len(result.get('financial_estimates', []))} items")
                print(f"   💾 Saved to: {self.csv_output_path}")
                
            except Exception as e:
                print(f"❌ Error logging to CSV: {e}")

    def create_extraction_prompt(self, text: str) -> str:
        """Create a structured prompt for Qwen data extraction."""
        # Sanitize the input text
        clean_text = self.sanitize_text(text)
        
        prompt = f"""<|im_start|>system
You are a financial analyst assistant specialized in extracting structured data from financial reports. Your task is to analyze the given text and extract specific financial information in JSON format.
<|im_end|>
<|im_start|>user
Analyze the following financial report text and extract the requested information. Respond ONLY with a valid JSON object containing the following fields:

Text to analyze:
{clean_text[:3000]}...

Extract these fields:
1. "ticker": Stock ticker symbol(s) mentioned in the report (array format)
2. "analysts": Name(s) of analyst(s) who wrote the report (array format)
3. "date": Date of the report (in YYYY-MM-DD format if possible)
4. "price_target_mid": Mid-case price target (numeric value or null)
5. "price_target_bull": Bull case price target (numeric value or null)
6. "price_target_bear": Bear case price target (numeric value or null)
7. "eps_target_mid": Mid-case EPS target (numeric value or null)
8. "eps_target_bull": Bull case EPS target (numeric value or null)
9. "eps_target_bear": Bear case EPS target (numeric value or null)
10. "financial_estimates": Array of objects with "item" and "estimate" fields for key financial statement line items

Example response format:
{{
  "ticker": ["SDR.L"],
  "analysts": ["Michael Werner", "Amit jagadeesh"],
  "date": "2019-12-23",
  "price_target_mid": 510,
  "price_target_bull": null,
  "price_target_bear": null,
  "eps_target_mid": null,
  "eps_target_bull": null,
  "eps_target_bear": null,
  "financial_estimates": [
    {{"item": "% Change YoY", "estimate": "-11%"}},
    {{"item": "Operating Profit", "estimate": "+18%"}},
    {{"item": "Revenue", "estimate": "+"}},
    {{"item": "Operating Costs", "estimate": "-"}},
    {{"item": "Year-end AuM's (ex JVs)", "estimate": ""}},
    {{"item": "Flows", "estimate": ""}},
    {{"item": "% Change HoH", "estimate": ""}},
    {{"item": "Free Float", "estimate": ""}}
  ]
}}

Respond with ONLY the JSON object. Start your response with {{ and end with }}.
<|im_end|>
<|im_start|>assistant
"""

        return prompt

    def extract_with_regex(self, text: str) -> Dict:
        """Fallback extraction using regex patterns."""
        result = {
            "ticker": [],
            "analysts": [],
            "date": "",
            "price_target_mid": None,
            "price_target_bull": None,
            "price_target_bear": None,
            "eps_target_mid": None,
            "eps_target_bull": None,
            "eps_target_bear": None,
            "financial_estimates": []
        }
        
        # Extract ticker symbols (common patterns) - collect all matches
        ticker_patterns = [
            r'\b([A-Z]{1,5})\s+(?:US|HK|LN|TT|CH|SS|SZ)',  # AAPL US, etc.
            r'\b([A-Z]{2,5})\s*\.\s*(?:US|HK|L|T|SS|SZ)',   # AAPL.US, etc.
            r'\(([A-Z]{1,5})\)',  # (AAPL)
            r'\b([A-Z]{1,5})\s+Inc',  # AAPL Inc
            r'\b([A-Z]{1,5})\s+Corp',  # AAPL Corp
            r'\b([A-Z]{1,5}\.[A-Z]{1,2})\b',  # SDR.L format
        ]
        
        tickers_found = set()
        for pattern in ticker_patterns:
            matches = re.findall(pattern, text[:1000])
            for match in matches:
                if len(match) >= 2:  # Filter out very short matches
                    tickers_found.add(match)
        
        result["ticker"] = list(tickers_found)
        
        # Extract dates
        date_patterns = [
            r'\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})',  # YYYY-MM-DD format (preferred)
            r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})',  # MM/DD/YYYY or DD/MM/YYYY
            r'\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4})',  # DD Month YYYY
            r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s+\d{4})',  # Month DD, YYYY
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text[:1000], re.IGNORECASE)
            if matches:
                result["date"] = matches[0]
                break
        
        # Extract analyst names - collect multiple analysts
        analyst_patterns = [
            r'(?:by|analyst|author)[:]\s*([A-Z][a-z]+\s+[A-Z][a-z]+)',  # by John Smith
            r'\b([A-Z][a-z]+\s+[A-Z][a-z]+),?\s+(?:analyst|research)',  # John Smith, analyst
            r'(?:contact|analyst)[:]\s*([A-Z][a-z]+\s+[A-Z][a-z]+)',  # contact: John Smith
            r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s*,\s*(?:CFA|analyst)',  # John Smith, CFA
            r'([a-z.]+@[a-z.]+\.com)',  # email patterns
        ]
        
        analysts_found = set()
        for pattern in analyst_patterns:
            matches = re.findall(pattern, text[:1500], re.IGNORECASE)
            for match in matches:
                if '@' in match:  # Email format - extract name part
                    name_part = match.split('@')[0].replace('.', ' ').title()
                    if len(name_part.split()) == 2:
                        analysts_found.add(name_part)
                else:
                    # Regular name format
                    if len(match.split()) == 2:  # Ensure it's first + last name
                        analysts_found.add(match)
        
        result["analysts"] = list(analysts_found)
        
        # Extract price targets (bull, bear, mid)
        price_target_patterns = [
            # Mid/base case targets
            (r'(?:price target|target price|PT)[:]\s*\$?(\d+(?:\.\d+)?)', 'price_target_mid'),
            (r'(?:base case|mid case)[:]\s*\$?(\d+(?:\.\d+)?)', 'price_target_mid'),
            # Bull case targets
            (r'(?:bull case|upside|optimistic)[:]\s*\$?(\d+(?:\.\d+)?)', 'price_target_bull'),
            # Bear case targets
            (r'(?:bear case|downside|pessimistic)[:]\s*\$?(\d+(?:\.\d+)?)', 'price_target_bear'),
        ]
        
        for pattern, target_type in price_target_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches and result[target_type] is None:
                try:
                    result[target_type] = float(matches[0])
                except ValueError:
                    continue
        
        # Extract EPS targets (bull, bear, mid)
        eps_target_patterns = [
            # Mid/base case EPS
            (r'(?:EPS|earnings per share)[:]\s*\$?(\d+(?:\.\d+)?)', 'eps_target_mid'),
            (r'(?:base case EPS|mid case EPS)[:]\s*\$?(\d+(?:\.\d+)?)', 'eps_target_mid'),
            # Bull case EPS
            (r'(?:bull case EPS|upside EPS)[:]\s*\$?(\d+(?:\.\d+)?)', 'eps_target_bull'),
            # Bear case EPS
            (r'(?:bear case EPS|downside EPS)[:]\s*\$?(\d+(?:\.\d+)?)', 'eps_target_bear'),
        ]
        
        for pattern, target_type in eps_target_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches and result[target_type] is None:
                try:
                    result[target_type] = float(matches[0])
                except ValueError:
                    continue
        
        # Extract financial estimates as structured data
        financial_estimate_patterns = [
            (r'(?:revenue|sales)[:]\s*([+\-]?\d*\.?\d*%?[+\-]*)', '% Change YoY'),
            (r'(?:operating profit|EBIT)[:]\s*([+\-]?\d*\.?\d*%?[+\-]*)', 'Operating Profit'),
            (r'(?:operating costs|expenses)[:]\s*([+\-]?\d*\.?\d*%?[+\-]*)', 'Operating Costs'),
            (r'(?:AuM|assets under management)[:]\s*([+\-]?\d*\.?\d*%?[+\-]*)', 'Year-end AuM\'s (ex JVs)'),
            (r'(?:flows|net flows)[:]\s*([+\-]?\d*\.?\d*%?[+\-]*)', 'Flows'),
            (r'(?:free float)[:]\s*([+\-]?\d*\.?\d*%?[+\-]*)', 'Free Float'),
        ]
        
        financial_estimates = []
        for pattern, item_name in financial_estimate_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                estimate_value = matches[0].strip()
                if estimate_value:
                    financial_estimates.append({
                        "item": item_name,
                        "estimate": estimate_value
                    })
        
        result["financial_estimates"] = financial_estimates
        
        return result

    def parse_llm_response(self, response: str) -> Dict:
        """Parse the LLM response and extract JSON data."""
        if not response or not response.strip():
            print("Warning: Empty response from LLM")
            return {
                "ticker": [],
                "analysts": [],
                "date": "",
                "price_target_mid": None,
                "price_target_bull": None,
                "price_target_bear": None,
                "eps_target_mid": None,
                "eps_target_bull": None,
                "eps_target_bear": None,
                "financial_estimates": []
            }
        
        try:
            # Try to find JSON in the response (more robust patterns)
            json_patterns = [
                r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested JSON
                r'\{.*?\}',  # Simple JSON
                r'```json\s*(\{.*?\})\s*```',  # JSON in code blocks
            ]
            
            for pattern in json_patterns:
                json_match = re.search(pattern, response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1) if len(json_match.groups()) > 0 else json_match.group()
                    return json.loads(json_str)
                    
        except Exception as e:
            print(f"JSON parsing error: {e}")
            print(f"Response was: {response[:200]}...")
        
        # If JSON parsing fails, return empty structure
        return {
            "ticker": [],
            "analysts": [],
            "date": "",
            "price_target_mid": None,
            "price_target_bull": None,
            "price_target_bear": None,
            "eps_target_mid": None,
            "eps_target_bull": None,
            "eps_target_bear": None,
            "financial_estimates": []
        }

    def extract_from_text(self, text: str, filename: str) -> Dict:
        """Extract financial data from a single text."""
        start_time = time.time()
        
        # First try regex extraction (more reliable)
        result = self.extract_with_regex(text)
        
        # Try to enhance with LLM if available
        try:
            prompt = self.create_extraction_prompt(text)
            
            # DEBUG: Print the actual prompt being sent
            print(f"\n[PROMPT DEBUG for {filename}]")
            print(f"Prompt length: {len(prompt)} characters")
            print(f"Full Prompt:\n{'='*60}")
            print(prompt)
            print(f"{'='*60}")
            
            # Validate prompt tokens before inference
            try:
                # Tokenize the prompt to check for issues
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
                print(f"Tokenized prompt shape: {input_ids.shape}")
                print(f"Max token ID in prompt: {input_ids.max().item()}")
                print(f"Tokenizer vocab size: {len(self.tokenizer)}")
                
                # Check for invalid token IDs
                if input_ids.max().item() >= len(self.tokenizer):
                    print(f"WARNING: Token ID {input_ids.max().item()} exceeds vocab size {len(self.tokenizer)}")
                    raise ValueError("Invalid token ID detected")
                    
            except Exception as token_error:
                print(f"Token validation failed: {token_error}")
                print("Skipping LLM inference due to tokenization issues")
                return result
            
            # Start profiling LLM inference
            self._start_gpu_profiling(f"llm_inference_{filename}")
            
            # Log GPU status before inference
            self._log_gpu_status(f"before_inference_{filename}")
            
            # Clear GPU cache and reset CUDA context if possible
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # Check for existing CUDA errors
                try:
                    torch.cuda.current_device()
                except RuntimeError as cuda_err:
                    print(f"Existing CUDA error detected: {cuda_err}")
                    return result
            
            # Thread-safe GPU access with additional error handling
            with self.gpu_lock:
                try:
                    # Optimized generation parameters for 3B model on L4 GPU
                    generated = self.generator(
                        prompt, 
                        max_new_tokens=600,  # Increased for better JSON generation
                        max_length=6000,     # Higher limit for 3B model
                        num_return_sequences=1,
                        temperature=0.2,     # Slightly lower for more consistent JSON
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        return_full_text=False,
                        eos_token_id=self.tokenizer.eos_token_id,
                        # Optimized parameters for structured output
                        early_stopping=True,
                        no_repeat_ngram_size=3,
                        repetition_penalty=1.05,
                        top_p=0.9,  # Nucleus sampling for better quality
                        top_k=50    # Limit vocabulary for more focused generation
                    )
                except RuntimeError as cuda_error:
                    print(f"CUDA error during generation: {cuda_error}")
                    # Try to recover CUDA context
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    raise cuda_error
            
            # Log GPU status after inference
            self._log_gpu_status(f"after_inference_{filename}")
            
            # End profiling
            self._end_gpu_profiling(f"llm_inference_{filename}")
            
            response = generated[0]['generated_text']
            
            # DEBUG: Print LLM output with better formatting
            print(f"\n{'='*80}")
            print(f"🤖 LLM RESPONSE for {filename}")
            print(f"{'='*80}")
            print(f"📊 Response Statistics:")
            print(f"   Length: {len(response)} characters")
            print(f"   Non-empty: {'✅' if response.strip() else '❌'}")
            print(f"\n📝 Raw LLM Response:")
            print(f"┌{'─'*78}┐")
            if response.strip():
                for line in response.split('\n'):
                    print(f"│ {line:<76} │")
            else:
                print(f"│ {'❌ EMPTY RESPONSE':<76} │")
            print(f"└{'─'*78}┘")
            
            llm_result = self.parse_llm_response(response)
            
            # DEBUG: Print parsed JSON result with better formatting
            print(f"\n🔍 EXTRACTED JSON for {filename}")
            print(f"{'─'*80}")
            print(json.dumps(llm_result, indent=2, ensure_ascii=False))
            print(f"{'─'*80}")
            
            # Merge results, preferring LLM results where available
            for key, value in llm_result.items():
                if value and str(value).strip():
                    result[key] = value
                    
            # Show final merged result
            print(f"\n📋 FINAL MERGED RESULT for {filename}")
            print(f"{'─'*80}")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            print(f"{'─'*80}")
                    
        except RuntimeError as cuda_error:
            print(f"CUDA error during LLM extraction for {filename}: {cuda_error}")
            
            # Try to recover from CUDA error
            if "device-side assert" in str(cuda_error) or "CUDA" in str(cuda_error):
                print("Attempting CUDA recovery...")
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        # Reset CUDA context
                        torch.cuda.init()
                        print("CUDA context reset successful")
                except Exception as recovery_error:
                    print(f"CUDA recovery failed: {recovery_error}")
            
            # End profiling even on error
            if hasattr(self, '_start_gpu_profiling'):
                try:
                    self._end_gpu_profiling(f"llm_inference_{filename}")
                except:
                    pass
                    
        except Exception as e:
            print(f"LLM extraction failed for {filename}: {e}")
            print(f"Error type: {type(e).__name__}")
            
            # End profiling even on error
            if hasattr(self, '_start_gpu_profiling'):
                try:
                    self._end_gpu_profiling(f"llm_inference_{filename}")
                except:
                    pass
        
        # Add filename for reference
        result["source_file"] = filename
        
        # Track processing time
        end_time = time.time()
        processing_time = end_time - start_time
        
        with self.performance_lock:
            self.processing_times.append(processing_time)
        
        # Immediately log to CSV
        self.log_result_to_csv(result.copy())  # Use copy to avoid modifying original
        
        # Show processing time for this file
        print(f"⏱️  Processing time: {processing_time:.2f}s for {filename}")
        
        return result

    def process_file_batch(self, file_paths: List[str]) -> List[Dict]:
        """Process a batch of files."""
        results = []
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                
                if len(text.strip()) < 100:  # Skip very short files
                    continue
                
                extracted_data = self.extract_from_text(text, os.path.basename(file_path))
                results.append(extracted_data)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        return results

    def process_directory(self, directory_path: str, output_path: str = "financial_data_extracted_qwen.csv", 
                         max_workers: int = 2, batch_size: int = 3):
        """Process all txt files in the directory and extract financial data using parallel processing."""
        txt_files = glob.glob(os.path.join(directory_path, "**/*.txt"), recursive=True)
        
        if not txt_files:
            print(f"No .txt files found in {directory_path}")
            return
        
        print(f"Found {len(txt_files)} text files to process")
        print(f"Using {max_workers} workers with batch size {batch_size} (Qwen 7B optimized)")
        
        # Initialize CSV for immediate logging
        self.initialize_csv(output_path)
        
        # Show processing estimates for the number of files found
        print(f"\n📈 PROCESSING ESTIMATES FOR {len(txt_files)} FILES:")
        estimates = self._estimate_processing_time(self.current_model_name, len(txt_files))
        print(f"   Estimated time per file: {estimates['base_time_per_file']:.1f}s")
        print(f"   Total estimated time (parallel): {estimates['estimated_parallel']/60:.1f} minutes")
        print(f"   Processing device: {estimates['gpu_factor']}")
        
        # Log initial GPU status
        self._log_gpu_status("processing_start")
        
        # Create batches of files
        file_batches = [txt_files[i:i + batch_size] for i in range(0, len(txt_files), batch_size)]
        
        results = []
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(self.process_file_batch, batch): batch 
                for batch in file_batches
            }
            
            # Collect results with progress bar
            with tqdm(total=len(txt_files), desc="Processing files") as pbar:
                for future in as_completed(future_to_batch):
                    batch = future_to_batch[future]
                    try:
                        batch_results = future.result()
                        results.extend(batch_results)
                        pbar.update(len(batch))
                    except Exception as e:
                        print(f"Error processing batch: {e}")
                        pbar.update(len(batch))
        
        if results:
            print(f"\n🎉 PROCESSING COMPLETE!")
            print(f"📊 Successfully processed {len(results)} files")
            print(f"💾 All results logged to: {output_path}")
            
            # Read the CSV to show final summary
            try:
                df = pd.read_csv(output_path)
                print(f"\n📋 FINAL CSV SUMMARY:")
                print(f"   Total records: {len(df)}")
                print(f"   Files with tickers: {df['ticker'].notna().sum()}")
                print(f"   Files with analysts: {df['analysts'].notna().sum()}")
                print(f"   Files with price targets: {df['price_target_mid'].notna().sum()}")
                
                # Display sample results from CSV
                print(f"\n📄 SAMPLE OF FINAL CSV DATA:")
                print(df[['source_file', 'ticker', 'analysts', 'date', 'price_target_mid']].head().to_string(index=False))
            except Exception as e:
                print(f"Error reading final CSV: {e}")
        else:
            print("❌ No data was successfully extracted")
            
        # Final GPU cleanup and profiling summary
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            self._log_gpu_status("processing_complete")
            
        # Print performance summary
        self._print_performance_summary(len(txt_files))
        
        # Print profiling summary
        self.print_profiling_summary()

def main():
    parser = argparse.ArgumentParser(description='Extract financial data from PDF text files using Qwen')
    parser.add_argument('--input_dir', '-i', default='/data/pdf_texts', 
                       help='Directory containing .txt files to process')
    parser.add_argument('--output', '-o', default='/home/lichenhui/results_qwen.csv',
                       help='Output CSV file path')
    parser.add_argument('--model', '-m', default='Qwen/Qwen2.5-3B-Instruct',
                       help='Hugging Face model to use for extraction (default: Qwen2.5-3B-Instruct, optimized for L4 GPU)')
    parser.add_argument('--workers', '-w', type=int, default=3,
                       help='Number of worker threads (default: 3, optimized for 3B model on L4 GPU)')
    parser.add_argument('--batch_size', '-b', type=int, default=4,
                       help='Batch size for processing files (default: 4, optimized for 3B model)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Input directory {args.input_dir} does not exist")
        return
    
    # Initialize extractor
    extractor = FinancialExtractor(model_name=args.model)
    
    # Process files with parallelization
    extractor.process_directory(args.input_dir, args.output, 
                               max_workers=args.workers, batch_size=args.batch_size)

if __name__ == "__main__":
    main()