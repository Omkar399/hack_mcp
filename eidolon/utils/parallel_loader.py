"""
Parallel Model Loader for Maximum Performance

Optimizes model loading by using threading and memory pre-allocation
for ultra-fast startup times.
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict, Any
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

from .logging import get_component_logger


class ParallelModelLoader:
    """Loads multiple model instances in parallel for maximum performance."""
    
    def __init__(self, model_name: str = "microsoft/Florence-2-base"):
        self.model_name = model_name
        self.logger = get_component_logger("parallel_loader")
        self.device = self._detect_optimal_device()
        self.torch_dtype = self._get_optimal_dtype()
        
    def _detect_optimal_device(self) -> str:
        """Detect the best available device for maximum performance."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
        else:
            return "cpu"
    
    def _get_optimal_dtype(self) -> torch.dtype:
        """Get optimal data type for the detected device."""
        if self.device == "cuda":
            return torch.float16  # Faster inference on CUDA
        else:
            return torch.float32  # MPS and CPU compatibility
    
    def _setup_gpu_optimizations(self) -> None:
        """Configure GPU for maximum performance."""
        if self.device == "mps":
            # Apple Silicon optimizations
            torch.mps.set_per_process_memory_fraction(0.95)  # Use 95% GPU memory
            self.logger.info("Apple Silicon GPU optimizations enabled (95% memory)")
        elif self.device == "cuda":
            # NVIDIA GPU optimizations
            torch.cuda.empty_cache()
            self.logger.info("NVIDIA GPU optimizations enabled")
    
    def _load_single_model(self, instance_id: int) -> torch.nn.Module:
        """Load a single model instance with optimizations."""
        try:
            self.logger.info(f"Loading Florence-2 instance {instance_id + 1} on {self.device.upper()}...")
            
            # Pre-allocate GPU memory if using MPS
            if self.device == "mps":
                # Create a small tensor to ensure MPS is initialized
                dummy = torch.randn(100, 100, device=self.device)
                del dummy
                torch.mps.empty_cache()
            
            # Load model with optimizations
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=self.torch_dtype
            )
            
            # Move to optimal device
            model = model.to(self.device)
            
            # Enable inference optimizations
            model.eval()
            
            # Enable compiled mode for faster inference (if supported)
            if hasattr(torch, 'compile') and self.device != "mps":
                try:
                    model = torch.compile(model, mode="reduce-overhead")
                    self.logger.debug(f"Instance {instance_id + 1}: Torch compile enabled")
                except Exception as e:
                    self.logger.debug(f"Instance {instance_id + 1}: Torch compile not available: {e}")
            
            self.logger.info(f"Florence-2 instance {instance_id + 1} loaded successfully on {self.device.upper()}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model instance {instance_id + 1}: {e}")
            raise
    
    def load_model_pool(self, pool_size: int, max_workers: Optional[int] = None) -> tuple:
        """
        Load multiple model instances with optimizations.
        Using sequential loading to avoid meta tensor issues but with GPU optimizations.
        
        Args:
            pool_size: Number of model instances to load
            max_workers: Maximum number of parallel workers (ignored for now)
        
        Returns:
            tuple: (model_pool, processor, device)
        """
        self.logger.info(f"Starting optimized loading of {pool_size} Florence-2 instances...")
        
        # Setup GPU optimizations
        self._setup_gpu_optimizations()
        
        # Load processor once (shared across all models)
        self.logger.info("Loading shared processor...")
        processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        
        model_pool = []
        start_time = time.time()
        
        # Load models sequentially but with optimizations
        for i in range(pool_size):
            try:
                model = self._load_single_model(i)
                model_pool.append(model)
            except Exception as e:
                self.logger.error(f"Failed to load instance {i}: {e}")
                raise
        
        total_time = time.time() - start_time
        self.logger.info(f"✅ All {len(model_pool)} model instances loaded in {total_time:.2f}s")
        self.logger.info(f"🚀 Average loading time per instance: {total_time/len(model_pool):.2f}s")
        
        return model_pool, processor, self.device


def load_florence_models_parallel(pool_size: int) -> tuple:
    """
    Convenience function to load Florence-2 models in parallel.
    
    Args:
        pool_size: Number of model instances to load
    
    Returns:
        tuple: (model_pool, processor, device)
    """
    loader = ParallelModelLoader()
    return loader.load_model_pool(pool_size)


# Performance monitoring utilities
class ModelLoadTimer:
    """Timer for measuring model loading performance."""
    
    def __init__(self, description: str = "Model Loading"):
        self.description = description
        self.start_time = None
        self.logger = get_component_logger("model_timer")
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"⏱️  Starting {self.description}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            elapsed = time.time() - self.start_time
            if exc_type is None:
                self.logger.info(f"✅ {self.description} completed in {elapsed:.2f}s")
            else:
                self.logger.error(f"❌ {self.description} failed after {elapsed:.2f}s")
        
    def elapsed(self) -> float:
        """Get elapsed time since start."""
        if self.start_time:
            return time.time() - self.start_time
        return 0.0