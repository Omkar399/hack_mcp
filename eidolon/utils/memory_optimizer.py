"""
Memory Optimization Utilities for Eidolon
Handles RAM constraints and performance optimization for 8-16GB systems.
"""

import os
import gc
import psutil
import torch
from typing import Dict, Any, Optional
from ..utils.logging import get_component_logger


class MemoryOptimizer:
    """Optimizes memory usage for systems with 8-16GB RAM."""
    
    def __init__(self, max_ram_gb: float = 8.0):
        self.logger = get_component_logger("memory_optimizer")
        self.max_ram_gb = max_ram_gb
        self.max_ram_bytes = max_ram_gb * 1024 * 1024 * 1024
        
        # Get system info
        self.total_ram = psutil.virtual_memory().total
        self.total_ram_gb = self.total_ram / (1024**3)
        
        self.logger.info(f"Memory optimizer initialized for {max_ram_gb}GB limit")
        self.logger.info(f"System total RAM: {self.total_ram_gb:.1f}GB")
        
        # Configure based on available RAM
        self._configure_for_system()
    
    def _configure_for_system(self):
        """Configure settings based on system RAM."""
        if self.total_ram_gb <= 8:
            self.config = {
                "vision_model_size": "base",
                "batch_size": 1,
                "max_screenshots_memory": 50,
                "enable_model_unloading": True,
                "use_cpu_only": True,
                "max_concurrent_analysis": 1,
                "enable_gc_aggressive": True
            }
            self.logger.warning("Low RAM detected (≤8GB) - Using conservative settings")
        
        elif self.total_ram_gb <= 16:
            self.config = {
                "vision_model_size": "base", 
                "batch_size": 2,
                "max_screenshots_memory": 100,
                "enable_model_unloading": False,
                "use_cpu_only": False,
                "max_concurrent_analysis": 2,
                "enable_gc_aggressive": True
            }
            self.logger.info("Medium RAM detected (8-16GB) - Using balanced settings")
        
        else:
            self.config = {
                "vision_model_size": "large",
                "batch_size": 4,
                "max_screenshots_memory": 200,
                "enable_model_unloading": False,
                "use_cpu_only": False,
                "max_concurrent_analysis": 3,
                "enable_gc_aggressive": False
            }
            self.logger.info("High RAM detected (>16GB) - Using performance settings")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        memory = psutil.virtual_memory()
        process = psutil.Process()
        
        return {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "percent": memory.percent,
            "process_mb": process.memory_info().rss / (1024**2),
            "process_percent": process.memory_percent()
        }
    
    def check_memory_available(self, required_mb: float = 1000) -> bool:
        """Check if enough memory is available for operation."""
        available_mb = psutil.virtual_memory().available / (1024**2)
        return available_mb > required_mb
    
    def optimize_torch_settings(self):
        """Optimize PyTorch settings for memory efficiency."""
        if not torch.cuda.is_available():
            # Use MPS on macOS if available, otherwise CPU
            if torch.backends.mps.is_available() and not self.config["use_cpu_only"]:
                os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
                self.device = "mps"
                self.logger.info("Using MPS (Metal Performance Shaders) device")
            else:
                self.device = "cpu"
                self.logger.info("Using CPU device for PyTorch")
        else:
            self.device = "cuda"
            # Set CUDA memory fraction
            torch.cuda.set_per_process_memory_fraction(0.7)
            self.logger.info("Using CUDA device with 70% memory fraction")
        
        # Optimize memory allocation
        if self.config["enable_gc_aggressive"]:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        
        return self.device
    
    def cleanup_memory(self, force_gc: bool = False):
        """Clean up memory and run garbage collection."""
        if self.config["enable_gc_aggressive"] or force_gc:
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        
        memory = self.get_memory_usage()
        self.logger.debug(f"Memory after cleanup: {memory['used_gb']:.1f}GB used, "
                         f"{memory['available_gb']:.1f}GB available")
    
    def get_optimal_batch_size(self, base_size: int = 1) -> int:
        """Get optimal batch size based on available memory."""
        available_gb = psutil.virtual_memory().available / (1024**3)
        
        if available_gb < 2:
            return 1
        elif available_gb < 4:
            return min(base_size, 2)
        elif available_gb < 8:
            return min(base_size, 4)
        else:
            return base_size
    
    def monitor_memory_usage(self, operation: str = ""):
        """Monitor and log memory usage for an operation."""
        memory = self.get_memory_usage()
        
        if memory["percent"] > 85:
            self.logger.warning(f"High memory usage during {operation}: "
                              f"{memory['percent']:.1f}% ({memory['used_gb']:.1f}GB)")
            self.cleanup_memory(force_gc=True)
        
        elif memory["percent"] > 70:
            self.logger.info(f"Memory usage during {operation}: "
                           f"{memory['percent']:.1f}% ({memory['used_gb']:.1f}GB)")
        
        return memory
    
    def configure_environment(self):
        """Configure environment variables for memory optimization."""
        env_vars = {
            # PyTorch optimizations
            "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            
            # Transformers optimizations
            "TRANSFORMERS_OFFLINE": "1" if self.config["use_cpu_only"] else "0",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            
            # General optimizations
            "PYTHONHASHSEED": "0",
            "CUDA_LAUNCH_BLOCKING": "1" if self.total_ram_gb <= 8 else "0"
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
            
        self.logger.info("Environment configured for memory optimization")
    
    def get_model_config(self, model_type: str = "vision") -> Dict[str, Any]:
        """Get optimized model configuration."""
        base_config = {
            "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
            "device_map": None,
            "low_cpu_mem_usage": True,
        }
        
        if model_type == "vision":
            if self.config["vision_model_size"] == "base":
                base_config.update({
                    "model_name": "microsoft/Florence-2-base",
                    "max_length": 512,
                })
            else:
                base_config.update({
                    "model_name": "microsoft/Florence-2-large", 
                    "max_length": 1024,
                })
        
        return base_config
    
    def should_unload_model(self) -> bool:
        """Check if models should be unloaded to save memory."""
        memory = self.get_memory_usage()
        return (self.config["enable_model_unloading"] and 
                memory["percent"] > 70)


# Global memory optimizer instance
_memory_optimizer = None

def get_memory_optimizer(max_ram_gb: Optional[float] = None) -> MemoryOptimizer:
    """Get global memory optimizer instance."""
    global _memory_optimizer
    
    if _memory_optimizer is None:
        if max_ram_gb is None:
            # Auto-detect reasonable limit
            total_ram_gb = psutil.virtual_memory().total / (1024**3)
            if total_ram_gb <= 8:
                max_ram_gb = 6.0  # Leave 2GB for system
            elif total_ram_gb <= 16:
                max_ram_gb = 12.0  # Leave 4GB for system
            else:
                max_ram_gb = total_ram_gb * 0.8  # Use 80% of available RAM
        
        _memory_optimizer = MemoryOptimizer(max_ram_gb)
        _memory_optimizer.configure_environment()
    
    return _memory_optimizer