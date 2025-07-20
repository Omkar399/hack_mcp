"""
Local AI Vision Models for Eidolon

Local vision model integration (Florence-2, CLIP, etc.)
Phase 3 implementation.
"""

import os
from ..utils.logging import get_component_logger


class LocalVisionModel:
    """Local vision model integration with Florence-2 and CLIP."""
    
    def __init__(self):
        self.logger = get_component_logger("models.local_vision")
        self.logger.info("Initializing local vision models")
        
        # Memory optimization
        from ..utils.memory_optimizer import get_memory_optimizer
        self.memory_optimizer = get_memory_optimizer()
        
        # Check if models are available
        self.is_available = self._check_model_availability()
        self.model = None
        self.processor = None
        
    def _check_model_availability(self):
        """Check if required dependencies are available for vision models."""
        try:
            import torch
            import transformers
            from transformers import AutoProcessor, AutoModelForCausalLM
            return True
        except ImportError:
            self.logger.warning("Vision model dependencies not available")
            return False
    
    def load_model(self, model_name=None):
        """Load a vision model for inference with memory optimization."""
        if not self.is_available:
            self.logger.error("Cannot load model - dependencies not available")
            return False
        
        # Check memory before loading
        if not self.memory_optimizer.check_memory_available(2000):  # Need 2GB
            self.logger.warning("Insufficient memory to load vision model")
            return False
            
        try:
            # Get optimized model config
            model_config = self.memory_optimizer.get_model_config("vision")
            if model_name is None:
                model_name = model_config["model_name"]
            
            self.logger.info(f"Loading vision model: {model_name}")
            
            # Import here to avoid memory usage if not needed
            import torch
            from transformers import AutoProcessor, AutoModelForCausalLM
            
            # Configure device
            device = self.memory_optimizer.optimize_torch_settings()
            
            # Load with memory optimization
            self.processor = AutoProcessor.from_pretrained(
                model_name, 
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=model_config["torch_dtype"],
                low_cpu_mem_usage=True,
                device_map=device if device != "cpu" else None
            )
            
            if device != "cpu":
                self.model = self.model.to(device)
            
            self.model.eval()  # Set to evaluation mode
            
            # Monitor memory after loading
            self.memory_optimizer.monitor_memory_usage("model_loading")
            
            self.logger.info(f"Successfully loaded vision model on {device}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            self.memory_optimizer.cleanup_memory(force_gc=True)
            return False
    
    def analyze_image(self, image_path: str, task: str = "<MORE_DETAILED_CAPTION>"):
        """Analyze an image using the loaded model."""
        if self.model is None or self.processor is None:
            if not self.load_model():
                return {"error": "Failed to load model"}
        
        try:
            from PIL import Image
            import torch
            
            # Load and process image
            image = Image.open(image_path).convert("RGB")
            
            # Monitor memory usage
            self.memory_optimizer.monitor_memory_usage("image_analysis")
            
            # Process with memory optimization
            with torch.no_grad():
                inputs = self.processor(
                    text=task,
                    images=image,
                    return_tensors="pt"
                )
                
                # Move to device if needed
                if hasattr(self.memory_optimizer, 'device') and self.memory_optimizer.device != "cpu":
                    inputs = {k: v.to(self.memory_optimizer.device) for k, v in inputs.items()}
                
                # Generate with memory constraints
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=256,
                    num_beams=1,  # Reduce for memory
                    do_sample=False
                )
                
                # Decode response
                generated_text = self.processor.batch_decode(
                    generated_ids, 
                    skip_special_tokens=False
                )[0]
                
                # Clean up the response
                parsed_answer = self.processor.post_process_generation(
                    generated_text, 
                    task=task, 
                    image_size=(image.width, image.height)
                )
            
            # Cleanup after analysis
            if self.memory_optimizer.should_unload_model():
                self.unload_model()
            else:
                self.memory_optimizer.cleanup_memory()
            
            return {
                "task": task,
                "result": parsed_answer,
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing image: {e}")
            self.memory_optimizer.cleanup_memory(force_gc=True)
            return {"error": str(e), "success": False}
    
    def unload_model(self):
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        self.memory_optimizer.cleanup_memory(force_gc=True)
        self.logger.info("Vision model unloaded")


# Alias for backward compatibility
VisionAnalyzer = LocalVisionModel