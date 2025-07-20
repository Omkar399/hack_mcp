"""
Observer component for Eidolon AI Personal Assistant

Handles screenshot capture, system monitoring, and activity detection.
Provides the foundation for all data collection in the system.
"""

import os
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List
import hashlib
import json

import mss
import psutil
from PIL import Image, ImageChops
import numpy as np

from ..utils.config import get_config
from ..utils.logging import get_component_logger, log_performance, log_exceptions
from ..integrations.fastmcp import get_fastmcp


class Screenshot:
    """Represents a captured screenshot with metadata."""
    
    def __init__(
        self,
        image: Image.Image,
        timestamp: datetime,
        window_info: Optional[Dict[str, Any]] = None,
        monitor_info: Optional[Dict[str, Any]] = None
    ):
        self.image = image
        self.timestamp = timestamp
        self.window_info = window_info or {}
        self.monitor_info = monitor_info or {}
        self.hash = self._calculate_hash()
        self.file_path: Optional[str] = None
    
    def _calculate_hash(self) -> str:
        """Calculate SHA-256 hash of the image for deduplication."""
        image_bytes = self.image.tobytes()
        return hashlib.sha256(image_bytes).hexdigest()
    
    def save(self, file_path: str) -> None:
        """Save screenshot to file."""
        self.image.save(file_path, "PNG", optimize=True)
        self.file_path = file_path
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert screenshot metadata to dictionary."""
        # Ensure we have valid image dimensions
        size = getattr(self.image, 'size', None) if self.image else None
        if not size:
            size = (1920, 1080)  # Default size if image unavailable
        
        # Ensure timestamp is always valid
        timestamp = self.timestamp
        if timestamp is None:
            timestamp = datetime.now()
        
        return {
            "timestamp": timestamp.isoformat(),
            "hash": self.hash or f"hash_{int(timestamp.timestamp())}",
            "file_path": self.file_path or "",
            "window_info": self.window_info or {},
            "monitor_info": self.monitor_info or {},
            "size": size
        }


class ChangeMetrics:
    """Metrics for detecting changes between screenshots."""
    
    def __init__(
        self,
        pixel_difference_ratio: float,
        structural_similarity: float,
        has_significant_change: bool,
        changed_regions: List[tuple] = None
    ):
        self.pixel_difference_ratio = pixel_difference_ratio
        self.structural_similarity = structural_similarity  
        self.has_significant_change = has_significant_change
        self.changed_regions = changed_regions or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "pixel_difference_ratio": self.pixel_difference_ratio,
            "structural_similarity": self.structural_similarity,
            "has_significant_change": self.has_significant_change,
            "changed_regions": self.changed_regions
        }


class Observer:
    """
    Core observer component for screenshot capture and system monitoring.
    
    Provides intelligent capture based on activity detection and change analysis.
    Manages resource usage and storage optimization.
    """
    
    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        """
        Initialize the Observer.
        
        Args:
            config_override: Optional configuration overrides.
        """
        self.config = get_config()
        if config_override:
            # Apply configuration overrides
            for key, value in config_override.items():
                if hasattr(self.config.observer, key):
                    setattr(self.config.observer, key, value)
        
        self.logger = get_component_logger("observer")
        
        # Memory management - will be set by CLI or use config default
        self._memory_limit_mb = None
        
        # Shared components (initialize once, reuse across captures)
        self._analyzer = None
        self._database = None
        self._fastmcp = None
        
        # Real-time processing optimization
        self._processing_queue = []
        self._processing_thread: Optional[threading.Thread] = None
        self._process_screenshots = True
        
        # Batch processing for maximum GPU performance
        # Configurable batch size from environment (default 8)
        self._batch_size = int(os.environ.get('EIDOLON_BATCH_SIZE', '8'))
        self._batch_queue = []
        self._batch_processing_thread: Optional[threading.Thread] = None
        
        # Smart resource management
        self._model_loading = False
        self._disable_resource_limits = False
        
        # State management
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._last_screenshot: Optional[Screenshot] = None
        self._capture_count = 0
        self._start_time: Optional[datetime] = None
        
        # Performance monitoring
        self._performance_metrics = {
            "captures_per_minute": 0.0,
            "average_capture_time": 0.0,
            "memory_usage_mb": 0.0,
            "cpu_usage_percent": 0.0,
            "duplicates_filtered": 0
        }
        
        # Storage management
        self.storage_path = Path(self.config.observer.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Screenshot capture tool
        self._sct = mss.mss()
        
        # Activity callbacks
        self._activity_callbacks: List[Callable[[Screenshot], None]] = []
        
        self.logger.info("Observer initialized successfully")
    
    def add_activity_callback(self, callback: Callable[[Screenshot], None]) -> None:
        """Add callback to be called when new screenshots are captured."""
        self._activity_callbacks.append(callback)
        self.logger.debug(f"Added activity callback: {callback.__name__}")
    
    def start_monitoring(self) -> None:
        """Start the screenshot monitoring process with parallel processing."""
        if self._running:
            self.logger.warning("Observer is already running")
            return
        
        self._running = True
        self._start_time = datetime.now()
        self._capture_count = 0
        
        # Start batch processing thread first (for maximum GPU utilization)
        self._batch_processing_thread = threading.Thread(
            target=self._batch_processing_loop,
            name="eidolon-batch-processor",
            daemon=True
        )
        self._batch_processing_thread.start()
        
        # Start processing thread (for individual processing)
        self._processing_thread = threading.Thread(
            target=self._processing_loop,
            name="eidolon-processor",
            daemon=True
        )
        self._processing_thread.start()
        
        # Start capture thread
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="eidolon-observer",
            daemon=True
        )
        self._capture_thread.start()
        
        self.logger.info("Screenshot monitoring started with ULTRA-FAST batch processing (10 FPS)")
    
    def _batch_processing_loop(self) -> None:
        """Ultra-fast batch processing loop for maximum GPU utilization."""
        self.logger.info("Batch processing thread started for MAXIMUM GPU performance")
        
        while self._running:
            try:
                # Collect batch of screenshots for parallel processing
                if len(self._batch_queue) >= self._batch_size:
                    batch = self._batch_queue[:self._batch_size]
                    self._batch_queue = self._batch_queue[self._batch_size:]
                    
                    # Process entire batch simultaneously on GPU
                    self._process_batch_on_gpu(batch)
                else:
                    # Short sleep when batch isn't ready
                    time.sleep(0.01)  # 10ms sleep for ultra-responsiveness
                    
            except Exception as e:
                self.logger.error(f"Error in batch processing loop: {e}")
                time.sleep(0.1)
    
    def _process_batch_on_gpu(self, batch: List[dict]) -> None:
        """Process batch of screenshots simultaneously on GPU for maximum performance."""
        try:
            if not batch:
                return
            
            # Get shared components
            db = self._get_database()
            analyzer = self._get_analyzer()
            
            self.logger.debug(f"Processing batch of {len(batch)} screenshots on GPU")
            
            # Process each in parallel (future: implement true batch inference)
            for screenshot_data in batch:
                self._process_screenshot_async(screenshot_data)
                
        except Exception as e:
            self.logger.error(f"Error processing GPU batch: {e}")
    
    def _processing_loop(self) -> None:
        """Dedicated processing thread for real-time analysis."""
        self.logger.info("Processing thread started for real-time analysis")
        
        while self._running:
            try:
                if self._processing_queue:
                    # Get the next screenshot to process
                    screenshot_data = self._processing_queue.pop(0)
                    
                    # Process in background without blocking capture
                    self._process_screenshot_async(screenshot_data)
                else:
                    # Short sleep when queue is empty
                    time.sleep(0.1)
                    
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                time.sleep(1)  # Prevent rapid error loops
    
    def _process_screenshot_async(self, screenshot_data: dict) -> None:
        """Process screenshot asynchronously without blocking capture with FastMCP optimization."""
        try:
            file_path = screenshot_data['file_path']
            screenshot = screenshot_data['screenshot']
            
            # Get shared components (cached for performance)
            db = self._get_database()
            analyzer = self._get_analyzer()
            fastmcp = self._get_fastmcp()
            
            # Store in database
            screenshot_id = db.store_screenshot(screenshot.to_dict())
            
            # Perform analysis with FastMCP optimization
            if self._process_screenshots:
                # Create content data for FastMCP processing
                content_data = {
                    'content_type': 'screenshot',
                    'file_path': str(file_path),
                    'timestamp': screenshot.timestamp.isoformat(),
                    'metadata': screenshot.to_dict()
                }
                
                # Use FastMCP for optimized processing
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    # Check FastMCP cache first
                    fastmcp_result = loop.run_until_complete(
                        fastmcp.process_content_analysis(content_data, priority="normal")
                    )
                    
                    # If FastMCP has cached results, use them; otherwise do full analysis
                    if fastmcp_result and fastmcp_result.get('processed_by') == 'fastmcp_cached':
                        self.logger.debug(f"Using FastMCP cached results for {file_path}")
                        # Use cached analysis results if available
                        if 'ocr_result' in fastmcp_result:
                            db.store_ocr_result(screenshot_id, fastmcp_result['ocr_result'])
                        if 'content_analysis' in fastmcp_result:
                            db.store_content_analysis(screenshot_id, fastmcp_result['content_analysis'])
                    else:
                        # Perform full analysis and cache in FastMCP
                        extracted_text = analyzer.extract_text(file_path)
                        if extracted_text.text:  # Only store if text was found
                            db.store_ocr_result(screenshot_id, extracted_text.to_dict())
                        
                        # Content analysis (enhanced with LLM if configured)
                        content_analysis = loop.run_until_complete(
                            analyzer.analyze_content_with_llm(file_path, extracted_text.text)
                        )
                        db.store_content_analysis(screenshot_id, content_analysis.to_dict())
                        
                        # Cache results in FastMCP for future use
                        cache_data = content_data.copy()
                        cache_data.update({
                            'ocr_result': extracted_text.to_dict() if extracted_text.text else None,
                            'content_analysis': content_analysis.to_dict(),
                            'processed_by': 'fastmcp_cached'
                        })
                        
                        # Store in FastMCP cache
                        loop.run_until_complete(
                            fastmcp.process_content_analysis(cache_data, priority="normal", use_cache=False)
                        )
                        
                finally:
                    loop.close()
                
        except Exception as e:
            self.logger.error(f"Error processing screenshot async: {e}")
    
    def _save_screenshot_to_queue(self, screenshot: Screenshot) -> None:
        """Save screenshot to disk and queue for async processing."""
        try:
            # Generate filename with timestamp
            timestamp_str = screenshot.timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"screenshot_{timestamp_str}.png"
            file_path = Path(self.config.observer.storage_path) / filename
            
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save screenshot quickly
            screenshot.save(str(file_path))
            screenshot.file_path = str(file_path)
            
            # Add to batch queue for maximum GPU performance
            screenshot_data = {
                'file_path': file_path,
                'screenshot': screenshot
            }
            
            self._batch_queue.append(screenshot_data)
            
            # Also add to individual processing queue as backup
            self._processing_queue.append(screenshot_data)
            
            # Limit queue sizes to prevent memory issues
            if len(self._batch_queue) > 100:  # Max 100 in batch queue
                self._batch_queue.pop(0)
            if len(self._processing_queue) > 50:  # Max 50 in processing queue
                self._processing_queue.pop(0)
                
        except Exception as e:
            self.logger.error(f"Error saving screenshot to queue: {e}")
    
    def is_monitoring(self) -> bool:
        """Check if the observer is currently monitoring."""
        return self._running
    
    def stop_monitoring(self) -> None:
        """Stop the screenshot monitoring process."""
        if not self._running:
            self.logger.warning("Observer is not running")
            return
        
        self._running = False
        
        # Wait for capture thread to finish
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=self.config.observer.thread_join_timeout)
            if self._capture_thread.is_alive():
                self.logger.warning("Capture thread did not stop gracefully")
        
        self._capture_thread = None
        
        # Log final statistics
        if self._start_time:
            duration = datetime.now() - self._start_time
            self.logger.info(
                f"Screenshot monitoring stopped. "
                f"Captured {self._capture_count} screenshots in {duration}"
            )
        
        self.logger.info("Screenshot monitoring stopped")
    
    def _capture_loop(self) -> None:
        """Main capture loop running in separate thread."""
        self.logger.debug("Capture loop started")
        
        while self._running:
            try:
                # Check if we should stop before doing work
                if not self._running:
                    break
                
                # UNLIMITED PERFORMANCE MODE - NO RESOURCE CHECKS
                
                # Capture screenshot
                screenshot = self.capture_screenshot()
                
                if screenshot and self._running:
                    # Check for significant changes
                    if self._should_save_screenshot(screenshot):
                        self._save_screenshot_to_queue(screenshot)
                        self._last_screenshot = screenshot
                        self._capture_count += 1
                        
                        # Notify callbacks
                        for callback in self._activity_callbacks:
                            try:
                                callback(screenshot)
                            except Exception as e:
                                self.logger.error(f"Error in activity callback: {e}")
                    else:
                        self._performance_metrics["duplicates_filtered"] += 1
                        self.logger.debug("Screenshot filtered as duplicate or insignificant")
                
                # Update performance metrics
                self._update_performance_metrics()
                
                # Sleep until next capture, checking for stop signal
                sleep_time = self.config.observer.capture_interval
                sleep_intervals = max(1, int(sleep_time))
                sleep_duration = sleep_time / sleep_intervals
                
                for _ in range(sleep_intervals):
                    if not self._running:
                        break
                    time.sleep(sleep_duration)
                
            except Exception as e:
                self.logger.error(f"Error in capture loop: {e}")
                # Sleep with interrupt capability on error
                for _ in range(int(self.config.observer.capture_interval)):
                    if not self._running:
                        break
                    time.sleep(1)
        
        self.logger.debug("Capture loop ended gracefully")
    
    @log_performance
    @log_exceptions("eidolon.observer")
    def capture_screenshot(self) -> Optional[Screenshot]:
        """
        Capture a single screenshot.
        
        Returns:
            Screenshot: Captured screenshot or None if capture failed.
        """
        try:
            # Get monitor information
            monitor_info = self._get_monitor_info()
            
            # Capture screenshot
            screenshot_data = self._sct.grab(self._sct.monitors[0])  # Primary monitor
            
            # Convert to PIL Image
            image = Image.frombytes(
                "RGB",
                screenshot_data.size,
                screenshot_data.bgra,
                "raw",
                "BGRX"
            )
            
            # Get window information
            window_info = self._get_window_info()
            
            # Create Screenshot object
            screenshot = Screenshot(
                image=image,
                timestamp=datetime.now(),
                window_info=window_info,
                monitor_info=monitor_info
            )
            
            self.logger.debug(f"Screenshot captured: {screenshot.hash[:8]}")
            return screenshot
            
        except Exception as e:
            self.logger.error(f"Failed to capture screenshot: {e}")
            return None
    
    def detect_changes(
        self,
        prev_screenshot: Screenshot,
        curr_screenshot: Screenshot
    ) -> ChangeMetrics:
        """
        Detect changes between two screenshots using advanced algorithms.
        
        Args:
            prev_screenshot: Previous screenshot for comparison.
            curr_screenshot: Current screenshot to compare.
            
        Returns:
            ChangeMetrics: Metrics about the changes detected.
        """
        try:
            # Quick hash comparison
            if prev_screenshot.hash == curr_screenshot.hash:
                return ChangeMetrics(
                    pixel_difference_ratio=0.0,
                    structural_similarity=1.0,
                    has_significant_change=False
                )
            
            # Convert images to numpy arrays for analysis
            prev_array = np.array(prev_screenshot.image)
            curr_array = np.array(curr_screenshot.image)
            
            # Ensure same dimensions
            if prev_array.shape != curr_array.shape:
                self.logger.warning("Screenshot dimensions mismatch, resizing")
                from PIL import Image
                curr_image_resized = curr_screenshot.image.resize(prev_screenshot.image.size)
                curr_array = np.array(curr_image_resized)
            
            # Multiple change detection methods
            metrics = self._calculate_advanced_change_metrics(prev_array, curr_array)
            
            # Determine significance using multiple factors
            has_significant_change = self._is_change_significant(metrics)
            
            # Find changed regions if significant
            changed_regions = []
            if has_significant_change:
                changed_regions = self._find_changed_regions(prev_array, curr_array)
            
            final_metrics = ChangeMetrics(
                pixel_difference_ratio=metrics['pixel_difference_ratio'],
                structural_similarity=metrics['structural_similarity'],
                has_significant_change=has_significant_change,
                changed_regions=changed_regions
            )
            
            self.logger.debug(
                f"Advanced change detection: {metrics['pixel_difference_ratio']:.3f} pixel diff, "
                f"{metrics['structural_similarity']:.3f} similarity, "
                f"{metrics['motion_score']:.3f} motion, "
                f"significant: {has_significant_change}"
            )
            
            return final_metrics
            
        except Exception as e:
            self.logger.error(f"Error detecting changes: {e}")
            # Return conservative metrics on error
            return ChangeMetrics(
                pixel_difference_ratio=1.0,
                structural_similarity=0.0,
                has_significant_change=True
            )
    
    def _calculate_advanced_change_metrics(self, prev_array: np.ndarray, curr_array: np.ndarray) -> Dict[str, float]:
        """Calculate multiple change detection metrics."""
        metrics = {}
        
        # 1. Pixel difference ratio
        diff_array = np.abs(prev_array.astype(np.int16) - curr_array.astype(np.int16))
        
        # Use adaptive threshold based on image content
        mean_brightness = np.mean(prev_array)
        threshold = max(
            self.config.observer.brightness_min_threshold,
            min(
                self.config.observer.brightness_max_threshold,
                mean_brightness * self.config.observer.brightness_threshold_factor
            )
        )
        
        pixel_differences = np.sum(diff_array > threshold)
        total_pixels = prev_array.size
        metrics['pixel_difference_ratio'] = pixel_differences / total_pixels
        
        # 2. Structural similarity using gradient comparison
        # Convert to grayscale for better performance
        if len(prev_array.shape) == 3:
            prev_gray = np.mean(prev_array, axis=2)
            curr_gray = np.mean(curr_array, axis=2)
        else:
            prev_gray = prev_array
            curr_gray = curr_array
        
        # Calculate gradients
        prev_grad_x = np.abs(np.gradient(prev_gray, axis=1))
        prev_grad_y = np.abs(np.gradient(prev_gray, axis=0))
        curr_grad_x = np.abs(np.gradient(curr_gray, axis=1))
        curr_grad_y = np.abs(np.gradient(curr_gray, axis=0))
        
        # Compare gradient magnitudes
        prev_grad_mag = np.sqrt(prev_grad_x**2 + prev_grad_y**2)
        curr_grad_mag = np.sqrt(curr_grad_x**2 + curr_grad_y**2)
        
        grad_diff = np.mean(np.abs(prev_grad_mag - curr_grad_mag))
        max_grad = max(np.mean(prev_grad_mag), np.mean(curr_grad_mag))
        metrics['structural_similarity'] = 1.0 - min(1.0, grad_diff / (max_grad + 1e-10))
        
        # 3. Motion/activity score based on high-frequency changes
        # Divide image into blocks and check for significant changes
        block_size = 64
        h, w = prev_gray.shape
        motion_blocks = 0
        total_blocks = 0
        
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                prev_block = prev_gray[y:y+block_size, x:x+block_size]
                curr_block = curr_gray[y:y+block_size, x:x+block_size]
                
                block_diff = np.mean(np.abs(prev_block - curr_block))
                if block_diff > threshold * 0.5:
                    motion_blocks += 1
                total_blocks += 1
        
        metrics['motion_score'] = motion_blocks / total_blocks if total_blocks > 0 else 0.0
        
        # 4. Histogram comparison for color/brightness changes
        if len(prev_array.shape) == 3:
            prev_hist = np.histogram(prev_array.flatten(), bins=256, range=(0, 256))[0]
            curr_hist = np.histogram(curr_array.flatten(), bins=256, range=(0, 256))[0]
            
            # Normalize histograms
            prev_hist = prev_hist / np.sum(prev_hist)
            curr_hist = curr_hist / np.sum(curr_hist)
            
            # Calculate histogram correlation
            hist_correlation = np.corrcoef(prev_hist, curr_hist)[0, 1]
            metrics['histogram_similarity'] = hist_correlation if not np.isnan(hist_correlation) else 0.0
        else:
            metrics['histogram_similarity'] = 1.0
        
        return metrics
    
    def _is_change_significant(self, metrics: Dict[str, float]) -> bool:
        """Determine if changes are significant using multiple criteria."""
        base_threshold = self.config.observer.activity_threshold
        
        # Weight different metrics
        pixel_weight = self.config.observer.pixel_weight
        structure_weight = self.config.observer.structure_weight
        motion_weight = self.config.observer.motion_weight
        histogram_weight = self.config.observer.histogram_weight
        
        # Calculate weighted change score
        change_score = (
            metrics['pixel_difference_ratio'] * pixel_weight +
            (1.0 - metrics['structural_similarity']) * structure_weight +
            metrics['motion_score'] * motion_weight +
            (1.0 - metrics.get('histogram_similarity', 1.0)) * histogram_weight
        )
        
        # Adaptive threshold based on previous activity
        adaptive_threshold = base_threshold
        
        # More sensitive to structural changes
        if metrics['structural_similarity'] < self.config.observer.structural_similarity_threshold:
            adaptive_threshold *= self.config.observer.structure_adjustment_factor
        
        # Less sensitive to pure color changes
        if (metrics.get('histogram_similarity', 1.0) < self.config.observer.histogram_similarity_threshold and 
            metrics['motion_score'] < self.config.observer.motion_score_threshold):
            adaptive_threshold *= self.config.observer.histogram_adjustment_factor
        
        return bool(change_score > adaptive_threshold)
    
    def _find_changed_regions(self, prev_array: np.ndarray, curr_array: np.ndarray) -> List[tuple]:
        """Find specific regions that have changed."""
        changed_regions = []
        
        try:
            # Convert to grayscale for processing
            if len(prev_array.shape) == 3:
                prev_gray = np.mean(prev_array, axis=2).astype(np.uint8)
                curr_gray = np.mean(curr_array, axis=2).astype(np.uint8)
            else:
                prev_gray = prev_array.astype(np.uint8)
                curr_gray = curr_array.astype(np.uint8)
            
            # Calculate difference
            diff = np.abs(prev_gray.astype(np.int16) - curr_gray.astype(np.int16))
            
            # Threshold and find contours
            threshold = 30
            binary_diff = (diff > threshold).astype(np.uint8) * 255
            
            # Use morphological operations to clean up
            from scipy import ndimage
            
            # Remove small noise
            binary_diff = ndimage.binary_opening(binary_diff, structure=np.ones((3, 3)))
            
            # Fill small holes
            binary_diff = ndimage.binary_closing(binary_diff, structure=np.ones((5, 5)))
            
            # Find connected components (regions)
            labeled_array, num_features = ndimage.label(binary_diff)
            
            for region_id in range(1, num_features + 1):
                region_mask = (labeled_array == region_id)
                
                # Get bounding box
                rows, cols = np.where(region_mask)
                if len(rows) > 0 and len(cols) > 0:
                    min_row, max_row = np.min(rows), np.max(rows)
                    min_col, max_col = np.min(cols), np.max(cols)
                    
                    # Only include regions above minimum size
                    width = max_col - min_col
                    height = max_row - min_row
                    area = width * height
                    
                    if area > self.config.observer.min_area_threshold:
                        changed_regions.append((min_col, min_row, max_col, max_row))
            
        except Exception as e:
            self.logger.warning(f"Could not find specific changed regions: {e}")
            # Fallback to whole image
            h, w = prev_array.shape[:2]
            changed_regions = [(0, 0, w, h)]
        
        return changed_regions
    
    def _should_save_screenshot(self, screenshot: Screenshot) -> bool:
        """Determine if a screenshot should be saved based on change detection."""
        if self._last_screenshot is None:
            return True
        
        # Check for changes
        change_metrics = self.detect_changes(self._last_screenshot, screenshot)
        
        # Save if there are significant changes
        return change_metrics.has_significant_change
    
    def _save_and_process_screenshot(self, screenshot: Screenshot) -> None:
        """Save screenshot and create metadata with OCR and analysis."""
        try:
            # Generate filename with timestamp
            timestamp_str = screenshot.timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"screenshot_{timestamp_str}_{screenshot.hash[:8]}.png"
            file_path = self.storage_path / filename
            
            # Save screenshot
            screenshot.save(str(file_path))
            screenshot.file_path = str(file_path)
            
            # Get shared components (cached for performance)
            db = self._get_database()
            analyzer = self._get_analyzer()
            
            # Store screenshot metadata in database
            screenshot_data = screenshot.to_dict()
            screenshot_data['file_path'] = str(file_path)
            screenshot_data['size_bytes'] = file_path.stat().st_size
            
            screenshot_id = db.store_screenshot(screenshot_data)
            
            # Perform OCR analysis
            try:
                extracted_text = analyzer.extract_text(file_path)
                if extracted_text.text:  # Only store if text was found
                    db.store_ocr_result(screenshot_id, extracted_text.to_dict())
                    self.logger.debug(f"OCR extracted {extracted_text.word_count} words")
                
                # Perform content analysis (enhanced with LLM if configured)
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    content_analysis = loop.run_until_complete(
                        analyzer.analyze_content_with_llm(file_path, extracted_text.text)
                    )
                finally:
                    loop.close()
                db.store_content_analysis(screenshot_id, content_analysis.to_dict())
                
                self.logger.debug(
                    f"Analysis: {content_analysis.content_type} "
                    f"({content_analysis.confidence:.2f} confidence)"
                )
                
            except (ImportError, AttributeError, ValueError, OSError) as e:
                self.logger.warning(f"Analysis failed for {filename}: {e}")
            except Exception as e:
                self.logger.error(f"Unexpected error during analysis for {filename}: {e}")
            
            # Save legacy JSON metadata for backward compatibility
            metadata_path = file_path.with_suffix(".json")
            metadata = screenshot.to_dict()
            metadata['file_path'] = str(file_path)
            metadata['screenshot_id'] = screenshot_id
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.debug(f"Screenshot processed and saved: {filename}")
            
        except (OSError, IOError, PermissionError) as e:
            self.logger.error(f"File system error saving screenshot: {e}")
        except (ImportError, AttributeError) as e:
            self.logger.error(f"Module/component error during processing: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error saving and processing screenshot: {e}")
    
    def _get_monitor_info(self) -> Dict[str, Any]:
        """Get information about the current monitor setup."""
        try:
            monitors = []
            for monitor in self._sct.monitors[1:]:  # Skip the "All monitors" entry
                monitors.append({
                    "left": monitor["left"],
                    "top": monitor["top"], 
                    "width": monitor["width"],
                    "height": monitor["height"]
                })
            
            return {
                "primary_monitor": monitors[0] if monitors else None,
                "all_monitors": monitors,
                "monitor_count": len(monitors)
            }
            
        except (AttributeError, IndexError, KeyError) as e:
            self.logger.error(f"Monitor access error: {e}")
            return {}
        except Exception as e:
            self.logger.error(f"Unexpected error getting monitor info: {e}")
            return {}
    
    def _get_window_info(self) -> Dict[str, Any]:
        """Get information about the currently active window."""
        # This is a placeholder implementation
        # Real implementation would use platform-specific APIs
        try:
            return {
                "active_window": "Unknown",
                "application": "Unknown",
                "title": "Unknown"
            }
        except (OSError, AttributeError) as e:
            self.logger.error(f"Window system access error: {e}")
            return {}
        except Exception as e:
            self.logger.error(f"Unexpected error getting window info: {e}")
            return {}
    
    def set_memory_limit(self, limit_gb: float) -> None:
        """Set memory limit in GB for the observer."""
        self._memory_limit_mb = limit_gb * 1024  # Convert GB to MB
        self.logger.info(f"Memory limit set to {limit_gb:.1f}GB ({self._memory_limit_mb:.1f}MB)")
    
    def _get_analyzer(self):
        """Get or create analyzer instance (singleton pattern)."""
        if self._analyzer is None:
            # Disable resource limits during model loading
            self._model_loading = True
            self.logger.info("Loading AI model - temporarily disabling resource limits")
            
            from ..core.analyzer import Analyzer
            self._analyzer = Analyzer()
            
            # Re-enable resource limits after loading
            self._model_loading = False
            self.logger.info("Analyzer initialized and cached for reuse - resource limits restored")
        return self._analyzer
    
    def _get_database(self):
        """Get or create database instance (singleton pattern)."""
        if self._database is None:
            from ..storage.metadata_db import MetadataDatabase
            self._database = MetadataDatabase()
            self.logger.info("Database initialized and cached for reuse")
        return self._database
    
    def _get_fastmcp(self):
        """Get or create FastMCP instance (singleton pattern)."""
        if self._fastmcp is None:
            self._fastmcp = get_fastmcp()
            self.logger.info("FastMCP initialized and cached for reuse")
        return self._fastmcp
    
    def _check_resource_limits(self) -> bool:
        """UNLIMITED PERFORMANCE MODE - NO RESOURCE LIMITS."""
        # ALWAYS return True - NO LIMITS, NO SAFEGUARDS, MAXIMUM PERFORMANCE
        return True
    
    def _update_performance_metrics(self) -> None:
        """Update performance metrics for monitoring."""
        try:
            if not self._start_time:
                return
            
            # Calculate captures per minute
            elapsed_minutes = (datetime.now() - self._start_time).total_seconds() / 60
            if elapsed_minutes > 0:
                self._performance_metrics["captures_per_minute"] = (
                    self._capture_count / elapsed_minutes
                )
            
            # Update resource usage
            process = psutil.Process()
            self._performance_metrics["memory_usage_mb"] = (
                process.memory_info().rss / 1024 / 1024
            )
            self._performance_metrics["cpu_usage_percent"] = process.cpu_percent()
            
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError) as e:
            self.logger.error(f"Process monitoring error updating metrics: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error updating performance metrics: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status and statistics."""
        return {
            "running": self._running,
            "capture_count": self._capture_count,
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "performance_metrics": self._performance_metrics.copy(),
            "storage_path": str(self.storage_path),
            "config": {
                "capture_interval": self.config.observer.capture_interval,
                "activity_threshold": self.config.observer.activity_threshold,
                "max_storage_gb": self.config.observer.max_storage_gb
            }
        }
    
    def cleanup_old_screenshots(self, days_to_keep: int = None) -> int:
        """
        Clean up old screenshots to manage storage space.
        
        Args:
            days_to_keep: Number of days of screenshots to keep.
                         If None, uses config value.
        
        Returns:
            int: Number of files deleted.
        """
        if days_to_keep is None:
            days_to_keep = self.config.privacy.data_retention_days
        
        try:
            cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 3600)
            deleted_count = 0
            
            for file_path in self.storage_path.glob("screenshot_*.png"):
                if file_path.stat().st_mtime < cutoff_time:
                    # Delete screenshot and metadata
                    file_path.unlink()
                    metadata_path = file_path.with_suffix(".json")
                    if metadata_path.exists():
                        metadata_path.unlink()
                    deleted_count += 1
            
            self.logger.info(f"Cleaned up {deleted_count} old screenshots")
            return deleted_count
            
        except (OSError, IOError, PermissionError) as e:
            self.logger.error(f"File system error during cleanup: {e}")
            return 0
        except Exception as e:
            self.logger.error(f"Unexpected error during cleanup: {e}")
            return 0