"""
Tests for the Observer component
"""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from eidolon.core.observer import Observer, Screenshot, ChangeMetrics
from eidolon.utils.config import Config, ObserverConfig


class TestObserver:
    """Test cases for the Observer component."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory for tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def mock_config(self, temp_storage):
        """Create mock configuration for tests."""
        config_override = {
            "storage_path": temp_storage,
            "capture_interval": 1,  # Fast for testing
            "max_storage_gb": 1.0,
            "max_cpu_percent": 50.0,
            "max_memory_mb": 1000,
            # Add new configuration attributes
            "thread_join_timeout": 5.0,
            "brightness_threshold_factor": 0.1,
            "brightness_min_threshold": 20,
            "brightness_max_threshold": 50,
            "structural_similarity_threshold": 0.9,
            "histogram_similarity_threshold": 0.9,
            "motion_score_threshold": 0.1,
            "pixel_weight": 0.4,
            "structure_weight": 0.3,
            "motion_weight": 0.2,
            "histogram_weight": 0.1,
            "structure_adjustment_factor": 0.8,
            "histogram_adjustment_factor": 1.5,
            "cpu_check_interval": 0.1,
            "min_area_threshold": 100,
            "text_confidence_threshold": 20,
            "ocr_confidence_threshold": 30,
            "sleep_interval_short": 1,
            "sleep_interval_status": 30
        }
        return config_override
    
    @pytest.fixture
    def observer(self, mock_config):
        """Create Observer instance with test configuration."""
        return Observer(config_override=mock_config)
    
    def test_observer_initialization(self, observer):
        """Test that Observer initializes correctly."""
        assert observer is not None
        assert not observer._running
        assert observer._capture_count == 0
        assert observer._last_screenshot is None
    
    def test_storage_directory_creation(self, observer):
        """Test that storage directory is created."""
        assert observer.storage_path.exists()
        assert observer.storage_path.is_dir()
    
    @patch('eidolon.core.observer.mss.mss')
    def test_screenshot_capture(self, mock_mss, observer):
        """Test screenshot capture functionality."""
        # Mock the screenshot data
        mock_screenshot = Mock()
        mock_screenshot.size = (1920, 1080)
        mock_screenshot.bgra = b'mock_image_data' * 1000  # Simulate image data
        
        mock_sct_instance = Mock()
        mock_sct_instance.grab.return_value = mock_screenshot
        mock_sct_instance.monitors = [
            {"left": 0, "top": 0, "width": 1920, "height": 1080},  # Primary monitor
            {"left": 0, "top": 0, "width": 1920, "height": 1080}   # All monitors entry
        ]
        
        mock_mss.return_value = mock_sct_instance
        
        with patch('PIL.Image.frombytes') as mock_frombytes:
            mock_image = Mock()
            mock_image.size = (1920, 1080)
            mock_image.tobytes.return_value = b'mock_image_bytes'
            mock_frombytes.return_value = mock_image
            
            screenshot = observer.capture_screenshot()
            
            assert screenshot is not None
            assert isinstance(screenshot, Screenshot)
            assert screenshot.image == mock_image
            assert screenshot.hash is not None
    
    def test_change_detection_identical_images(self, observer):
        """Test change detection with identical images."""
        # Create mock screenshots with same hash
        mock_image = Mock()
        mock_image.tobytes.return_value = b'same_image_data'
        
        screenshot1 = Screenshot(mock_image, observer._start_time or time.time())
        screenshot2 = Screenshot(mock_image, observer._start_time or time.time())
        
        # Force same hash for test
        screenshot1.hash = "same_hash"
        screenshot2.hash = "same_hash"
        
        metrics = observer.detect_changes(screenshot1, screenshot2)
        
        assert isinstance(metrics, ChangeMetrics)
        assert metrics.pixel_difference_ratio == 0.0
        assert metrics.structural_similarity == 1.0
        assert not metrics.has_significant_change
    
    def test_change_detection_different_images(self, observer):
        """Test change detection with different images."""
        import numpy as np
        from PIL import Image
        
        # Create different images
        image1_array = np.zeros((100, 100, 3), dtype=np.uint8)
        image2_array = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        image1 = Image.fromarray(image1_array)
        image2 = Image.fromarray(image2_array)
        
        screenshot1 = Screenshot(image1, observer._start_time or time.time())
        screenshot2 = Screenshot(image2, observer._start_time or time.time())
        
        metrics = observer.detect_changes(screenshot1, screenshot2)
        
        assert isinstance(metrics, ChangeMetrics)
        assert metrics.pixel_difference_ratio > 0.0
        assert metrics.structural_similarity < 1.0
        assert metrics.has_significant_change
    
    def test_start_stop_monitoring(self, observer):
        """Test starting and stopping monitoring."""
        assert not observer._running
        
        # Start monitoring
        observer.start_monitoring()
        assert observer._running
        assert observer._capture_thread is not None
        
        # Give it a moment to start
        time.sleep(0.1)
        
        # Stop monitoring
        observer.stop_monitoring()
        assert not observer._running
    
    def test_resource_limit_checking(self, observer):
        """Test resource limit checking."""
        # Should pass with reasonable limits
        assert observer._check_resource_limits()
        
        # Test with very low limits
        observer.config.observer.max_memory_mb = 1  # 1MB limit
        observer.config.observer.max_cpu_percent = 0.1  # 0.1% CPU limit
        
        # Should fail with unreasonably low limits
        # (though this depends on actual system usage)
        result = observer._check_resource_limits()
        # Don't assert the result as it depends on system state
        assert isinstance(result, bool)
    
    def test_performance_metrics_update(self, observer):
        """Test performance metrics updating."""
        # Initialize with start time
        observer._start_time = time.time()
        observer._capture_count = 5
        
        observer._update_performance_metrics()
        
        metrics = observer._performance_metrics
        assert "captures_per_minute" in metrics
        assert "memory_usage_mb" in metrics
        assert "cpu_usage_percent" in metrics
        assert isinstance(metrics["captures_per_minute"], float)
    
    def test_get_status(self, observer):
        """Test status information retrieval."""
        status = observer.get_status()
        
        assert isinstance(status, dict)
        assert "running" in status
        assert "capture_count" in status
        assert "performance_metrics" in status
        assert "storage_path" in status
        assert "config" in status
        
        assert status["running"] == observer._running
        assert status["capture_count"] == observer._capture_count
    
    def test_cleanup_old_screenshots(self, observer, temp_storage):
        """Test cleanup of old screenshots."""
        # Create some mock screenshot files
        storage_path = Path(temp_storage)
        
        # Create test files with different ages
        old_file = storage_path / "screenshot_old.png"
        old_file.touch()
        
        new_file = storage_path / "screenshot_new.png" 
        new_file.touch()
        
        # Set old file to be older than retention period
        old_time = time.time() - (2 * 24 * 3600)  # 2 days ago
        old_file.touch(times=(old_time, old_time))
        
        # Cleanup with 1 day retention
        deleted_count = observer.cleanup_old_screenshots(days_to_keep=1)
        
        # Should delete the old file but keep the new one
        assert not old_file.exists()
        assert new_file.exists()
        assert deleted_count >= 1
    
    def test_activity_callbacks(self, observer):
        """Test activity callback functionality."""
        callback_called = False
        received_screenshot = None
        
        def test_callback(screenshot):
            nonlocal callback_called, received_screenshot
            callback_called = True
            received_screenshot = screenshot
        
        observer.add_activity_callback(test_callback)
        
        # Create mock screenshot
        mock_image = Mock()
        mock_image.tobytes.return_value = b'test_data'
        screenshot = Screenshot(mock_image, time.time())
        
        # Manually call the callback (simulating capture)
        for callback in observer._activity_callbacks:
            callback(screenshot)
        
        assert callback_called
        assert received_screenshot == screenshot


class TestScreenshot:
    """Test cases for the Screenshot class."""
    
    def test_screenshot_creation(self):
        """Test Screenshot object creation."""
        from PIL import Image
        import numpy as np
        
        # Create test image
        image_array = np.zeros((100, 100, 3), dtype=np.uint8)
        image = Image.fromarray(image_array)
        timestamp = time.time()
        
        screenshot = Screenshot(image, timestamp)
        
        assert screenshot.image == image
        assert screenshot.timestamp == timestamp
        assert screenshot.hash is not None
        assert isinstance(screenshot.hash, str)
        assert len(screenshot.hash) == 64  # SHA-256 hash length
    
    def test_screenshot_hash_consistency(self):
        """Test that identical images produce identical hashes."""
        from PIL import Image
        import numpy as np
        
        # Create identical images
        image_array = np.zeros((50, 50, 3), dtype=np.uint8)
        image1 = Image.fromarray(image_array.copy())
        image2 = Image.fromarray(image_array.copy())
        
        screenshot1 = Screenshot(image1, time.time())
        screenshot2 = Screenshot(image2, time.time())
        
        assert screenshot1.hash == screenshot2.hash
    
    def test_screenshot_to_dict(self):
        """Test conversion to dictionary."""
        from PIL import Image
        import numpy as np
        
        image_array = np.zeros((100, 100, 3), dtype=np.uint8)
        image = Image.fromarray(image_array)
        timestamp = time.time()
        
        window_info = {"title": "Test Window"}
        monitor_info = {"width": 1920, "height": 1080}
        
        screenshot = Screenshot(
            image=image,
            timestamp=timestamp,
            window_info=window_info,
            monitor_info=monitor_info
        )
        
        data_dict = screenshot.to_dict()
        
        assert isinstance(data_dict, dict)
        assert "timestamp" in data_dict
        assert "hash" in data_dict
        assert "window_info" in data_dict
        assert "monitor_info" in data_dict
        assert "size" in data_dict
        
        assert data_dict["window_info"] == window_info
        assert data_dict["monitor_info"] == monitor_info
        assert data_dict["size"] == image.size


class TestChangeMetrics:
    """Test cases for the ChangeMetrics class."""
    
    def test_change_metrics_creation(self):
        """Test ChangeMetrics object creation."""
        metrics = ChangeMetrics(
            pixel_difference_ratio=0.1,
            structural_similarity=0.9,
            has_significant_change=True,
            changed_regions=[(0, 0, 100, 100)]
        )
        
        assert metrics.pixel_difference_ratio == 0.1
        assert metrics.structural_similarity == 0.9
        assert metrics.has_significant_change is True
        assert len(metrics.changed_regions) == 1
    
    def test_change_metrics_to_dict(self):
        """Test conversion to dictionary."""
        metrics = ChangeMetrics(
            pixel_difference_ratio=0.2,
            structural_similarity=0.8,
            has_significant_change=False
        )
        
        data_dict = metrics.to_dict()
        
        assert isinstance(data_dict, dict)
        assert data_dict["pixel_difference_ratio"] == 0.2
        assert data_dict["structural_similarity"] == 0.8
        assert data_dict["has_significant_change"] is False
        assert "changed_regions" in data_dict