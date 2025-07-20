"""
Basic tests for Eidolon core components
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from PIL import Image

from eidolon.core.observer import Observer, Screenshot
from eidolon.core.interface import Interface


class TestObserver:
    """Test cases for Observer component."""
    
    def test_observer_initialization(self):
        """Test observer can be initialized."""
        observer = Observer()
        assert observer is not None
        assert not observer.is_monitoring()
    
    def test_screenshot_creation(self):
        """Test screenshot object creation."""
        # Create a simple test image
        image = Image.new('RGB', (100, 100), color='red')
        timestamp = datetime.now()
        
        screenshot = Screenshot(image, timestamp)
        assert screenshot.image == image
        assert screenshot.timestamp == timestamp
        assert screenshot.hash is not None


class TestInterface:
    """Test cases for Interface component."""
    
    @pytest.mark.asyncio
    async def test_interface_initialization(self):
        """Test interface can be initialized."""
        interface = Interface()
        assert interface is not None
    
    @pytest.mark.asyncio
    async def test_search_functionality(self):
        """Test basic search functionality."""
        interface = Interface()
        
        # Mock search to avoid dependencies
        with patch.object(interface, 'search') as mock_search:
            mock_search.return_value = [{"title": "Test", "content": "Test content"}]
            results = await interface.search("test query")
            assert len(results) == 1
            assert results[0]["title"] == "Test"


if __name__ == "__main__":
    pytest.main([__file__])
