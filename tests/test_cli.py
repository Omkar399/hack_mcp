"""
Tests for Eidolon CLI functionality
"""

import pytest
from click.testing import CliRunner
from unittest.mock import Mock, patch

from eidolon.cli import cli


class TestCLI:
    """Test cases for CLI commands."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_cli_help(self):
        """Test CLI help command."""
        result = self.runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'Eidolon AI Personal Assistant' in result.output
    
    def test_status_command(self):
        """Test status command."""
        with patch('eidolon.cli.get_observer') as mock_observer:
            mock_observer.return_value.get_status.return_value = {
                'running': False,
                'capture_count': 0
            }
            with patch('eidolon.cli.get_interface') as mock_interface:
                mock_interface.return_value.get_status.return_value = {}
                
                result = self.runner.invoke(cli, ['status'])
                assert result.exit_code == 0
                assert 'System Status' in result.output
    
    def test_search_command(self):
        """Test search command."""
        import asyncio
        
        async def mock_search(query, limit=10):
            return [
                {'title': 'Test Result', 'content': 'Test content', 'timestamp': '2024-01-01'}
            ]
        
        with patch('eidolon.cli.get_interface') as mock_interface:
            mock_interface.return_value.search = mock_search
            
            result = self.runner.invoke(cli, ['search', 'test query'])
            assert result.exit_code == 0
            assert 'Search results' in result.output


if __name__ == "__main__":
    pytest.main([__file__])
