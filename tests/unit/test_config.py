"""
Tests for the configuration system
"""

import pytest
import tempfile
import os
from pathlib import Path

from eidolon.utils.config import (
    Config, ObserverConfig, AnalysisConfig, MemoryConfig,
    load_config, save_config, substitute_env_vars
)


class TestConfig:
    """Test cases for configuration system."""
    
    def test_default_config_creation(self):
        """Test creating config with default values."""
        config = Config()
        
        assert isinstance(config.observer, ObserverConfig)
        assert isinstance(config.analysis, AnalysisConfig)
        assert isinstance(config.memory, MemoryConfig)
        
        # Test some default values
        assert config.observer.capture_interval == 10
        assert config.observer.activity_threshold == 0.05
        assert config.memory.chunk_size == 512
        assert config.privacy.auto_redaction is True
    
    def test_observer_config_validation(self):
        """Test ObserverConfig validation."""
        # Valid config
        config = ObserverConfig(
            capture_interval=30,
            activity_threshold=0.1,
            max_storage_gb=100.0
        )
        assert config.capture_interval == 30
        assert config.activity_threshold == 0.1
        assert config.max_storage_gb == 100.0
        
        # Test validation bounds
        with pytest.raises(ValueError):
            ObserverConfig(capture_interval=0)  # Too low
            
        with pytest.raises(ValueError):
            ObserverConfig(activity_threshold=1.5)  # Too high
    
    def test_config_directory_methods(self):
        """Test configuration directory helper methods."""
        config = Config()
        
        # Test directory path methods
        data_dir = config.get_data_dir()
        logs_dir = config.get_logs_dir()
        
        assert isinstance(data_dir, Path)
        assert isinstance(logs_dir, Path)
    
    def test_ensure_directories(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config()
            config.observer.storage_path = f"{temp_dir}/screenshots"
            config.logging.file_path = f"{temp_dir}/logs/eidolon.log"
            config.memory.db_path = f"{temp_dir}/data/eidolon.db"
            
            config.ensure_directories()
            
            assert Path(config.observer.storage_path).exists()
            assert Path(config.logging.file_path).parent.exists()
            assert Path(config.memory.db_path).parent.exists()


class TestConfigLoading:
    """Test cases for configuration loading and saving."""
    
    def test_substitute_env_vars(self):
        """Test environment variable substitution."""
        # Set test environment variable
        os.environ["TEST_API_KEY"] = "test_key_123"
        
        config_dict = {
            "api_key": "${TEST_API_KEY}",
            "normal_value": "no_substitution",
            "nested": {
                "api_secret": "${TEST_API_KEY}"
            }
        }
        
        result = substitute_env_vars(config_dict)
        
        assert result["api_key"] == "test_key_123"
        assert result["normal_value"] == "no_substitution"
        assert result["nested"]["api_secret"] == "test_key_123"
        
        # Clean up
        del os.environ["TEST_API_KEY"]
    
    def test_substitute_env_vars_missing(self):
        """Test environment variable substitution with missing vars."""
        config_dict = {
            "missing_var": "${NON_EXISTENT_VAR}"
        }
        
        result = substitute_env_vars(config_dict)
        
        # Should keep original value if env var doesn't exist
        assert result["missing_var"] == "${NON_EXISTENT_VAR}"
    
    def test_load_config_from_file(self):
        """Test loading configuration from YAML file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_content = f"""
observer:
  capture_interval: 20
  activity_threshold: 0.1
  storage_path: "{temp_dir}/custom/screenshots"

analysis:
  routing:
    importance_threshold: 0.8
    cost_limit_daily: 5.0

memory:
  db_path: "{temp_dir}/custom/eidolon.db"

logging:
  file_path: "{temp_dir}/custom/logs/eidolon.log"

privacy:
  auto_redaction: false
  data_retention_days: 180
"""
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write(yaml_content)
                temp_path = f.name
            
            try:
                config = load_config(temp_path)
                
                assert config.observer.capture_interval == 20
                assert config.observer.activity_threshold == 0.1
                assert config.observer.storage_path == f"{temp_dir}/custom/screenshots"
                assert config.analysis.routing["importance_threshold"] == 0.8
                assert config.analysis.routing["cost_limit_daily"] == 5.0
                assert config.privacy.auto_redaction is False
                assert config.privacy.data_retention_days == 180
                
            finally:
                os.unlink(temp_path)
    
    def test_load_config_with_env_vars(self):
        """Test loading configuration with environment variables."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set test environment variables
            os.environ["TEST_STORAGE_PATH"] = f"{temp_dir}/env/storage"
            os.environ["TEST_INTERVAL"] = "15"
            
            yaml_content = f"""
observer:
  capture_interval: ${{TEST_INTERVAL}}
  storage_path: "${{TEST_STORAGE_PATH}}"
memory:
  db_path: "{temp_dir}/env/eidolon.db"
logging:
  file_path: "{temp_dir}/env/logs/eidolon.log"
"""
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write(yaml_content)
                temp_path = f.name
            
            try:
                config = load_config(temp_path)
                
                # Note: YAML will load TEST_INTERVAL as string, but Pydantic should convert
                assert config.observer.storage_path == f"{temp_dir}/env/storage"
                
            finally:
                os.unlink(temp_path)
                del os.environ["TEST_STORAGE_PATH"]
                del os.environ["TEST_INTERVAL"]
    
    def test_load_config_nonexistent_file(self):
        """Test loading configuration from nonexistent file."""
        # Should return default config when no file found
        config = load_config("/nonexistent/path/config.yaml")
        assert isinstance(config, Config)
        
        # Should use default values
        assert config.observer.capture_interval == 10
    
    def test_save_config(self):
        """Test saving configuration to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config()
            config.observer.capture_interval = 25
            config.observer.storage_path = f"{temp_dir}/test/screenshots"
            config.memory.db_path = f"{temp_dir}/test/eidolon.db"
            config.logging.file_path = f"{temp_dir}/test/logs/eidolon.log"
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                temp_path = f.name
            
            try:
                save_config(config, temp_path)
                
                # Load it back and verify
                loaded_config = load_config(temp_path)
                assert loaded_config.observer.capture_interval == 25
                assert loaded_config.observer.storage_path == f"{temp_dir}/test/screenshots"
                
            finally:
                os.unlink(temp_path)
    
    def test_invalid_yaml_config(self):
        """Test handling of invalid YAML configuration."""
        invalid_yaml = """
observer:
  capture_interval: 10
    invalid_indentation: true
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError):
                load_config(temp_path)
                
        finally:
            os.unlink(temp_path)
    
    def test_config_validation_errors(self):
        """Test configuration validation with invalid values."""
        yaml_content = """
observer:
  capture_interval: -5  # Invalid: must be positive
  activity_threshold: 2.0  # Invalid: must be <= 1.0
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError):
                load_config(temp_path)
                
        finally:
            os.unlink(temp_path)


class TestConfigComponents:
    """Test individual configuration components."""
    
    def test_observer_config_defaults(self):
        """Test ObserverConfig default values."""
        config = ObserverConfig()
        
        assert config.capture_interval == 10
        assert config.activity_threshold == 0.05
        assert config.storage_path == "./data/screenshots"
        assert config.max_storage_gb == 50.0
        assert config.monitor_keyboard is True
        assert config.monitor_mouse is True
        assert len(config.sensitive_patterns) > 0
        assert "password" in config.sensitive_patterns
    
    def test_analysis_config_defaults(self):
        """Test AnalysisConfig default values."""
        config = AnalysisConfig()
        
        assert "vision" in config.local_models
        assert "clip" in config.local_models
        assert "embedding" in config.local_models
        assert config.routing["importance_threshold"] == 0.7
        assert config.routing["local_first"] is True
        assert config.ocr["engine"] == "tesseract"
    
    def test_memory_config_defaults(self):
        """Test MemoryConfig default values."""
        config = MemoryConfig()
        
        assert config.vector_db == "chromadb"
        assert config.chunk_size == 512
        assert config.overlap == 50
        assert config.metadata_db == "sqlite"
        assert config.search["max_results"] == 50
        assert config.search["enable_semantic_search"] is True
    
    def test_privacy_config_defaults(self):
        """Test PrivacyConfig default values."""
        from eidolon.utils.config import PrivacyConfig
        
        config = PrivacyConfig()
        
        assert config.local_only_mode is False
        assert config.auto_redaction is True
        assert config.data_retention_days == 365
        assert config.encrypt_at_rest is True
        assert config.pause_on_sensitive_apps is True