"""
Configuration management for Eidolon AI Personal Assistant

Handles loading and validation of configuration files with environment variable
substitution and type checking.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv


logger = logging.getLogger(__name__)


class ObserverConfig(BaseModel):
    """Configuration for the Observer component."""
    
    capture_interval: float = Field(default=0.1, ge=0.05, le=300)  # Ultra-fast 10 FPS default
    activity_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    storage_path: str = Field(default="./data/screenshots")
    max_storage_gb: float = Field(default=50.0, ge=1.0)
    monitor_keyboard: bool = Field(default=True)
    monitor_mouse: bool = Field(default=True)
    monitor_window_changes: bool = Field(default=True)
    max_cpu_percent: float = Field(default=1000.0, ge=0.1, le=10000.0)  # UNLIMITED - No CPU limits
    max_memory_mb: int = Field(default=100000, ge=100)  # UNLIMITED - No memory limits
    sensitive_patterns: list[str] = Field(default_factory=lambda: [
        "password", "api_key", "secret", "token", "ssn", "credit_card"
    ])
    excluded_apps: list[str] = Field(default_factory=lambda: [
        "1Password", "Keychain Access", "LastPass", "Bitwarden"
    ])
    
    # Timing settings
    sleep_interval_short: float = Field(default=1.0, ge=0.1)
    sleep_interval_status: float = Field(default=30.0, ge=1.0)
    
    # Processing thresholds
    min_area_threshold: int = Field(default=100, ge=1)
    text_confidence_threshold: float = Field(default=20.0, ge=0.0, le=100.0)
    ocr_confidence_threshold: float = Field(default=30.0, ge=0.0, le=100.0)
    
    # Image analysis thresholds
    brightness_threshold_factor: float = Field(default=0.1, ge=0.01, le=1.0)
    brightness_min_threshold: int = Field(default=20, ge=1, le=255)
    brightness_max_threshold: int = Field(default=50, ge=1, le=255)
    structural_similarity_threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    histogram_similarity_threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    motion_score_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    
    # Change detection weights
    pixel_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    structure_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    motion_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    histogram_weight: float = Field(default=0.1, ge=0.0, le=1.0)
    
    # Adaptive threshold factors
    structure_adjustment_factor: float = Field(default=0.8, ge=0.1, le=2.0)
    histogram_adjustment_factor: float = Field(default=1.5, ge=0.1, le=3.0)
    
    # Performance settings
    cpu_check_interval: float = Field(default=0.1, ge=0.01, le=5.0)
    thread_join_timeout: float = Field(default=5.0, ge=1.0, le=30.0)


class AnalysisConfig(BaseModel):
    """Configuration for the Analysis component."""
    
    local_models: Dict[str, str] = Field(default_factory=lambda: {
        "vision": "microsoft/florence-2-base",
        "clip": "openai/clip-vit-base-patch32",
        "embedding": "sentence-transformers/all-MiniLM-L6-v2"
    })
    cloud_apis: Dict[str, Optional[str]] = Field(default_factory=lambda: {
        "gemini_key": None,
        "claude_key": None,
        "openrouter_key": None,
        "openai_key": None
    })
    routing: Dict[str, Union[float, bool]] = Field(default_factory=lambda: {
        "importance_threshold": 0.7,
        "cost_limit_daily": 10.0,
        "local_first": True
    })
    llm_enhanced_analysis: bool = Field(default=False)
    ocr: Dict[str, Union[str, list, float]] = Field(default_factory=lambda: {
        "engine": "tesseract",
        "languages": ["en"],
        "confidence_threshold": 0.6
    })


class MemoryConfig(BaseModel):
    """Configuration for the Memory component."""
    
    vector_db: str = Field(default="chromadb")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    chunk_size: int = Field(default=512, ge=100, le=2048)
    overlap: int = Field(default=50, ge=0, le=200)
    metadata_db: str = Field(default="sqlite")
    db_path: str = Field(default="./data/eidolon.db")
    vector_dimension: int = Field(default=384, ge=100, le=2048)
    search: Dict[str, Union[int, float, bool]] = Field(default_factory=lambda: {
        "max_results": 50,
        "similarity_threshold": 0.7,
        "enable_semantic_search": True,
        "enable_keyword_search": True
    })


class InterfaceConfig(BaseModel):
    """Configuration for the Interface component."""
    
    cli: Dict[str, Union[str, bool]] = Field(default_factory=lambda: {
        "default_output_format": "table",
        "show_timestamps": True,
        "show_confidence_scores": True
    })
    web: Dict[str, Union[str, int, bool]] = Field(default_factory=lambda: {
        "enabled": False,
        "host": "localhost",
        "port": 8000,
        "debug": False
    })
    api: Dict[str, Union[int, bool]] = Field(default_factory=lambda: {
        "enabled": False,
        "rate_limit": 100,
        "auth_required": False
    })


class PrivacyConfig(BaseModel):
    """Configuration for privacy and security settings."""
    
    local_only_mode: bool = Field(default=False)
    auto_redaction: bool = Field(default=True)
    data_retention_days: int = Field(default=365, ge=1)
    encrypt_at_rest: bool = Field(default=True)
    encryption_key_path: str = Field(default="./data/.key")
    pause_on_sensitive_apps: bool = Field(default=True)
    require_confirmation_for_cloud: bool = Field(default=False)


class LoggingConfig(BaseModel):
    """Configuration for logging settings."""
    
    level: str = Field(default="INFO")
    file_path: str = Field(default="./logs/eidolon.log")
    max_file_size_mb: int = Field(default=10, ge=1)
    backup_count: int = Field(default=5, ge=1)
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class MonitoringConfig(BaseModel):
    """Configuration for performance monitoring."""
    
    enabled: bool = Field(default=True)
    metrics_collection_interval: int = Field(default=60, ge=10)
    alert_thresholds: Dict[str, float] = Field(default_factory=lambda: {
        "cpu_percent": 10.0,
        "memory_mb": 1000.0,
        "disk_usage_percent": 90.0
    })


class DevelopmentConfig(BaseModel):
    """Configuration for development and debugging."""
    
    debug_mode: bool = Field(default=False)
    mock_ai_responses: bool = Field(default=False)
    save_debug_screenshots: bool = Field(default=False)
    verbose_logging: bool = Field(default=False)


class Config(BaseModel):
    """Main configuration class for Eidolon."""
    
    observer: ObserverConfig = Field(default_factory=ObserverConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    interface: InterfaceConfig = Field(default_factory=InterfaceConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    development: DevelopmentConfig = Field(default_factory=DevelopmentConfig)
    
    @field_validator('observer', 'analysis', 'memory', 'interface', 'privacy', 
                    'logging', 'monitoring', 'development', mode='before')
    @classmethod
    def convert_dict_to_config(cls, v):
        """Convert dictionary values to appropriate config objects."""
        if isinstance(v, dict):
            return v
        return v

    def get_data_dir(self) -> Path:
        """Get the data directory path."""
        return Path(self.observer.storage_path).parent
    
    def get_logs_dir(self) -> Path:
        """Get the logs directory path."""
        return Path(self.logging.file_path).parent
    
    def ensure_directories(self, skip_readonly: bool = True) -> None:
        """Ensure all required directories exist."""
        directories = [
            self.get_data_dir(),
            self.get_logs_dir(),
            Path(self.observer.storage_path),
            Path(self.memory.db_path).parent,
        ]
        
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Ensured directory exists: {directory}")
            except (OSError, PermissionError) as e:
                if skip_readonly and ("Read-only file system" in str(e) or 
                                     "Permission denied" in str(e) or
                                     e.errno in (30, 13)):  # EROFS, EACCES
                    logger.warning(f"Skipping directory creation due to read-only filesystem: {directory}")
                    continue
                else:
                    logger.error(f"Failed to create directory {directory}: {e}")
                    raise


def substitute_env_vars(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively substitute environment variables in configuration."""
    
    def _substitute(obj):
        if isinstance(obj, dict):
            return {key: _substitute(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [_substitute(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            env_var = obj[2:-1]
            return os.getenv(env_var, obj)
        else:
            return obj
    
    return _substitute(config_dict)


def load_config(config_path: Optional[str] = None, skip_directory_creation: bool = None) -> Config:
    """
    Load configuration from YAML file with environment variable substitution.
    
    Args:
        config_path: Path to configuration file. If None, uses default locations.
        skip_directory_creation: Whether to skip directory creation. If None, auto-detects testing.
        
    Returns:
        Config: Validated configuration object.
        
    Raises:
        FileNotFoundError: If configuration file is not found and not in testing.
        ValueError: If configuration is invalid.
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Auto-detect testing environment if not specified
    if skip_directory_creation is None:
        skip_directory_creation = (
            os.getenv("PYTEST_CURRENT_TEST") is not None or
            os.getenv("TESTING") == "1" or
            "pytest" in os.getenv("_", "")
        )
    
    # Determine config file path
    if config_path is None:
        # Try multiple default locations
        possible_paths = [
            "./eidolon/config/settings.yaml",
            "./settings.yaml", 
            "~/.eidolon/settings.yaml",
            "/etc/eidolon/settings.yaml"
        ]
        
        config_path = None
        for path in possible_paths:
            expanded_path = Path(path).expanduser()
            if expanded_path.exists():
                config_path = str(expanded_path)
                break
        
        if config_path is None:
            logger.warning("No configuration file found, using defaults")
            config = Config()
            if not skip_directory_creation:
                config.ensure_directories()
            return config
    
    # Check if file exists before trying to open it
    if not Path(config_path).exists():
        if skip_directory_creation:
            # In testing, return default config for nonexistent files
            logger.warning(f"Configuration file not found (testing mode): {config_path}, using defaults")
            return Config()
        else:
            logger.error(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load and parse YAML
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        if config_dict is None:
            config_dict = {}
            
        # Substitute environment variables
        config_dict = substitute_env_vars(config_dict)
        
        # Create and validate config
        config = Config(**config_dict)
        
        # Ensure required directories exist (unless skipped)
        if not skip_directory_creation:
            config.ensure_directories()
        
        logger.info(f"Configuration loaded from: {config_path}")
        return config
        
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise ValueError(f"Invalid YAML configuration: {e}")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise ValueError(f"Configuration error: {e}")


def save_config(config: Config, config_path: str, skip_directory_creation: bool = None) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration object to save.
        config_path: Path where to save the configuration.
        skip_directory_creation: Whether to skip directory creation. Auto-detects testing if None.
    """
    try:
        # Convert config to dictionary
        config_dict = config.model_dump()
        
        # Auto-detect testing environment if not specified
        if skip_directory_creation is None:
            skip_directory_creation = (
                os.getenv("PYTEST_CURRENT_TEST") is not None or
                os.getenv("TESTING") == "1" or
                "pytest" in os.getenv("_", "")
            )
        
        # Ensure directory exists (unless skipped)
        if not skip_directory_creation:
            try:
                Path(config_path).parent.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError) as e:
                if "Read-only file system" in str(e) or "Permission denied" in str(e):
                    logger.warning(f"Skipping directory creation due to read-only filesystem: {Path(config_path).parent}")
                else:
                    raise
        
        # Save to YAML file
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to: {config_path}")
        
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        raise


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config(config_path: Optional[str] = None, skip_directory_creation: bool = None) -> Config:
    """Reload the global configuration instance."""
    global _config
    _config = load_config(config_path, skip_directory_creation)
    return _config