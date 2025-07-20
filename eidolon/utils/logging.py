"""
Logging setup and management for Eidolon AI Personal Assistant

Provides structured logging with file rotation, different log levels,
and integration with the configuration system.
"""

import os
import logging
import logging.config
from pathlib import Path
from typing import Optional
import yaml
from .config import get_config


def setup_logging(
    config_path: Optional[str] = None,
    log_level: Optional[str] = None,
    log_file: Optional[str] = None
) -> None:
    """
    Set up logging configuration for Eidolon.
    
    Args:
        config_path: Path to logging configuration YAML file.
        log_level: Override log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Override log file path.
    """
    # Determine logging config file path
    if config_path is None:
        possible_paths = [
            "./eidolon/config/logging.yaml",
            "./logging.yaml",
            "~/.eidolon/logging.yaml"
        ]
        
        config_path = None
        for path in possible_paths:
            expanded_path = Path(path).expanduser()
            if expanded_path.exists():
                config_path = str(expanded_path)
                break
    
    # Load logging configuration
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                logging_config = yaml.safe_load(f)
            
            # Create logs directory if it doesn't exist
            logs_dir = Path("./logs")
            logs_dir.mkdir(exist_ok=True)
            
            # Apply configuration
            logging.config.dictConfig(logging_config)
            
        except Exception as e:
            # Fallback to basic configuration if loading fails
            _setup_basic_logging(log_level, log_file)
            logging.error(f"Failed to load logging configuration from {config_path}: {e}")
            return
    else:
        # Use basic configuration if no config file found
        _setup_basic_logging(log_level, log_file)
    
    # Override log level if specified
    if log_level:
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # Also update eidolon logger
        eidolon_logger = logging.getLogger("eidolon")
        eidolon_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Log successful setup
    logger = logging.getLogger(__name__)
    logger.info("Logging system initialized successfully")


def _setup_basic_logging(log_level: Optional[str] = None, log_file: Optional[str] = None) -> None:
    """Set up basic logging configuration as fallback."""
    
    # Get configuration
    try:
        config = get_config()
        if log_level is None:
            log_level = config.logging.level
        if log_file is None:
            log_file = config.logging.file_path
    except Exception:
        # Use hardcoded defaults if config fails
        log_level = log_level or "INFO"
        log_file = log_file or "./logs/eidolon.log"
    
    # Create logs directory
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Set up basic configuration
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name, typically __name__ of the calling module.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    return logging.getLogger(name)


def set_log_level(level: str, logger_name: Optional[str] = None) -> None:
    """
    Change log level dynamically.
    
    Args:
        level: New log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        logger_name: Specific logger name, or None for root logger.
    """
    try:
        log_level = getattr(logging, level.upper())
        if logger_name:
            logger = logging.getLogger(logger_name)
        else:
            logger = logging.getLogger()
        
        logger.setLevel(log_level)
        logging.info(f"Log level changed to {level.upper()} for {logger_name or 'root logger'}")
        
    except AttributeError:
        raise ValueError(f"Invalid log level: {level}")


def add_file_handler(
    logger_name: str,
    file_path: str,
    level: str = "INFO",
    format_string: Optional[str] = None
) -> None:
    """
    Add a file handler to a specific logger.
    
    Args:
        logger_name: Name of the logger to add handler to.
        file_path: Path to the log file.
        level: Log level for this handler.
        format_string: Custom format string for this handler.
    """
    logger = logging.getLogger(logger_name)
    
    # Create file handler
    handler = logging.FileHandler(file_path, encoding='utf-8')
    handler.setLevel(getattr(logging, level.upper()))
    
    # Set format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    logging.info(f"Added file handler for {logger_name}: {file_path}")


def remove_handlers(logger_name: str) -> None:
    """
    Remove all handlers from a logger.
    
    Args:
        logger_name: Name of the logger to remove handlers from.
    """
    logger = logging.getLogger(logger_name)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    
    logging.info(f"Removed all handlers from logger: {logger_name}")


class LoggerAdapter(logging.LoggerAdapter):
    """
    Enhanced logger adapter with additional context information.
    
    Automatically adds context like component name, user ID, session ID, etc.
    """
    
    def __init__(self, logger: logging.Logger, extra: dict):
        super().__init__(logger, extra)
    
    def process(self, msg, kwargs):
        """Add extra context to log messages."""
        # Add process ID and thread info
        extra = self.extra.copy()
        extra.update({
            'pid': os.getpid(),
            'thread': f"{kwargs.get('thread', 'main')}"
        })
        
        return f"[{extra.get('component', 'unknown')}] {msg}", kwargs


def get_component_logger(component_name: str, **extra_context) -> LoggerAdapter:
    """
    Get a logger adapter for a specific component with context.
    
    Args:
        component_name: Name of the component (e.g., 'observer', 'analyzer').
        **extra_context: Additional context to include in log messages.
        
    Returns:
        LoggerAdapter: Logger with component context.
    """
    logger = logging.getLogger(f"eidolon.{component_name}")
    
    context = {'component': component_name}
    context.update(extra_context)
    
    return LoggerAdapter(logger, context)


def log_performance(func):
    """
    Decorator to log function performance metrics.
    
    Logs function name, execution time, and memory usage.
    """
    import time
    import functools
    import psutil
    import os
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(f"eidolon.performance.{func.__module__}")
        
        # Get initial metrics
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_time = time.time()
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            
            # Calculate metrics
            end_time = time.time()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            # Log performance
            logger.debug(
                f"{func.__name__} executed in {execution_time:.3f}s, "
                f"memory delta: {memory_delta:+.2f}MB"
            )
            
            return result
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            
            logger.error(
                f"{func.__name__} failed after {execution_time:.3f}s: {str(e)}"
            )
            raise
    
    return wrapper


def log_exceptions(logger_name: Optional[str] = None):
    """
    Decorator to automatically log exceptions with full traceback.
    
    Args:
        logger_name: Specific logger to use, defaults to function's module.
    """
    def decorator(func):
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if logger_name:
                logger = logging.getLogger(logger_name)
            else:
                logger = logging.getLogger(func.__module__)
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception(
                    f"Exception in {func.__name__}: {str(e)}"
                )
                raise
        
        return wrapper
    return decorator