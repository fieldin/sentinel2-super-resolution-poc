"""
Utility functions for logging, retries, and file operations.
"""
import logging
import sys
import time
import os
import json
from pathlib import Path
from functools import wraps
from typing import Any, Callable, Optional, TypeVar
from datetime import datetime

# Type variable for generic retry decorator
T = TypeVar('T')


def setup_logging(name: str = "up42-poc", level: int = logging.INFO) -> logging.Logger:
    """
    Configure structured logging for the application.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exponential_base: Base for exponential backoff calculation
        exceptions: Tuple of exception types to catch
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            logger = setup_logging()
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(base_delay * (exponential_base ** attempt), max_delay)
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}: {e}")
            
            raise last_exception
        return wrapper
    return decorator


def ensure_directory(path: str | Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: str | Path) -> dict:
    """
    Read a JSON file.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Parsed JSON data as dictionary
    """
    with open(path, 'r') as f:
        return json.load(f)


def write_json(data: dict, path: str | Path, indent: int = 2) -> None:
    """
    Write data to a JSON file.
    
    Args:
        data: Data to write
        path: Output file path
        indent: JSON indentation level
    """
    path = Path(path)
    ensure_directory(path.parent)
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent, default=str)


def generate_timestamp() -> str:
    """
    Generate a timestamp string for file naming.
    
    Returns:
        Timestamp in format YYYYMMDD_HHMMSS
    """
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def get_file_size_mb(path: str | Path) -> float:
    """
    Get file size in megabytes.
    
    Args:
        path: File path
        
    Returns:
        File size in MB
    """
    return os.path.getsize(path) / (1024 * 1024)


def find_latest_file(directory: str | Path, pattern: str = "*.tif") -> Optional[Path]:
    """
    Find the most recently modified file matching a pattern.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern for matching files
        
    Returns:
        Path to the latest file, or None if no files found
    """
    directory = Path(directory)
    if not directory.exists():
        return None
    
    files = list(directory.glob(pattern))
    if not files:
        return None
    
    return max(files, key=lambda f: f.stat().st_mtime)


def find_latest_metadata(directory: str | Path) -> Optional[dict]:
    """
    Find and load the most recent metadata JSON file.
    
    Args:
        directory: Directory to search
        
    Returns:
        Metadata dictionary or None if not found
    """
    latest = find_latest_file(directory, "*_meta.json")
    if latest:
        return read_json(latest)
    return None

