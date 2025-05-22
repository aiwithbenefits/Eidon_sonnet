import asyncio
import time
import logging
import psutil
import gc
from functools import wraps
from datetime import datetime
from typing import Optional, Dict, Any, Callable, Union
import httpx
import threading
from collections import defaultdict, deque

import config

logger = logging.getLogger(__name__)

# Performance monitoring
class PerformanceMonitor:
    def __init__(self):
        self._metrics = defaultdict(deque)
        self._lock = threading.Lock()
        self._start_time = time.time()

    def record_metric(self, name: str, value: float, max_history: int = 100):
        """Record a performance metric."""
        with self._lock:
            metrics = self._metrics[name]
            metrics.append((time.time(), value))
            while len(metrics) > max_history:
                metrics.popleft()

    def get_metrics(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        with self._lock:
            metrics = self._metrics[name]
            if not metrics:
                return {}

            values = [v for _, v in metrics]
            return {
                'count': len(values),
                'avg': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'latest': values[-1] if values else 0
            }

    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get all recorded metrics."""
        with self._lock:
            return {name: self.get_metrics(name) for name in self._metrics.keys()}

# Global performance monitor
_perf_monitor = PerformanceMonitor() if config.ENABLE_PERFORMANCE_MONITORING else None

def timestamp_ms_to_human_readable(ts_milliseconds: int) -> str:
    """Convert timestamp to human readable format with better error handling."""
    if ts_milliseconds is None:
        return "N/A"
    try:
        dt = datetime.fromtimestamp(ts_milliseconds / 1000.0)
        return dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    except (ValueError, OSError, OverflowError):
        return "Invalid Timestamp"

def get_current_timestamp_ms() -> int:
    """Get current timestamp in milliseconds."""
    return int(time.time() * 1000)

def performance_monitor(metric_name: str):
    """Decorator to monitor function performance."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not _perf_monitor:
                return await func(*args, **kwargs)

            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                _perf_monitor.record_metric(f"{metric_name}_duration", duration)
                _perf_monitor.record_metric(f"{metric_name}_success", 1)
                return result
            except Exception as e:
                duration = time.time() - start_time
                _perf_monitor.record_metric(f"{metric_name}_duration", duration)
                _perf_monitor.record_metric(f"{metric_name}_error", 1)
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not _perf_monitor:
                return func(*args, **kwargs)

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                _perf_monitor.record_metric(f"{metric_name}_duration", duration)
                _perf_monitor.record_metric(f"{metric_name}_success", 1)
                return result
            except Exception as e:
                duration = time.time() - start_time
                _perf_monitor.record_metric(f"{metric_name}_duration", duration)
                _perf_monitor.record_metric(f"{metric_name}_error", 1)
                raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

def async_retry(
    max_retries: int,
    delay_seconds: float = 1.0,
    backoff_multiplier: float = 2.0,
    max_delay: float = 60.0,
    allowed_exceptions=(httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError)
):
    """Enhanced retry decorator with exponential backoff and jitter."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):  # +1 for initial attempt
                try:
                    return await func(*args, **kwargs)
                except allowed_exceptions as e:
                    last_exception = e

                    # Don't retry on final attempt
                    if attempt == max_retries:
                        break

                    # Handle HTTP status errors
                    if isinstance(e, httpx.HTTPStatusError):
                        if not (500 <= e.response.status_code < 600 or e.response.status_code == 429):
                            logger.error(f"Non-retryable HTTP error in {func.__name__}: {e.response.status_code}")
                            raise

                        if e.response.status_code == 429:
                            retry_after = int(e.response.headers.get("Retry-After", delay_seconds))
                            delay = min(retry_after, max_delay)
                        else:
                            delay = min(delay_seconds * (backoff_multiplier ** attempt), max_delay)
                    else:
                        delay = min(delay_seconds * (backoff_multiplier ** attempt), max_delay)

                    # Add jitter to prevent thundering herd
                    import random
                    jitter = random.uniform(0.1, 0.3) * delay
                    actual_delay = delay + jitter

                    logger.warning(
                        f"Retryable error in {func.__name__} (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                        f"Retrying in {actual_delay:.2f}s..."
                    )
                    await asyncio.sleep(actual_delay)

                except Exception as e:
                    logger.error(f"Unexpected error in {func.__name__} attempt {attempt + 1}: {e}", exc_info=True)
                    last_exception = e

                    if attempt == max_retries:
                        break

                    delay = min(delay_seconds * (backoff_multiplier ** attempt), max_delay)
                    await asyncio.sleep(delay)

            logger.error(f"{func.__name__} failed after {max_retries + 1} attempts.")
            if last_exception:
                raise last_exception
            raise Exception(f"{func.__name__} failed after {max_retries + 1} attempts (unknown error).")

        return wrapper
    return decorator

def get_system_resources() -> Dict[str, Any]:
    """Get current system resource usage."""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            'memory_rss_mb': memory_info.rss / 1024 / 1024,
            'memory_vms_mb': memory_info.vms / 1024 / 1024,
            'memory_percent': process.memory_percent(),
            'cpu_percent': process.cpu_percent(),
            'num_threads': process.num_threads(),
            'num_fds': process.num_fds() if hasattr(process, 'num_fds') else None,
            'system_memory_percent': psutil.virtual_memory().percent,
            'system_cpu_percent': psutil.cpu_percent(),
            'disk_usage_percent': psutil.disk_usage(config.APPDATA_FOLDER).percent
        }
    except Exception as e:
        logger.error(f"Error getting system resources: {e}")
        return {}

def check_memory_usage() -> bool:
    """Check if memory usage is within limits."""
    try:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        limit_mb = config.get_memory_limit() / 1024 / 1024

        if memory_mb > limit_mb:
            logger.warning(f"Memory usage ({memory_mb:.1f}MB) exceeds limit ({limit_mb:.1f}MB)")
            return False
        return True
    except Exception as e:
        logger.error(f"Error checking memory usage: {e}")
        return True

def force_garbage_collection():
    """Force garbage collection and log memory stats."""
    if not config.ENABLE_PERFORMANCE_MONITORING:
        gc.collect()
        return

    before = get_system_resources()
    gc.collect()
    after = get_system_resources()

    memory_freed = before.get('memory_rss_mb', 0) - after.get('memory_rss_mb', 0)
    if memory_freed > 1:  # Only log if significant memory was freed
        logger.info(f"Garbage collection freed {memory_freed:.1f}MB")

def get_performance_metrics() -> Dict[str, Any]:
    """Get all performance metrics."""
    if not _perf_monitor:
        return {}

    metrics = _perf_monitor.get_all_metrics()
    metrics['system_resources'] = get_system_resources()
    metrics['uptime_seconds'] = time.time() - _perf_monitor._start_time

    return metrics

# Platform detection helpers
def is_macos() -> bool:
    """Check if running on macOS."""
    return config.PLATFORM_SYSTEM == "Darwin"

def is_windows() -> bool:
    """Check if running on Windows."""
    return config.PLATFORM_SYSTEM == "Windows"

def is_linux() -> bool:
    """Check if running on Linux."""
    return config.PLATFORM_SYSTEM == "Linux"

def get_platform_info() -> Dict[str, str]:
    """Get detailed platform information."""
    import platform
    return {
        'system': platform.system(),
        'release': platform.release(),
        'version': platform.version(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'python_version': platform.python_version()
    }

# File system utilities
def get_directory_size(path: str) -> int:
    """Get total size of directory in bytes."""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except (OSError, FileNotFoundError):
                    continue
    except Exception as e:
        logger.error(f"Error calculating directory size for {path}: {e}")
    return total_size

def cleanup_old_files(directory: str, max_age_days: int, pattern: str = "*") -> int:
    """Clean up old files in directory."""
    import glob
    import os
    from datetime import datetime, timedelta

    cutoff_time = datetime.now() - timedelta(days=max_age_days)
    cutoff_timestamp = cutoff_time.timestamp()

    pattern_path = os.path.join(directory, pattern)
    deleted_count = 0

    try:
        for filepath in glob.glob(pattern_path):
            try:
                if os.path.getmtime(filepath) < cutoff_timestamp:
                    os.remove(filepath)
                    deleted_count += 1
            except (OSError, FileNotFoundError):
                continue
    except Exception as e:
        logger.error(f"Error during file cleanup in {directory}: {e}")

    return deleted_count

# Context managers for resource management
class ResourceMonitor:
    """Context manager for monitoring resource usage."""

    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_resources = None
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        self.start_resources = get_system_resources()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not config.ENABLE_PERFORMANCE_MONITORING:
            return

        duration = time.time() - self.start_time
        end_resources = get_system_resources()

        memory_delta = (end_resources.get('memory_rss_mb', 0) -
                       self.start_resources.get('memory_rss_mb', 0))

        logger.info(
            f"Operation '{self.operation_name}' completed in {duration:.2f}s, "
            f"memory delta: {memory_delta:+.1f}MB"
        )

        if _perf_monitor:
            _perf_monitor.record_metric(f"{self.operation_name}_duration", duration)
            _perf_monitor.record_metric(f"{self.operation_name}_memory_delta", memory_delta)

# Async context manager for database operations
class AsyncResourceMonitor:
    """Async context manager for monitoring resource usage."""

    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_resources = None
        self.start_time = None

    async def __aenter__(self):
        self.start_time = time.time()
        self.start_resources = get_system_resources()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if not config.ENABLE_PERFORMANCE_MONITORING:
            return

        duration = time.time() - self.start_time
        end_resources = get_system_resources()

        memory_delta = (end_resources.get('memory_rss_mb', 0) -
                       self.start_resources.get('memory_rss_mb', 0))

        logger.debug(
            f"Async operation '{self.operation_name}' completed in {duration:.2f}s, "
            f"memory delta: {memory_delta:+.1f}MB"
        )

        if _perf_monitor:
            _perf_monitor.record_metric(f"{self.operation_name}_duration", duration)
            _perf_monitor.record_metric(f"{self.operation_name}_memory_delta", memory_delta)