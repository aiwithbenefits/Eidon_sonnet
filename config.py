import os
import platform
from dotenv import load_dotenv
from typing import Optional
import logging

load_dotenv()

# --- Core Paths ---
APPDATA_FOLDER = os.path.expanduser(os.getenv("EIDON_APPDATA_FOLDER", "~/.eidon_backend"))
DB_PATH = os.path.join(APPDATA_FOLDER, "eidon_history.sqlite")
SCREENSHOTS_PATH = os.path.join(APPDATA_FOLDER, "screenshots")

# --- Performance & Resource Management ---
# Database connection pooling
DB_POOL_SIZE = int(os.getenv("EIDON_DB_POOL_SIZE", "5"))
DB_TIMEOUT = int(os.getenv("EIDON_DB_TIMEOUT", "30"))
DB_BUSY_TIMEOUT = int(os.getenv("EIDON_DB_BUSY_TIMEOUT", "10000"))

# Memory management
MAX_MEMORY_USAGE_MB = int(os.getenv("EIDON_MAX_MEMORY_MB", "1024"))
EMBEDDING_CACHE_SIZE = int(os.getenv("EIDON_EMBEDDING_CACHE_SIZE", "1000"))
IMAGE_PROCESSING_BATCH_SIZE = int(os.getenv("EIDON_IMAGE_BATCH_SIZE", "5"))

# --- Screenshotting & Capture Service ---
ENABLE_AUTOMATIC_SCREEN_CAPTURE = os.getenv("EIDON_AUTO_CAPTURE", "True").lower() == "true"
CAPTURE_INTERVAL_SECONDS = int(os.getenv("EIDON_CAPTURE_INTERVAL", 5))
IDLE_THRESHOLD_SECONDS = int(os.getenv("EIDON_IDLE_THRESHOLD", 15))
MIN_CAPTURE_INTERVAL_DURING_ACTIVITY_SECONDS = float(os.getenv("EIDON_MIN_CAPTURE_DURING_ACTIVITY", 0.5))
SIMILARITY_THRESHOLD = float(os.getenv("EIDON_SIMILARITY_THRESHOLD", 0.95))
WEBP_QUALITY = int(os.getenv("EIDON_WEBP_QUALITY", 80))
MAX_SCREENSHOT_WIDTH = int(os.getenv("EIDON_MAX_SCREENSHOT_WIDTH", 1920))
MAX_SCREENSHOT_HEIGHT = int(os.getenv("EIDON_MAX_SCREENSHOT_HEIGHT", 1200))
MULTI_MONITOR_CAPTURE_MODE = os.getenv("EIDON_MULTI_MONITOR_MODE", "active")

# Queue management
CAPTURE_QUEUE_SIZE = int(os.getenv("EIDON_CAPTURE_QUEUE_SIZE", "50"))
PROCESSING_QUEUE_SIZE = int(os.getenv("EIDON_PROCESSING_QUEUE_SIZE", "20"))

# --- OCR Configuration ---
ENABLE_OCR = os.getenv("ENABLE_OCR", "true").lower() == "true"
OCR_CONFIDENCE_THRESHOLD = float(os.getenv("EIDON_OCR_CONFIDENCE_THRESHOLD", "0.5"))
OCR_TEXT_MIN_LENGTH = int(os.getenv("EIDON_OCR_TEXT_MIN_LENGTH", "3"))

# --- Platform-specific Configuration ---
IS_MACOS = platform.system() == 'Darwin'
USE_APPLE_VISION = IS_MACOS and ENABLE_OCR

# --- Voyage AI ---
VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY")
VOYAGE_EMBED_MODEL = os.getenv("VOYAGE_EMBED_MODEL", "voyage-3.5")
VOYAGE_EMBEDDING_DIM = int(os.getenv("VOYAGE_EMBEDDING_DIM", 1024))
VOYAGE_API_TIMEOUT_SECONDS = int(os.getenv("VOYAGE_API_TIMEOUT", 60))
VOYAGE_API_RETRIES = int(os.getenv("VOYAGE_API_RETRIES", 3))
VOYAGE_BATCH_SIZE = int(os.getenv("VOYAGE_BATCH_SIZE", "10"))

# --- XAI (Grok) ---
XAI_API_KEY = os.environ.get("XAI_API_KEY")
XAI_MODEL_NAME = os.getenv("XAI_MODEL_NAME", "grok-3-latest")
XAI_API_BASE_URL = os.getenv("XAI_API_BASE_URL", "https://api.x.ai/v1")
XAI_API_TIMEOUT_SECONDS = int(os.getenv("XAI_API_TIMEOUT", 180))
XAI_API_RETRIES = int(os.getenv("XAI_API_RETRIES", "2"))

# --- Search & API ---
MAX_SEARCH_RESULTS_FOR_LLM_CONTEXT = int(os.getenv("EIDON_MAX_SEARCH_RESULTS_LLM", "7"))
MAX_RECENT_EMBEDDINGS_FOR_DEDUPLICATION = int(os.getenv("EIDON_MAX_RECENT_EMBEDDINGS_DEDUP", "10"))
LLM_MAX_TOOL_ITERATIONS = int(os.getenv("EIDON_LLM_MAX_TOOL_ITERATIONS", "3"))
LLM_CONVERSATION_HISTORY_MAX_TURNS = int(os.getenv("EIDON_LLM_CONVERSATION_HISTORY_MAX_TURNS", "10"))

# --- Logging & Debug ---
LOG_LEVEL = os.getenv("EIDON_LOG_LEVEL", "INFO").upper()
ENABLE_PERFORMANCE_MONITORING = os.getenv("EIDON_ENABLE_PERF_MONITORING", "false").lower() == "true"

# --- File Management ---
MAX_SCREENSHOT_AGE_DAYS = int(os.getenv("EIDON_MAX_SCREENSHOT_AGE_DAYS", "30"))
CLEANUP_INTERVAL_HOURS = int(os.getenv("EIDON_CLEANUP_INTERVAL_HOURS", "24"))
MAX_DISK_USAGE_GB = float(os.getenv("EIDON_MAX_DISK_USAGE_GB", "10.0"))

# --- Ensure directories exist ---
def ensure_dirs():
    """Create necessary directories with proper permissions."""
    for path in [APPDATA_FOLDER, SCREENSHOTS_PATH]:
        if path:
            try:
                os.makedirs(path, exist_ok=True, mode=0o755)
            except PermissionError as e:
                logging.error(f"Permission denied creating directory {path}: {e}")
                raise
            except Exception as e:
                logging.error(f"Error creating directory {path}: {e}")
                raise

ensure_dirs()

# --- Platform Specific ---
PLATFORM_SYSTEM = platform.system()

# --- Resource Limits ---
def get_memory_limit() -> int:
    """Get memory limit in bytes."""
    return MAX_MEMORY_USAGE_MB * 1024 * 1024

def get_disk_limit() -> int:
    """Get disk usage limit in bytes."""
    return int(MAX_DISK_USAGE_GB * 1024 * 1024 * 1024)