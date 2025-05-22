import asyncio
import io
import json
import logging
import os
import platform
import time
import uuid
import gc
import psutil
import weakref
from typing import List, Optional, Dict, Any, Union, Set
from collections import defaultdict
from functools import lru_cache
from contextlib import asynccontextmanager
import base64
import aiosqlite
import sqlite3

# Enhanced imports for performance monitoring and resource management
try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False
    logging.warning("aiofiles library not found. /api/screenshot/{filename} endpoint will be disabled.")

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Request, BackgroundTasks
from fastapi.responses import JSONResponse, Response as FastAPIResponse
from fastapi.exception_handlers import http_exception_handler as fastapi_http_exception_handler
from pydantic import BaseModel, Field, ValidationError
from PIL import Image, UnidentifiedImageError
import numpy as np
import httpx

# Eidon Core Imports
from screen_capture_handler import ScreenCaptureService, PYNPUT_AVAILABLE, MACOS_LIBS_AVAILABLE
import config
from database import (
    create_db as init_database,
    insert_entry,
    get_all_entries,
    get_entry_by_timestamp,
    get_recent_entries_with_embeddings,
    count_entries,
    Entry as DbEntry,
    delete_entry_by_id as db_delete_entry_by_id,
    get_entry_by_filename as db_get_entry_by_filename,
    _row_to_entry as db_row_to_entry
)
from nlp import (
    get_text_embedding,
    get_query_embedding,
    cosine_similarity,
)

# Import the appropriate OCR processor based on platform
if platform.system() == 'Darwin':
    from vision_ocr import (
        extract_text_from_image as extract_text_from_image_vision,
        convert_pil_to_bytes as convert_image_to_target_format_bytes
    )
    extract_text_from_image = extract_text_from_image_vision
else:
    async def extract_text_from_image(image_data: bytes, input_mime_type: str = "image/png") -> str:
        logger.warning("Apple Vision not available on this platform - no OCR will be performed")
        return ""

    from io import BytesIO
    def convert_image_to_target_format_bytes(image: Image.Image, format: str = "WEBP", quality: int = 80) -> bytes:
        img_byte_arr = BytesIO()
        save_kwargs = {"format": format.upper()}
        if format.upper() in ["WEBP", "JPEG"]:
            save_kwargs["quality"] = quality
        image.save(img_byte_arr, **save_kwargs)
        return img_byte_arr.getvalue()

from utils import (
    timestamp_ms_to_human_readable,
    get_current_timestamp_ms,
    async_retry,
    is_macos
)

# --- Enhanced Resource Management Classes ---

class MemoryManager:
    """Enhanced memory management and monitoring."""

    def __init__(self):
        self.memory_threshold_mb = 1024  # 1GB threshold
        self.gc_interval = 300  # 5 minutes
        self.last_gc_time = time.time()
        self.memory_stats = defaultdict(int)
        self._weak_refs: Set[weakref.ref] = set()

    def register_object(self, obj):
        """Register object for weak reference tracking."""
        try:
            ref = weakref.ref(obj, self._cleanup_ref)
            self._weak_refs.add(ref)
        except TypeError:
            pass  # Object doesn't support weak references

    def _cleanup_ref(self, ref):
        """Clean up dead weak references."""
        self._weak_refs.discard(ref)

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "percent": process.memory_percent(),
                "available_mb": psutil.virtual_memory().available / 1024 / 1024
            }
        except Exception as e:
            logger.warning(f"Failed to get memory stats: {e}")
            return {}

    async def periodic_cleanup(self):
        """Perform periodic memory cleanup."""
        current_time = time.time()
        if current_time - self.last_gc_time > self.gc_interval:
            memory_before = self.get_memory_usage()

            # Clean up dead weak references
            dead_refs = [ref for ref in self._weak_refs if ref() is None]
            for ref in dead_refs:
                self._weak_refs.discard(ref)

            # Force garbage collection
            collected = gc.collect()

            memory_after = self.get_memory_usage()
            self.last_gc_time = current_time

            if collected > 0:
                logger.debug(f"GC collected {collected} objects. "
                           f"Memory: {memory_before.get('rss_mb', 0):.1f}MB → {memory_after.get('rss_mb', 0):.1f}MB")

    def should_cleanup(self) -> bool:
        """Check if cleanup is needed based on memory usage."""
        memory_stats = self.get_memory_usage()
        return memory_stats.get('rss_mb', 0) > self.memory_threshold_mb


class DatabasePool:
    """Database connection pool manager."""

    def __init__(self, max_connections: int = 10, timeout: float = 30.0):
        self.max_connections = max_connections
        self.timeout = timeout
        self._pool: List[aiosqlite.Connection] = []
        self._in_use: Set[aiosqlite.Connection] = set()
        self._lock = asyncio.Lock()
        self._created_connections = 0

    async def get_connection(self) -> aiosqlite.Connection:
        """Get a connection from the pool."""
        async with self._lock:
            # Try to reuse existing connection
            if self._pool:
                conn = self._pool.pop()
                self._in_use.add(conn)
                return conn

            # Create new connection if under limit
            if self._created_connections < self.max_connections:
                conn = await aiosqlite.connect(config.DB_PATH, timeout=self.timeout)
                conn.row_factory = sqlite3.Row
                self._created_connections += 1
                self._in_use.add(conn)
                return conn

            # Wait for a connection to become available
            raise Exception("Connection pool exhausted")

    async def return_connection(self, conn: aiosqlite.Connection):
        """Return a connection to the pool."""
        async with self._lock:
            if conn in self._in_use:
                self._in_use.remove(conn)
                # Verify connection is still valid
                try:
                    await conn.execute("SELECT 1")
                    self._pool.append(conn)
                except Exception:
                    # Connection is bad, close it
                    await conn.close()
                    self._created_connections -= 1

    async def close_all(self):
        """Close all connections in the pool."""
        async with self._lock:
            # Close pooled connections
            for conn in self._pool:
                try:
                    await conn.close()
                except Exception:
                    pass

            # Close in-use connections
            for conn in self._in_use:
                try:
                    await conn.close()
                except Exception:
                    pass

            self._pool.clear()
            self._in_use.clear()
            self._created_connections = 0


class HTTPClientManager:
    """Optimized HTTP client with connection pooling."""

    def __init__(self):
        self._clients: Dict[str, httpx.AsyncClient] = {}
        self._client_usage: Dict[str, int] = defaultdict(int)
        self._max_clients = 5
        self._timeout = httpx.Timeout(30.0, connect=10.0)

        # Connection limits for better resource management
        self._limits = httpx.Limits(
            max_keepalive_connections=10,
            max_connections=20,
            keepalive_expiry=30.0
        )

    async def get_client(self, base_url: str = None) -> httpx.AsyncClient:
        """Get or create an HTTP client for the given base URL."""
        client_key = base_url or "default"

        if client_key not in self._clients:
            # Clean up least used client if we hit the limit
            if len(self._clients) >= self._max_clients:
                least_used = min(self._client_usage.items(), key=lambda x: x[1])
                await self._cleanup_client(least_used[0])

            # Create new client
            client_config = {
                "timeout": self._timeout,
                "limits": self._limits,
                "follow_redirects": True
            }
            if base_url:
                client_config["base_url"] = base_url

            self._clients[client_key] = httpx.AsyncClient(**client_config)

        self._client_usage[client_key] += 1
        return self._clients[client_key]

    async def _cleanup_client(self, client_key: str):
        """Clean up a specific client."""
        if client_key in self._clients:
            await self._clients[client_key].aclose()
            del self._clients[client_key]
            del self._client_usage[client_key]

    async def cleanup_all(self):
        """Clean up all HTTP clients."""
        for client in self._clients.values():
            try:
                await client.aclose()
            except Exception as e:
                logger.warning(f"Error closing HTTP client: {e}")

        self._clients.clear()
        self._client_usage.clear()

# --- Performance Optimization Utilities ---

@lru_cache(maxsize=1000)
def get_mime_type_cached(filename: str) -> str:
    """Cached MIME type detection."""
    if filename.lower().endswith('.webp'):
        return 'image/webp'
    elif filename.lower().endswith('.png'):
        return 'image/png'
    elif filename.lower().endswith(('.jpg', '.jpeg')):
        return 'image/jpeg'
    else:
        return 'application/octet-stream'


class ConversationManager:
    """Enhanced conversation management with TTL and memory optimization."""

    def __init__(self, max_conversations: int = 100, ttl_seconds: int = 3600):
        self.max_conversations = max_conversations
        self.ttl_seconds = ttl_seconds
        self.conversations: Dict[str, Dict] = {}
        self.access_times: Dict[str, float] = {}

    def get_conversation(self, session_id: str) -> List[Dict[str, str]]:
        """Get conversation history for a session."""
        current_time = time.time()

        # Clean up expired conversations
        self._cleanup_expired(current_time)

        if session_id not in self.conversations:
            self.conversations[session_id] = {"messages": [], "created": current_time}

        self.access_times[session_id] = current_time
        return self.conversations[session_id]["messages"]

    def update_conversation(self, session_id: str, messages: List[Dict[str, str]]):
        """Update conversation history."""
        current_time = time.time()
        if session_id in self.conversations:
            self.conversations[session_id]["messages"] = messages
            self.access_times[session_id] = current_time

    def _cleanup_expired(self, current_time: float):
        """Clean up expired conversations."""
        expired_sessions = []

        for session_id, access_time in self.access_times.items():
            if current_time - access_time > self.ttl_seconds:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            self.conversations.pop(session_id, None)
            self.access_times.pop(session_id, None)

        # Also enforce max conversations limit
        if len(self.conversations) > self.max_conversations:
            # Remove oldest conversations
            sorted_sessions = sorted(self.access_times.items(), key=lambda x: x[1])
            excess_count = len(self.conversations) - self.max_conversations

            for session_id, _ in sorted_sessions[:excess_count]:
                self.conversations.pop(session_id, None)
                self.access_times.pop(session_id, None)

# --- Global Resource Managers ---
memory_manager = MemoryManager()
db_pool = DatabasePool()
http_client_manager = HTTPClientManager()
conversation_manager = ConversationManager()

# --- Logging Setup ---
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(process)d|%(threadName)s] - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
    format=LOG_FORMAT,
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Reduce verbosity of external libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("pynput").setLevel(logging.WARNING)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.INFO)
logging.getLogger("multipart").setLevel(logging.INFO)

logger = logging.getLogger("eidon_backend.app")

# --- Global State & Services ---
capture_data_queue = asyncio.Queue(maxsize=config.CAPTURE_INTERVAL_SECONDS * 10)
screen_capture_service_instance: Optional[ScreenCaptureService] = None
active_background_tasks = set()

# --- Enhanced Pydantic Models ---

class StandardResponse(BaseModel):
    """Standard response wrapper for all API endpoints."""
    success: bool = True
    message: str = ""
    data: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    timestamp: int = Field(default_factory=get_current_timestamp_ms)

class ErrorDetail(BaseModel):
    """Detailed error information."""
    error_type: str
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: int = Field(default_factory=get_current_timestamp_ms)

class CaptureDataPayload(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image string (PNG or JPEG preferred).")
    app_name: Optional[str] = Field(None, description="Name of the application.")
    window_title: Optional[str] = Field(None, description="Title of the active window.")
    page_url: Optional[str] = Field(None, description="URL if the active window is a browser.")
    timestamp_ms: int = Field(default_factory=get_current_timestamp_ms, description="Timestamp in milliseconds (UTC).")

class EntryResponseModel(BaseModel):
    id: Optional[int]
    app: Optional[str]
    title: Optional[str]
    text: Optional[str] = Field(None, description="OCR'd text. Can be null or empty.")
    timestamp: int
    human_readable_timestamp: str
    filename: Optional[str]
    page_url: Optional[str]
    relevance_score: Optional[float] = Field(None, description="Similarity score for search results.")
    embedding_present: bool = Field(description="Whether a text embedding exists for this entry.")
    embedding_dim: Optional[int] = Field(None, description="Dimension of the embedding if present.")

    @classmethod
    def from_db_entry(cls, db_entry: DbEntry, include_text: bool = True):
        text_to_show = db_entry.text
        if not include_text and db_entry.text:
            text_to_show = (db_entry.text[:100] + "...") if len(db_entry.text) > 100 else db_entry.text

        return cls(
            id=db_entry.id,
            app=db_entry.app,
            title=db_entry.title,
            text=text_to_show,
            timestamp=db_entry.timestamp,
            human_readable_timestamp=timestamp_ms_to_human_readable(db_entry.timestamp),
            filename=db_entry.filename,
            page_url=db_entry.page_url,
            relevance_score=db_entry.relevance_score,
            embedding_present=db_entry.embedding.size > 0,
            embedding_dim=db_entry.embedding.shape[0] if db_entry.embedding.size > 0 else None
        )

class SearchQueryPayload(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language search query.")
    limit: int = Field(default=config.MAX_SEARCH_RESULTS_FOR_LLM_CONTEXT, gt=0, le=50, description="Max items for LLM context.")

class SearchToolResultItemModel(BaseModel):
    timestamp: int
    app: Optional[str]
    title: Optional[str]
    page_url: Optional[str]
    text_snippet: str
    relevance_score: Optional[float]
    filename: Optional[str]

class SearchResponseModel(BaseModel):
    query_received: str
    llm_final_answer: Optional[str] = Field(None, description="LLM's summarized answer to the user.")
    retrieved_entries: List[EntryResponseModel] = Field(default=[], description="List of relevant entries found and presented by the LLM.")

class StatusResponseModel(BaseModel):
    status: str
    total_entries_in_db: int
    voyage_api_configured: bool
    xai_api_configured: bool
    automatic_capture_enabled_in_config: bool
    capture_service_running: bool
    capture_queue_size: int
    platform: str
    active_processing_tasks: int
    memory_usage: Optional[Dict[str, float]] = None
    performance_stats: Optional[Dict[str, Any]] = None

# --- Enhanced Error Handling ---

def create_error_response(
    error_type: str,
    message: str,
    status_code: int = 500,
    error_code: str = None,
    details: Dict[str, Any] = None
) -> JSONResponse:
    """Create standardized error response."""
    error_detail = ErrorDetail(
        error_type=error_type,
        error_code=error_code,
        details=details or {}
    )

    response_data = StandardResponse(
        success=False,
        message=message,
        error=error_detail.dict()
    )

    return JSONResponse(
        status_code=status_code,
        content=response_data.dict()
    )

def create_success_response(
    data: Any = None,
    message: str = "Operation completed successfully",
    status_code: int = 200
) -> JSONResponse:
    """Create standardized success response."""
    response_data = StandardResponse(
        success=True,
        message=message,
        data=data
    )

    return JSONResponse(
        status_code=status_code,
        content=response_data.dict()
    )

# --- FastAPI Lifespan Handler ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    logger.info("Eidon Backend API starting up...")

    try:
        await init_database()
    except Exception as e:
        logger.critical(f"DATABASE INITIALIZATION FAILED: {e}. Eidon will not function correctly.", exc_info=True)
        yield
        return

    global screen_capture_service_instance
    if config.ENABLE_AUTOMATIC_SCREEN_CAPTURE:
        if not PYNPUT_AVAILABLE and (is_macos() or platform.system() in ['Windows', 'Linux']):
            logger.warning("pynput is not available, idle detection for automatic capture will be disabled.")
        if not MACOS_LIBS_AVAILABLE and is_macos():
            logger.warning("macOS specific libraries not found. Metadata capture will be limited.")

        screen_capture_service_instance = ScreenCaptureService(capture_data_queue)
        if not await screen_capture_service_instance.start():
            logger.error("Screen capture service failed to start.")
    else:
        logger.info("Automatic screen capture is DISABLED by server configuration.")

    # Start background tasks
    queue_processor_task = asyncio.create_task(capture_queue_processor_bg_task())
    active_background_tasks.add(queue_processor_task)
    queue_processor_task.add_done_callback(active_background_tasks.discard)

    # Start periodic cleanup task
    cleanup_task = asyncio.create_task(periodic_cleanup_task())
    active_background_tasks.add(cleanup_task)
    cleanup_task.add_done_callback(active_background_tasks.discard)

    logger.info("Eidon Backend API startup complete.")

    yield

    # Shutdown code
    logger.info("Eidon Backend API shutting down...")

    if screen_capture_service_instance:
        logger.info("Stopping screen capture service...")
        await screen_capture_service_instance.stop()

    logger.info("Signaling capture queue processor to stop...")
    await capture_data_queue.put(None)

    # Wait for background tasks to complete
    if active_background_tasks:
        logger.info(f"Waiting for {len(active_background_tasks)} background tasks to complete...")
        try:
            await asyncio.wait_for(
                asyncio.gather(*active_background_tasks, return_exceptions=True),
                timeout=15.0
            )
        except asyncio.TimeoutError:
            logger.warning("Timed out waiting for background tasks to complete.")

    # Cleanup resources
    await db_pool.close_all()
    await http_client_manager.cleanup_all()

    logger.info("Eidon Backend API shutdown complete.")

# --- FastAPI App Initialization ---

app = FastAPI(
    title="Eidon Backend API",
    version="0.3.0",
    description="Enhanced backend service for capturing, processing, and searching personal digital activity.",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# --- Enhanced Exception Handler ---

@app.exception_handler(Exception)
async def enhanced_exception_handler(request: Request, exc: Exception):
    """Enhanced exception handler with detailed error responses."""
    logger.error(f"Unhandled exception for request {request.method} {request.url}: {exc}", exc_info=True)

    if isinstance(exc, HTTPException):
        return create_error_response(
            error_type="HTTPException",
            message=exc.detail,
            status_code=exc.status_code,
            error_code=f"HTTP_{exc.status_code}"
        )
    elif isinstance(exc, ValidationError):
        return create_error_response(
            error_type="ValidationError",
            message="Request validation failed",
            status_code=422,
            error_code="VALIDATION_FAILED",
            details={"validation_errors": exc.errors()}
        )
    else:
        return create_error_response(
            error_type=type(exc).__name__,
            message="An unexpected internal server error occurred",
            status_code=500,
            error_code="INTERNAL_ERROR"
        )

# --- Background Tasks ---

async def periodic_cleanup_task():
    """Periodic cleanup task for memory and resource management."""
    logger.info("Periodic cleanup task started.")

    while True:
        try:
            await asyncio.sleep(300)  # Run every 5 minutes

            # Perform memory cleanup
            await memory_manager.periodic_cleanup()

            # Log performance stats
            memory_stats = memory_manager.get_memory_usage()
            if memory_stats:
                logger.debug(f"Memory usage: {memory_stats['rss_mb']:.1f}MB "
                           f"({memory_stats['percent']:.1f}%)")

            # Clean up conversations
            conversation_manager._cleanup_expired(time.time())

        except asyncio.CancelledError:
            logger.info("Periodic cleanup task was cancelled.")
            break
        except Exception as e:
            logger.error(f"Error in periodic cleanup task: {e}", exc_info=True)

async def capture_queue_processor_bg_task():
    """Enhanced capture queue processor with better error handling."""
    logger.info("Capture data queue processor background task started.")

    while True:
        try:
            capture_event = await capture_data_queue.get()
            if capture_event is None:
                logger.info("Capture queue processor received shutdown sentinel. Exiting.")
                capture_data_queue.task_done()
                break

            # Register for memory tracking
            memory_manager.register_object(capture_event)

            # Process the event
            task = asyncio.create_task(process_single_capture_event(capture_event))
            active_background_tasks.add(task)
            task.add_done_callback(active_background_tasks.discard)

            capture_data_queue.task_done()

            # Perform cleanup if needed
            if memory_manager.should_cleanup():
                await memory_manager.periodic_cleanup()

        except asyncio.CancelledError:
            logger.info("Capture queue processor task was cancelled.")
            break
        except Exception as e:
            logger.critical(f"CRITICAL ERROR in capture queue processor: {e}", exc_info=True)
            await asyncio.sleep(10)

# --- Enhanced Core Processing Logic ---

async def process_single_capture_event(capture_event: Dict[str, Any]) -> Optional[DbEntry]:
    """Enhanced processing with better resource management."""
    timestamp_ms = capture_event.get("timestamp_ms", get_current_timestamp_ms())
    image_bytes = capture_event.get("image_bytes")
    app_name = capture_event.get("app_name")
    window_title = capture_event.get("window_title")
    page_url = capture_event.get("page_url")
    source = capture_event.get("source", "unknown_source")

    if not image_bytes:
        logger.warning(f"No image_bytes in capture_event for ts {timestamp_ms} from {source}.")
        return None

    filename_base = f"{timestamp_ms}_{uuid.uuid4().hex[:6]}"
    filename_webp = f"{filename_base}.webp"
    filepath_webp = os.path.join(config.SCREENSHOTS_PATH, filename_webp)

    pil_image = None
    try:
        # Process image with proper resource management
        pil_image = Image.open(io.BytesIO(image_bytes))
        memory_manager.register_object(pil_image)

        webp_bytes_for_storage = convert_image_to_target_format_bytes(
            pil_image, "WEBP", config.WEBP_QUALITY
        )

        if not webp_bytes_for_storage:
            logger.error(f"Failed to convert image to WEBP for {filename_webp}.")
            return None

        # Save file
        with open(filepath_webp, "wb") as f:
            f.write(webp_bytes_for_storage)

    except UnidentifiedImageError:
        logger.error(f"Unidentified image format for capture ts {timestamp_ms}.")
        return None
    except Exception as e:
        logger.error(f"Error processing image {filename_webp}: {e}", exc_info=True)
        return None
    finally:
        if pil_image:
            pil_image.close()

    # OCR processing
    ocr_text = ""
    if config.ENABLE_OCR and config.USE_APPLE_VISION:
        try:
            logger.debug("Extracting text using Apple Vision...")
            ocr_text = await extract_text_from_image(
                image_data=image_bytes,
                input_mime_type="image/png"
            )
            logger.debug(f"Extracted {len(ocr_text)} characters of text")
        except Exception as e:
            logger.error(f"Error during OCR: {e}", exc_info=True)
            ocr_text = ""

    # Text embedding
    text_emb = np.array([], dtype=np.float32)
    if ocr_text.strip() and config.VOYAGE_EMBED_MODEL and config.VOYAGE_API_KEY:
        try:
            text_emb = await get_text_embedding(ocr_text)
        except Exception as e:
            logger.error(f"Text embedding failed: {e}", exc_info=True)

    # Similarity check with enhanced database connection handling
    if text_emb.size > 0 and config.SIMILARITY_THRESHOLD > 0:
        conn = await db_pool.get_connection()
        try:
            recent_db_entries = await get_recent_entries_with_embeddings(
                limit=config.MAX_RECENT_EMBEDDINGS_FOR_DEDUPLICATION
            )

            for recent_entry in recent_db_entries:
                if recent_entry.embedding.size > 0:
                    similarity = cosine_similarity(text_emb, recent_entry.embedding)
                    if similarity > config.SIMILARITY_THRESHOLD:
                        logger.info(f"DUPLICATE DETECTED: similarity {similarity:.4f}")
                        try:
                            os.remove(filepath_webp)
                        except Exception as e_rm:
                            logger.error(f"Failed to remove duplicate file: {e_rm}")
                        return None
        finally:
            await db_pool.return_connection(conn)

    # Database insertion
    inserted_id = await insert_entry(
        timestamp=timestamp_ms,
        text=ocr_text,
        embedding=text_emb if text_emb.size > 0 else None,
        app=app_name,
        title=window_title,
        filename=filename_webp,
        page_url=page_url
    )

    if inserted_id is not None:
        logger.info(f"PROCESSED & STORED Entry ID {inserted_id}, ts {timestamp_ms}")
        return DbEntry(
            id=inserted_id,
            app=app_name,
            title=window_title,
            text=ocr_text,
            timestamp=timestamp_ms,
            embedding=text_emb,
            filename=filename_webp,
            page_url=page_url
        )
    else:
        logger.warning(f"DB insert FAILED for ts {timestamp_ms}")
        return None

# --- Enhanced XAI LLM Integration ---

XAI_SYSTEM_PROMPT = (
    "You are Eidon, an advanced AI assistant integrated into a personal digital history system. "
    "Your primary function is to help users search, analyze, and understand their recorded activities. "
    "You have access to a powerful search tool to query the database of activities. "
    "Provide accurate, helpful, and concise responses based on the retrieved information."
)

async def search_eidon_database_tool_impl(
    keywords: Optional[str] = None,
    app_filter: Optional[str] = None,
    title_filter: Optional[str] = None,
    url_filter: Optional[str] = None,
    limit: int = config.MAX_SEARCH_RESULTS_FOR_LLM_CONTEXT,
    after_timestamp: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Enhanced database search with connection pooling."""
    logger.info(f"Searching database with: keywords={keywords}, app={app_filter}")

    conn = await db_pool.get_connection()
    try:
        query = """
            SELECT id, app, title, text, timestamp, embedding, filename, page_url
            FROM entries
            WHERE 1=1
        """
        params = []

        if keywords:
            query += " AND (text LIKE ? OR title LIKE ? OR app LIKE ?)"
            search_term = f"%{keywords}%"
            params.extend([search_term] * 3)

        if app_filter:
            query += " AND app LIKE ?"
            params.append(f"%{app_filter}%")

        if title_filter:
            query += " AND title LIKE ?"
            params.append(f"%{title_filter}%")

        if url_filter:
            query += " AND page_url LIKE ?"
            params.append(f"%{url_filter}%")

        if after_timestamp is not None:
            query += " AND timestamp >= ?"
            params.append(after_timestamp)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor = await conn.execute(query, params)
        rows = await cursor.fetchall()

        # Convert to formatted results
        result = []
        for row in rows:
            entry = await db_row_to_entry(row, load_embeddings=False)
            if entry:
                result.append({
                    "timestamp": entry.timestamp,
                    "app": entry.app,
                    "title": entry.title,
                    "page_url": entry.page_url,
                    "text_snippet": (entry.text or "")[:200] + ("..." if entry.text and len(entry.text) > 200 else ""),
                    "relevance_score": 0.9,
                    "filename": entry.filename
                })

        return result

    except Exception as e:
        logger.error(f"Error searching database: {e}", exc_info=True)
        return []
    finally:
        await db_pool.return_connection(conn)

XAI_TOOLS_AVAILABLE = [
    {
        "type": "function",
        "function": {
            "name": "search_eidon_database_tool_impl",
            "description": "Search the user's digital history for activities, documents, or web pages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {"type": "string", "description": "Keywords for semantic search"},
                    "app_filter": {"type": "string", "description": "Filter by application name"},
                    "title_filter": {"type": "string", "description": "Filter by window title"},
                    "url_filter": {"type": "string", "description": "Filter by URL"},
                    "limit": {"type": "integer", "description": "Max results to retrieve"}
                },
                "required": []
            }
        }
    }
]

AVAILABLE_TOOL_FUNCTIONS_MAP = {"search_eidon_database_tool_impl": search_eidon_database_tool_impl}

@async_retry(max_retries=config.XAI_API_RETRIES, delay_seconds=3.0)
async def call_xai_llm_with_tools_robust(
    messages: List[Dict[str, str]],
    tools: Optional[List[Dict]] = None,
    tool_choice: Optional[Union[str, Dict]] = "auto"
) -> Dict:
    """Enhanced XAI API call with connection pooling."""
    if not config.XAI_API_KEY or not config.XAI_MODEL_NAME:
        raise HTTPException(status_code=503, detail="XAI LLM service not configured")

    headers = {
        "Authorization": f"Bearer {config.XAI_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    payload = {
        "model": config.XAI_MODEL_NAME,
        "messages": messages,
        "temperature": 0.6
    }

    if tools:
        payload["tools"] = tools
    if tool_choice:
        payload["tool_choice"] = tool_choice

    client = await http_client_manager.get_client(config.XAI_API_BASE_URL)

    try:
        response = await client.post("/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"XAI API HTTP error: {e.response.status_code}")
        if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
            raise HTTPException(status_code=e.response.status_code, detail=f"XAI API error: {e.response.text}")
        raise
    except Exception as e:
        logger.error(f"XAI API error: {e}", exc_info=True)
        raise

# --- Enhanced API Endpoints ---

@app.post("/api/capture/base64", summary="Submit Base64 Encoded Image")
async def api_submit_capture_base64(payload: CaptureDataPayload):
    """Enhanced base64 capture endpoint with proper response handling."""
    try:
        image_bytes = base64.b64decode(payload.image_base64)
        if not image_bytes:
            raise ValueError("Empty image data")

        # Validate image
        try:
            img_v = Image.open(io.BytesIO(image_bytes))
            img_v.verify()
            img_v.close()
        except Exception as img_e:
            return create_error_response(
                error_type="ValidationError",
                message=f"Invalid image data: {str(img_e)}",
                status_code=400,
                error_code="INVALID_IMAGE"
            )

    except Exception as e:
        return create_error_response(
            error_type="ValidationError",
            message=f"Invalid base64 data: {str(e)}",
            status_code=400,
            error_code="INVALID_BASE64"
        )

    capture_event = {
        "timestamp_ms": payload.timestamp_ms,
        "image_bytes": image_bytes,
        "app_name": payload.app_name,
        "window_title": payload.window_title,
        "page_url": payload.page_url,
        "source": "api_base64"
    }

    try:
        capture_data_queue.put_nowait(capture_event)
        return create_success_response(
            data={"timestamp": payload.timestamp_ms},
            message="Capture data accepted for processing",
            status_code=202
        )
    except asyncio.QueueFull:
        return create_error_response(
            error_type="ServiceError",
            message="Capture queue is full",
            status_code=503,
            error_code="QUEUE_FULL"
        )

@app.post("/api/capture/upload", summary="Upload Image File")
async def api_submit_capture_upload(
    image_file: UploadFile = File(...),
    app_name: Optional[str] = Form(None),
    window_title: Optional[str] = Form(None),
    page_url: Optional[str] = Form(None),
    timestamp_ms: Optional[int] = Form(None),
):
    """Enhanced upload endpoint with proper response handling."""
    ts_ms = timestamp_ms if timestamp_ms is not None else get_current_timestamp_ms()

    if image_file.size > 20 * 1024 * 1024:
        return create_error_response(
            error_type="ValidationError",
            message="File too large (max 20MB)",
            status_code=413,
            error_code="FILE_TOO_LARGE"
        )

    image_bytes = await image_file.read()
    await image_file.close()

    if not image_bytes:
        return create_error_response(
            error_type="ValidationError",
            message="Empty file uploaded",
            status_code=400,
            error_code="EMPTY_FILE"
        )

    try:
        img_v = Image.open(io.BytesIO(image_bytes))
        img_v.verify()
        img_v.close()
    except Exception as img_e:
        return create_error_response(
            error_type="ValidationError",
            message=f"Invalid image file: {str(img_e)}",
            status_code=400,
            error_code="INVALID_IMAGE_FILE"
        )

    capture_event = {
        "timestamp_ms": ts_ms,
        "image_bytes": image_bytes,
        "app_name": app_name,
        "window_title": window_title,
        "page_url": page_url,
        "source": f"api_upload:{image_file.filename}"
    }

    try:
        capture_data_queue.put_nowait(capture_event)
        return create_success_response(
            data={"filename": image_file.filename, "timestamp": ts_ms},
            message="Image upload accepted for processing",
            status_code=202
        )
    except asyncio.QueueFull:
        return create_error_response(
            error_type="ServiceError",
            message="Capture queue is full",
            status_code=503,
            error_code="QUEUE_FULL"
        )

@app.post("/api/search", summary="Perform LLM-driven Search")
async def api_perform_search_with_llm(payload: SearchQueryPayload, request: Request):
    """Enhanced search endpoint with proper response handling."""
    user_query = payload.query
    session_id = request.client.host if request.client else "default_session"

    # Get conversation history using the enhanced manager
    current_conversation = conversation_manager.get_conversation(session_id)
    current_conversation.append({"role": "user", "content": user_query})

    # Trim history if needed
    if len(current_conversation) > config.LLM_CONVERSATION_HISTORY_MAX_TURNS * 2:
        current_conversation[:] = current_conversation[-(config.LLM_CONVERSATION_HISTORY_MAX_TURNS * 2):]

    messages_for_llm = [{"role": "system", "content": XAI_SYSTEM_PROMPT}] + current_conversation
    final_llm_answer_content = "I encountered an error processing your request."
    retrieved_db_entries_for_api_response: List[DbEntry] = []

    try:
        for iteration in range(config.LLM_MAX_TOOL_ITERATIONS):
            logger.info(f"LLM Search iteration {iteration+1} for session {session_id}")

            xai_response_json = await call_xai_llm_with_tools_robust(
                messages=messages_for_llm,
                tools=XAI_TOOLS_AVAILABLE,
                tool_choice="auto"
            )

            if not xai_response_json.get("choices") or not xai_response_json["choices"][0].get("message"):
                return create_error_response(
                    error_type="APIError",
                    message="Invalid response from LLM service",
                    status_code=502,
                    error_code="INVALID_LLM_RESPONSE"
                )

            response_message = xai_response_json["choices"][0]["message"]
            messages_for_llm.append(response_message)

            tool_calls = response_message.get("tool_calls")
            if tool_calls:
                logger.info(f"LLM requesting tool calls: {[tc.get('function',{}).get('name') for tc in tool_calls]}")
                tool_results_for_llm_turn = []

                for tool_call in tool_calls:
                    fn_name = tool_call.get("function", {}).get("name")
                    tool_id = tool_call.get("id")

                    if not fn_name or not tool_id:
                        logger.error(f"Malformed tool_call: {tool_call}")
                        tool_results_for_llm_turn.append({
                            "tool_call_id": tool_id,
                            "role": "tool",
                            "name": fn_name,
                            "content": json.dumps({"error": "Malformed tool call"})
                        })
                        continue

                    if fn_name in AVAILABLE_TOOL_FUNCTIONS_MAP:
                        try:
                            args_str = tool_call.get("function", {}).get("arguments", "{}")
                            args_dict = json.loads(args_str)
                            tool_fn = AVAILABLE_TOOL_FUNCTIONS_MAP[fn_name]
                            tool_output_data = await tool_fn(**args_dict)
                            tool_results_for_llm_turn.append({
                                "tool_call_id": tool_id,
                                "role": "tool",
                                "name": fn_name,
                                "content": json.dumps(tool_output_data or [])
                            })
                        except json.JSONDecodeError as e:
                            tool_results_for_llm_turn.append({
                                "tool_call_id": tool_id,
                                "role": "tool",
                                "name": fn_name,
                                "content": json.dumps({"error": f"Invalid JSON: {str(e)}"})
                            })
                        except Exception as e:
                            logger.error(f"Tool execution error: {e}", exc_info=True)
                            tool_results_for_llm_turn.append({
                                "tool_call_id": tool_id,
                                "role": "tool",
                                "name": fn_name,
                                "content": json.dumps({"error": f"Tool failed: {str(e)}"})
                            })
                    else:
                        tool_results_for_llm_turn.append({
                            "tool_call_id": tool_id,
                            "role": "tool",
                            "name": fn_name,
                            "content": json.dumps({"error": f"Unknown tool: {fn_name}"})
                        })

                messages_for_llm.extend(tool_results_for_llm_turn)
            else:
                final_llm_answer_content = response_message.get("content")
                logger.info(f"LLM provided final answer for session {session_id}")
                break
        else:
            logger.warning(f"LLM search exceeded max iterations for session {session_id}")
            final_llm_answer_content = "I tried multiple analysis steps but couldn't complete your request. Please try rephrasing."

        # Update conversation history
        conversation_manager.update_conversation(
            session_id,
            [msg for msg in messages_for_llm if msg.get("role") != "system"]
        )

        # Extract entries from tool results
        last_search_tool_output_content_str = None
        for msg in reversed(messages_for_llm):
            if msg.get("role") == "tool" and msg.get("name") == "search_eidon_database_tool_impl":
                last_search_tool_output_content_str = msg.get("content")
                break

        if last_search_tool_output_content_str:
            try:
                tool_data_list = json.loads(last_search_tool_output_content_str)
                if isinstance(tool_data_list, list):
                    for item_dict in tool_data_list[:payload.limit]:
                        ts = item_dict.get("timestamp")
                        if ts:
                            db_e = await get_entry_by_timestamp(ts, load_embedding=False)
                            if db_e:
                                db_e.relevance_score = item_dict.get("relevance_score", db_e.relevance_score)
                                retrieved_db_entries_for_api_response.append(db_e)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse tool output: {e}")

        response_data = SearchResponseModel(
            query_received=user_query,
            llm_final_answer=final_llm_answer_content,
            retrieved_entries=[EntryResponseModel.from_db_entry(e) for e in retrieved_db_entries_for_api_response]
        )

        return create_success_response(
            data=response_data.dict(),
            message="Search completed successfully"
        )

    except HTTPException as e:
        return create_error_response(
            error_type="HTTPException",
            message=e.detail,
            status_code=e.status_code,
            error_code=f"HTTP_{e.status_code}"
        )
    except Exception as e:
        logger.critical(f"Critical error in search: {e}", exc_info=True)
        return create_error_response(
            error_type="InternalError",
            message="An internal server error occurred",
            status_code=500,
            error_code="SEARCH_ERROR"
        )

@app.get("/api/status", summary="Get Service Status")
async def api_get_service_status():
    """Enhanced status endpoint with detailed information."""
    try:
        db_count = await count_entries()
    except Exception as e:
        logger.error(f"Error counting entries: {e}")
        db_count = -1

    capture_running = False
    if screen_capture_service_instance:
        try:
            capture_running = screen_capture_service_instance.is_running()
        except Exception:
            capture_running = False

    # Get memory and performance stats
    memory_stats = memory_manager.get_memory_usage()
    performance_stats = {
        "active_tasks": len(active_background_tasks),
        "conversation_count": len(conversation_manager.conversations),
        "queue_size": capture_data_queue.qsize()
    }

    status_data = StatusResponseModel(
        status="operational" if config.VOYAGE_API_KEY and config.XAI_API_KEY else "degraded",
        total_entries_in_db=db_count,
        voyage_api_configured=bool(config.VOYAGE_API_KEY),
        xai_api_configured=bool(config.XAI_API_KEY),
        automatic_capture_enabled_in_config=config.ENABLE_AUTOMATIC_SCREEN_CAPTURE,
        capture_service_running=capture_running,
        capture_queue_size=capture_data_queue.qsize(),
        platform=config.PLATFORM_SYSTEM,
        active_processing_tasks=len(active_background_tasks),
        memory_usage=memory_stats,
        performance_stats=performance_stats
    )

    return create_success_response(
        data=status_data.dict(),
        message="Status retrieved successfully"
    )

@app.get("/api/entries", summary="List Database Entries")
async def api_list_database_entries(limit: int = 20, offset: int = 0, text_summary: bool = True):
    """Enhanced entries listing with proper response handling."""
    if not (0 <= offset):
        return create_error_response(
            error_type="ValidationError",
            message="Offset must be non-negative",
            status_code=400,
            error_code="INVALID_OFFSET"
        )

    if not (1 <= limit <= 200):
        return create_error_response(
            error_type="ValidationError",
            message="Limit must be between 1 and 200",
            status_code=400,
            error_code="INVALID_LIMIT"
        )

    try:
        db_entries = await get_all_entries(load_embeddings=False, limit=limit, offset=offset)
        entries_data = [EntryResponseModel.from_db_entry(e, include_text=not text_summary) for e in db_entries]

        return create_success_response(
            data=entries_data,
            message=f"Retrieved {len(entries_data)} entries"
        )
    except Exception as e:
        logger.error(f"Error retrieving entries: {e}", exc_info=True)
        return create_error_response(
            error_type="DatabaseError",
            message="Failed to retrieve entries",
            status_code=500,
            error_code="DB_ERROR"
        )

@app.get("/api/entry/ts/{timestamp_ms}", summary="Get Entry by Timestamp")
async def api_get_database_entry_by_timestamp(timestamp_ms: int):
    """Enhanced entry retrieval by timestamp."""
    try:
        db_entry = await get_entry_by_timestamp(timestamp_ms, load_embedding=False)
        if not db_entry:
            return create_error_response(
                error_type="NotFoundError",
                message="Entry not found",
                status_code=404,
                error_code="ENTRY_NOT_FOUND"
            )

        entry_data = EntryResponseModel.from_db_entry(db_entry)
        return create_success_response(
            data=entry_data.dict(),
            message="Entry retrieved successfully"
        )
    except Exception as e:
        logger.error(f"Error retrieving entry by timestamp: {e}", exc_info=True)
        return create_error_response(
            error_type="DatabaseError",
            message="Failed to retrieve entry",
            status_code=500,
            error_code="DB_ERROR"
        )

@app.delete("/api/entry/{entry_id}", summary="Delete Entry by ID")
async def api_delete_database_entry(entry_id: int):
    """Enhanced entry deletion with proper response handling."""
    try:
        deleted = await db_delete_entry_by_id(entry_id)
        if not deleted:
            return create_error_response(
                error_type="NotFoundError",
                message=f"Entry with ID {entry_id} not found",
                status_code=404,
                error_code="ENTRY_NOT_FOUND"
            )

        logger.info(f"Deleted entry with ID {entry_id}")
        return create_success_response(
            message=f"Entry {entry_id} deleted successfully"
        )
    except Exception as e:
        logger.error(f"Error deleting entry: {e}", exc_info=True)
        return create_error_response(
            error_type="DatabaseError",
            message="Failed to delete entry",
            status_code=500,
            error_code="DB_ERROR"
        )

# Screenshot serving endpoint (if aiofiles available)
if AIOFILES_AVAILABLE:
    @app.get("/api/screenshot/{filename}", summary="Get Screenshot File")
    async def api_get_screenshot_image_file(filename: str):
        """Enhanced screenshot serving with proper response handling."""
        if ".." in filename or filename.startswith(("/", "\\")):
            return create_error_response(
                error_type="ValidationError",
                message="Invalid filename",
                status_code=400,
                error_code="INVALID_FILENAME"
            )

        file_path = os.path.join(config.SCREENSHOTS_PATH, filename)
        if not os.path.isfile(file_path):
            return create_error_response(
                error_type="NotFoundError",
                message="Screenshot file not found",
                status_code=404,
                error_code="FILE_NOT_FOUND"
            )

        mime_type = get_mime_type_cached(filename)

        try:
            async with aiofiles.open(file_path, "rb") as f:
                content = await f.read()
            return FastAPIResponse(content=content, media_type=mime_type)
        except Exception as e:
            logger.error(f"Error serving screenshot {filename}: {e}", exc_info=True)
            return create_error_response(
                error_type="FileError",
                message="Could not read screenshot file",
                status_code=500,
                error_code="FILE_READ_ERROR"
            )

# --- Main execution ---
if __name__ == "__main__":
    logger.info("Starting Eidon Backend with enhanced features...")

    # Configuration checks
    if not config.VOYAGE_API_KEY:
        logger.critical("VOYAGE_API_KEY not set. Embedding functionality will fail.")
    if not config.XAI_API_KEY:
        logger.critical("XAI_API_KEY not set. LLM search will fail.")

    uvicorn.run(
        "app_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=config.LOG_LEVEL.lower(),
    )