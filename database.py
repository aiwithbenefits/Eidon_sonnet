import asyncio
import os
import time
import sqlite3
import aiosqlite
import numpy as np
from typing import Any, List, Optional, Tuple, Dict, Union
import logging

import config
from utils import timestamp_ms_to_human_readable

logger = logging.getLogger(__name__)

logger.info(f"Database path from config: {config.DB_PATH}")
db_dir = os.path.dirname(config.DB_PATH)
if not os.access(db_dir, os.W_OK):
    logger.critical(f"CRITICAL: No write permissions for database directory: {db_dir}. Eidon may not function correctly.")

class Entry:
    def __init__(self, id: Optional[int], app: Optional[str], title: Optional[str],
                 text: Optional[str], timestamp: int,
                 embedding: Optional[np.ndarray], filename: Optional[str],
                 page_url: Optional[str], relevance_score: Optional[float] = None):
        self.id = id
        self.app = app
        self.title = title
        self.text = text
        self.timestamp = timestamp
        self.embedding = embedding if embedding is not None and embedding.size > 0 else np.array([], dtype=np.float32)
        self.filename = filename
        self.page_url = page_url
        self.relevance_score = relevance_score

    def __repr__(self):
        return (f"Entry(id={self.id}, ts={self.timestamp}, app='{self.app}', "
                f"title='{self.title[:30] if self.title else ''}...', score={self.relevance_score}, "
                f"has_emb={self.embedding.size > 0})")

    def to_dict(self, include_embedding_summary=True):
        data = {
            "id": self.id,
            "app": self.app,
            "title": self.title,
            "text": self.text,
            "timestamp": self.timestamp,
            "human_readable_timestamp": timestamp_ms_to_human_readable(self.timestamp),
            "filename": self.filename,
            "page_url": self.page_url,
            "relevance_score": self.relevance_score,
        }
        if include_embedding_summary:
            data["embedding_present"] = self.embedding.size > 0
            if self.embedding.size > 0:
                data["embedding_dim"] = self.embedding.shape[0]
        return data

async def create_db() -> None:
    try:
        # Directory creation handled by config.ensure_dirs()
        if not os.path.exists(config.DB_PATH):
            with open(config.DB_PATH, 'w'):
                pass
            os.chmod(config.DB_PATH, 0o664)
            logger.info(f"Created new database file: {config.DB_PATH}")

        async with aiosqlite.connect(config.DB_PATH, timeout=10) as conn:
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA foreign_keys=ON")
            await conn.execute("PRAGMA busy_timeout = 5000")

            # Create main table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    app TEXT,
                    title TEXT,
                    text TEXT,
                    timestamp INTEGER UNIQUE NOT NULL,
                    embedding BLOB,
                    filename TEXT UNIQUE NOT NULL,
                    page_url TEXT
                )
            """)

            # Create indexes
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON entries (timestamp DESC)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_app ON entries (app)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_page_url ON entries (page_url)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_filename ON entries (filename)")
            
            # Create index for entries with embeddings (using LENGTH check instead of non-existent column)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_embedding_exists 
                ON entries (timestamp DESC) 
                WHERE embedding IS NOT NULL AND LENGTH(embedding) > 0
            """)

            # Column check/addition logic
            cursor = await conn.execute("PRAGMA table_info(entries)")
            columns = [column[1] for column in await cursor.fetchall()]
            required_columns = {
                "filename": "TEXT UNIQUE NOT NULL",
                "page_url": "TEXT",
                "app": "TEXT",
                "title": "TEXT"
            }
            
            for col_name, col_type in required_columns.items():
                if col_name not in columns:
                    try:
                        # For NOT NULL columns, we need to handle existing data
                        if "NOT NULL" in col_type and col_name == "filename":
                            # First add as nullable
                            await conn.execute(f"ALTER TABLE entries ADD COLUMN {col_name} TEXT")
                            # Update existing rows with a default filename
                            await conn.execute("""
                                UPDATE entries 
                                SET filename = 'legacy_' || id || '_' || timestamp || '.webp' 
                                WHERE filename IS NULL
                            """)
                            logger.info(f"Added missing '{col_name}' column and populated with default values.")
                        else:
                            await conn.execute(f"ALTER TABLE entries ADD COLUMN {col_name} {col_type}")
                            logger.info(f"Added missing '{col_name}' column to 'entries' table.")
                    except sqlite3.OperationalError as e:
                        logger.warning(f"Could not add column {col_name}, might already exist: {e}")

            await conn.commit()
            logger.info("Database initialized/verified successfully.")
            
    except sqlite3.Error as e:
        logger.error(f"Database error during initialization: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error during DB creation: {e}", exc_info=True)
        raise

async def insert_entry(
    timestamp: int,
    text: Optional[str] = None,
    embedding: Optional[np.ndarray] = None,
    app: Optional[str] = None,
    title: Optional[str] = None,
    filename: Optional[str] = None,
    page_url: Optional[str] = None
) -> Optional[int]:
    if not filename:
        logger.error(f"Attempted to insert entry for ts {timestamp} without a filename. Skipping.")
        return None

    embedding_bytes = b''
    if embedding is not None and embedding.size > 0:
        try:
            if embedding.ndim > 1:
                embedding = embedding.flatten()
            if embedding.dtype != np.float32:
                embedding = embedding.astype(np.float32)
            embedding_bytes = embedding.tobytes()
        except Exception as e:
            logger.error(f"Error converting embedding to bytes for ts {timestamp}, fn {filename}: {e}", exc_info=True)
            embedding_bytes = b''

    inserted_row_id: Optional[int] = None
    try:
        async with aiosqlite.connect(config.DB_PATH, timeout=10) as conn:
            await conn.execute("PRAGMA busy_timeout = 5000")
            cursor = await conn.execute(
                """
                INSERT INTO entries (app, title, text, timestamp, embedding, filename, page_url)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (app, title, text, timestamp, embedding_bytes, filename, page_url)
            )
            await conn.commit()
            inserted_row_id = cursor.lastrowid
            if inserted_row_id:
                logger.info(f"DB: Inserted ID {inserted_row_id}, ts {timestamp}, fn {filename}")
            else:
                logger.warning(f"DB: Insert for ts {timestamp}, fn {filename} did not yield row ID.")

    except sqlite3.IntegrityError as e:
        if "UNIQUE constraint failed: entries.timestamp" in str(e):
            logger.warning(f"DB: Entry with ts {timestamp} already exists (UNIQUE constraint). Fn: {filename}. Skipping.")
        elif "UNIQUE constraint failed: entries.filename" in str(e):
            logger.warning(f"DB: Entry with fn '{filename}' already exists (UNIQUE constraint). Ts: {timestamp}. Skipping.")
        else:
            logger.error(f"DB IntegrityError for ts {timestamp}, fn {filename}: {e}", exc_info=True)
    except sqlite3.OperationalError as e:
        if "database is locked" in str(e).lower():
            logger.error(f"DB locked for ts {timestamp}, fn {filename}: {e}. Consider increasing busy_timeout.")
        else:
            logger.error(f"DB OperationalError for ts {timestamp}, fn {filename}: {e}", exc_info=True)
    except sqlite3.Error as e:
        logger.error(f"DB error for ts {timestamp}, fn {filename}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected DB error for ts {timestamp}, fn {filename}: {e}", exc_info=True)

    return inserted_row_id

async def _row_to_entry(row_data: sqlite3.Row, load_embeddings: bool) -> Optional[Entry]:
    if not row_data:
        return None
        
    embedding_array = np.array([], dtype=np.float32)
    if load_embeddings:
        emb_blob = row_data["embedding"]
        if emb_blob and len(emb_blob) > 0:
            try:
                temp_array = np.frombuffer(emb_blob, dtype=np.float32)
                # Validate dimension if possible
                if hasattr(config, 'VOYAGE_EMBEDDING_DIM') and config.VOYAGE_EMBEDDING_DIM and temp_array.size > 0:
                    if temp_array.size == config.VOYAGE_EMBEDDING_DIM:
                        embedding_array = temp_array
                    else:
                        logger.warning(f"Embedding blob size {len(emb_blob)} for ID {row_data['id']} "
                                     f"(shape {temp_array.shape}) is not VOYAGE_EMBEDDING_DIM {config.VOYAGE_EMBEDDING_DIM}. "
                                     f"Returning empty embedding.")
                elif temp_array.size > 0:
                    embedding_array = temp_array
            except ValueError as ve:
                logger.error(f"Error deserializing embedding for entry ID {row_data['id']}: {ve}. "
                           f"Blob size: {len(emb_blob)}")
                
    return Entry(
        id=row_data["id"],
        app=row_data["app"],
        title=row_data["title"],
        text=row_data["text"],
        timestamp=row_data["timestamp"],
        embedding=embedding_array,
        filename=row_data["filename"],
        page_url=row_data["page_url"]
    )

async def get_all_entries(load_embeddings: bool = True, limit: Optional[int] = None, offset: Optional[int] = 0) -> List[Entry]:
    entries_list: List[Entry] = []
    sql_query = "SELECT id, app, title, text, timestamp, embedding, filename, page_url FROM entries ORDER BY timestamp DESC"
    params = []
    
    if limit is not None:
        sql_query += " LIMIT ?"
        params.append(limit)
        if offset is not None and offset > 0:
            sql_query += " OFFSET ?"
            params.append(offset)
            
    try:
        async with aiosqlite.connect(config.DB_PATH, timeout=10) as conn:
            conn.row_factory = sqlite3.Row
            cursor = await conn.execute(sql_query, params)
            rows = await cursor.fetchall()
            for row_data in rows:
                entry = await _row_to_entry(row_data, load_embeddings)
                if entry:
                    entries_list.append(entry)
    except sqlite3.Error as e:
        logger.error(f"Database error while fetching entries: {e}", exc_info=True)
    return entries_list

async def get_entry_by_timestamp(timestamp_val: int, load_embedding: bool = True) -> Optional[Entry]:
    try:
        async with aiosqlite.connect(config.DB_PATH, timeout=10) as conn:
            conn.row_factory = sqlite3.Row
            cursor = await conn.execute(
                "SELECT id, app, title, text, timestamp, embedding, filename, page_url FROM entries WHERE timestamp = ?",
                (timestamp_val,)
            )
            row_data = await cursor.fetchone()
            return await _row_to_entry(row_data, load_embedding)
    except sqlite3.Error as e:
        logger.error(f"DB error fetching entry by timestamp {timestamp_val}: {e}", exc_info=True)
    return None

async def get_entry_by_filename(filename: str, load_embedding: bool = True) -> Optional[Entry]:
    try:
        async with aiosqlite.connect(config.DB_PATH, timeout=10) as conn:
            conn.row_factory = sqlite3.Row
            cursor = await conn.execute(
                "SELECT id, app, title, text, timestamp, embedding, filename, page_url FROM entries WHERE filename = ?",
                (filename,)
            )
            row_data = await cursor.fetchone()
            return await _row_to_entry(row_data, load_embedding)
    except sqlite3.Error as e:
        logger.error(f"DB error fetching entry by filename {filename}: {e}", exc_info=True)
    return None

async def get_recent_entries_with_embeddings(
    limit: int = 10, 
    after_timestamp: Optional[int] = None
) -> List[Entry]:
    recent_entries: List[Entry] = []
    try:
        query = """
            SELECT id, app, title, text, timestamp, embedding, filename, page_url
            FROM entries
            WHERE embedding IS NOT NULL AND LENGTH(embedding) > ?
        """
        
        # Calculate minimum expected bytes for embedding
        min_embedding_bytes = 4  # At least 1 float32
        if hasattr(config, 'VOYAGE_EMBEDDING_DIM') and config.VOYAGE_EMBEDDING_DIM:
            min_embedding_bytes = config.VOYAGE_EMBEDDING_DIM * 4 // 2  # At least half expected
            
        params = [min_embedding_bytes]
        
        if after_timestamp is not None:
            query += " AND timestamp >= ?"
            params.append(after_timestamp)
            
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        async with aiosqlite.connect(config.DB_PATH, timeout=10) as conn:
            conn.row_factory = sqlite3.Row
            cursor = await conn.execute(query, params)
            rows = await cursor.fetchall()
            for row_data in rows:
                entry = await _row_to_entry(row_data, load_embeddings=True)
                if entry and entry.embedding.size > 0:
                    recent_entries.append(entry)
    except sqlite3.Error as e:
        logger.error(f"DB error fetching recent embeddings: {e}", exc_info=True)
    return recent_entries

async def get_timestamps_only(limit: Optional[int] = None, offset: Optional[int] = 0) -> List[int]:
    timestamps_list: List[int] = []
    sql_query = "SELECT timestamp FROM entries ORDER BY timestamp DESC"
    params = []
    
    if limit is not None:
        sql_query += " LIMIT ?"
        params.append(limit)
        if offset is not None and offset > 0:
            sql_query += " OFFSET ?"
            params.append(offset)
            
    try:
        async with aiosqlite.connect(config.DB_PATH, timeout=10) as conn:
            cursor = await conn.execute(sql_query, params)
            rows = await cursor.fetchall()
            timestamps_list = [row[0] for row in rows]
    except sqlite3.Error as e:
        logger.error(f"Database error while fetching timestamps: {e}", exc_info=True)
    return timestamps_list

async def count_entries() -> int:
    count = 0
    try:
        async with aiosqlite.connect(config.DB_PATH, timeout=10) as conn:
            cursor = await conn.execute("SELECT COUNT(*) FROM entries")
            result = await cursor.fetchone()
            if result:
                count = result[0]
    except sqlite3.Error as e:
        logger.error(f"Database error while counting entries: {e}", exc_info=True)
    return count

async def delete_entry_by_id(entry_id: int) -> bool:
    """Deletes an entry by its primary key ID."""
    try:
        async with aiosqlite.connect(config.DB_PATH, timeout=10) as conn:
            cursor = await conn.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
            await conn.commit()
            return cursor.rowcount > 0
    except sqlite3.Error as e:
        logger.error(f"Database error deleting entry ID {entry_id}: {e}", exc_info=True)
        return False

async def get_entry_by_id(entry_id: int, load_embedding: bool = True) -> Optional[Entry]:
    """Get entry by its primary key ID."""
    try:
        async with aiosqlite.connect(config.DB_PATH, timeout=10) as conn:
            conn.row_factory = sqlite3.Row
            cursor = await conn.execute(
                "SELECT id, app, title, text, timestamp, embedding, filename, page_url FROM entries WHERE id = ?",
                (entry_id,)
            )
            row_data = await cursor.fetchone()
            return await _row_to_entry(row_data, load_embedding)
    except sqlite3.Error as e:
        logger.error(f"DB error fetching entry by ID {entry_id}: {e}", exc_info=True)
    return None

# Enhanced database maintenance functions
async def vacuum_database() -> bool:
    """Vacuum the database to reclaim space and optimize performance."""
    try:
        async with aiosqlite.connect(config.DB_PATH, timeout=30) as conn:
            await conn.execute("VACUUM")
            logger.info("Database vacuum completed successfully.")
            return True
    except sqlite3.Error as e:
        logger.error(f"Error during database vacuum: {e}", exc_info=True)
        return False

async def analyze_database() -> bool:
    """Analyze the database to update query planner statistics."""
    try:
        async with aiosqlite.connect(config.DB_PATH, timeout=10) as conn:
            await conn.execute("ANALYZE")
            logger.info("Database analysis completed successfully.")
            return True
    except sqlite3.Error as e:
        logger.error(f"Error during database analysis: {e}", exc_info=True)
        return False

async def get_database_stats() -> Dict[str, Any]:
    """Get database statistics for monitoring."""
    stats = {}
    try:
        async with aiosqlite.connect(config.DB_PATH, timeout=10) as conn:
            # Get table info
            cursor = await conn.execute("SELECT COUNT(*) FROM entries")
            result = await cursor.fetchone()
            stats['total_entries'] = result[0] if result else 0
            
            # Get entries with embeddings
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM entries WHERE embedding IS NOT NULL AND LENGTH(embedding) > 0"
            )
            result = await cursor.fetchone()
            stats['entries_with_embeddings'] = result[0] if result else 0
            
            # Get database file size
            if os.path.exists(config.DB_PATH):
                stats['db_file_size_mb'] = os.path.getsize(config.DB_PATH) / (1024 * 1024)
            
            # Get date range
            cursor = await conn.execute("SELECT MIN(timestamp), MAX(timestamp) FROM entries")
            result = await cursor.fetchone()
            if result and result[0] and result[1]:
                stats['earliest_timestamp'] = result[0]
                stats['latest_timestamp'] = result[1]
                stats['time_span_days'] = (result[1] - result[0]) / (1000 * 60 * 60 * 24)
                
    except sqlite3.Error as e:
        logger.error(f"Error getting database stats: {e}", exc_info=True)
        stats['error'] = str(e)
    
    return stats

# Self-test updated for robustness
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Ensure config has VOYAGE_EMBEDDING_DIM for tests
    if not hasattr(config, 'VOYAGE_EMBEDDING_DIM') or not config.VOYAGE_EMBEDDING_DIM:
        config.VOYAGE_EMBEDDING_DIM = 1024
        logger.warning(f"Using default VOYAGE_EMBEDDING_DIM={config.VOYAGE_EMBEDDING_DIM} for database self-test.")

    async def test_db_operations():
        logger.info("Running database.py self-test...")
        await create_db()

        ts1 = int(time.time() * 1000)
        fn1 = f"test_{ts1}.webp"
        emb1 = np.random.rand(config.VOYAGE_EMBEDDING_DIM).astype(np.float32)

        id1 = await insert_entry(
            timestamp=ts1, text="Test entry 1", embedding=emb1,
            app="TestApp", title="Test Window 1", filename=fn1, page_url="http://example.com/test1"
        )
        assert id1 is not None, "Insertion failed for entry 1"

        # Test duplicate timestamp
        id_dup_ts = await insert_entry(timestamp=ts1, text="Duplicate TS", filename=f"test_dup_ts_{ts1}.webp")
        assert id_dup_ts is None, "Duplicate timestamp insertion should be skipped"

        # Test duplicate filename
        ts_dup_fn = int(time.time() * 1000) + 100
        id_dup_fn = await insert_entry(timestamp=ts_dup_fn, text="Duplicate FN", filename=fn1)
        assert id_dup_fn is None, "Duplicate filename insertion should be skipped"

        ts2 = int(time.time() * 1000) + 200
        fn2 = f"test_{ts2}.webp"
        id2 = await insert_entry(
            timestamp=ts2, text="Test entry 2, no embedding",
            app="TestApp", title="Test Window 2", filename=fn2
        )
        assert id2 is not None, "Insertion failed for entry 2"

        entry1_ret_ts = await get_entry_by_timestamp(ts1)
        assert entry1_ret_ts and entry1_ret_ts.id == id1 and np.allclose(entry1_ret_ts.embedding, emb1)

        entry1_ret_fn = await get_entry_by_filename(fn1)
        assert entry1_ret_fn and entry1_ret_fn.id == id1

        entry1_ret_id = await get_entry_by_id(id1)
        assert entry1_ret_id and entry1_ret_id.timestamp == ts1

        all_entries = await get_all_entries(limit=5)
        assert len(all_entries) >= 2

        recent_with_embeddings = await get_recent_entries_with_embeddings(limit=5)
        assert len(recent_with_embeddings) >= 1
        assert recent_with_embeddings[0].id == id1
        assert recent_with_embeddings[0].embedding.size == config.VOYAGE_EMBEDDING_DIM

        timestamps = await get_timestamps_only(limit=3)
        assert ts2 in timestamps and ts1 in timestamps

        total_count_before_delete = await count_entries()
        assert total_count_before_delete >= 2

        # Test database stats
        stats = await get_database_stats()
        assert stats['total_entries'] >= 2
        assert stats['entries_with_embeddings'] >= 1

        deleted = await delete_entry_by_id(id1)
        assert deleted
        total_count_after_delete = await count_entries()
        assert total_count_after_delete == total_count_before_delete - 1
        
        # Verify the entry with the specific timestamp is gone
        assert await get_entry_by_timestamp(ts1) is None

        logger.info("database.py self-test completed successfully.")
        
        # Clean up remaining test entry
        if id2:
            await delete_entry_by_id(id2)

    asyncio.run(test_db_operations())