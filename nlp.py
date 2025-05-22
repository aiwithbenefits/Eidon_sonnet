import asyncio
import logging
import numpy as np
from functools import lru_cache
from typing import Dict, List, Optional, Tuple
import time
import hashlib
import httpx
from collections import OrderedDict
import threading

import config
from utils import async_retry

logger = logging.getLogger(__name__)

# Thread-safe embedding cache with LRU eviction
class ThreadSafeEmbeddingCache:
    def __init__(self, max_size: int = None):
        self.max_size = max_size or config.EMBEDDING_CACHE_SIZE
        self._cache = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[np.ndarray]:
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key].copy()
            self._misses += 1
            return None

    def put(self, key: str, value: np.ndarray):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                self._cache[key] = value.copy()
                # Evict oldest if over capacity
                while len(self._cache) > self.max_size:
                    self._cache.popitem(last=False)

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> Dict[str, int]:
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0
            return {
                'size': len(self._cache),
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate
            }

# Global cache instance
_embedding_cache = ThreadSafeEmbeddingCache()

# Voyage client initialization
voyage_client = None
voyage_sdk_imported_successfully = False

try:
    import voyageai
    if config.VOYAGE_API_KEY:
        try:
            voyage_client = voyageai.Client(
                api_key=config.VOYAGE_API_KEY,
                max_retries=0  # We handle retries with our decorator
            )
            voyage_sdk_imported_successfully = True
            logger.info(f"VoyageAI SDK client initialized. Model: {config.VOYAGE_EMBED_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize VoyageAI SDK client: {e}", exc_info=True)
    else:
        logger.warning("VOYAGE_API_KEY not set. VoyageAI embeddings will not be available.")
except ImportError:
    logger.error("VoyageAI SDK not found. Install with: pip install voyageai")
except Exception as e:
    logger.error(f"Unexpected error during VoyageAI SDK setup: {e}", exc_info=True)

def _create_cache_key(text: str, input_type: str) -> str:
    """Create a deterministic cache key for text and input type."""
    # Use hash for consistent key generation and memory efficiency
    text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    return f"{input_type}:{text_hash}"

def _preprocess_text(text: str) -> str:
    """Preprocess text for better embedding quality."""
    if not text:
        return ""

    # Basic cleaning
    text = text.strip()

    # Remove excessive whitespace
    import re
    text = re.sub(r'\s+', ' ', text)

    # Truncate if too long (Voyage has token limits)
    max_chars = 8000  # Conservative limit
    if len(text) > max_chars:
        text = text[:max_chars]
        logger.debug(f"Truncated text from {len(text)} to {max_chars} characters")

    return text

@async_retry(max_retries=config.VOYAGE_API_RETRIES, delay_seconds=1.5)
async def _get_embedding_voyage_via_sdk(text: str, input_type: str) -> np.ndarray:
    """Internal function to get embedding using VoyageAI SDK with caching and optimization."""
    dtype = np.float32

    if not voyage_sdk_imported_successfully or not voyage_client:
        logger.warning("Voyage SDK not available")
        return np.array([], dtype=dtype)

    if not text or len(text.strip()) == 0:
        return np.array([], dtype=dtype)

    # Preprocess text
    processed_text = _preprocess_text(text)
    if not processed_text:
        return np.array([], dtype=dtype)

    # Check cache
    cache_key = _create_cache_key(processed_text, input_type)
    cached_result = _embedding_cache.get(cache_key)
    if cached_result is not None:
        logger.debug(f"Cache hit for embedding (type: {input_type})")
        return cached_result

    try:
        # Use asyncio.to_thread for better async handling
        start_time = time.time()
        response = await asyncio.to_thread(
            voyage_client.embed,
            texts=[processed_text],
            model=config.VOYAGE_EMBED_MODEL,
            input_type=input_type,
            truncation=True
        )

        duration = time.time() - start_time
        logger.debug(f"Voyage API call took {duration:.2f}s for {len(processed_text)} chars")

        if response and response.embeddings and len(response.embeddings) > 0:
            embedding_vector = np.array(response.embeddings[0], dtype=dtype)

            # Validate embedding dimensions
            if config.VOYAGE_EMBEDDING_DIM and embedding_vector.shape[0] != config.VOYAGE_EMBEDDING_DIM:
                logger.error(
                    f"Voyage embedding dimension mismatch! Expected {config.VOYAGE_EMBEDDING_DIM}, "
                    f"got {embedding_vector.shape[0]} for model {config.VOYAGE_EMBED_MODEL}"
                )
                return np.array([], dtype=dtype)

            # Cache the result
            _embedding_cache.put(cache_key, embedding_vector)

            logger.debug(f"Generated embedding: shape={embedding_vector.shape}, type={input_type}")
            return embedding_vector.copy()
        else:
            logger.error(f"Voyage SDK returned no embeddings for text: '{processed_text[:70]}...'")
            return np.array([], dtype=dtype)

    except voyageai.error.VoyageAIError as e:
        logger.error(f"VoyageAI SDK error for '{processed_text[:70]}...': {e}")
        if hasattr(e, 'http_status') and e.http_status == 429:
            # Create a mock response for retry handling
            raise httpx.HTTPStatusError(
                message=str(e),
                request=None,
                response=httpx.Response(status_code=429, headers={"Retry-After": "10"})
            )
        raise
    except Exception as e:
        logger.error(f"Unexpected error in Voyage SDK embed: {e}", exc_info=True)
        return np.array([], dtype=dtype)

async def get_text_embedding(text: str) -> np.ndarray:
    """Generate embedding for document text with optimization."""
    return await _get_embedding_voyage_via_sdk(text, input_type="document")

async def get_query_embedding(query_text: str) -> np.ndarray:
    """Generate embedding for search query with optimization."""
    return await _get_embedding_voyage_via_sdk(query_text, input_type="query")

async def get_embeddings_batch(texts: List[str], input_type: str = "document") -> List[np.ndarray]:
    """Get embeddings for multiple texts in batch for better efficiency."""
    if not texts:
        return []

    # Check cache first
    results = []
    uncached_texts = []
    uncached_indices = []

    for i, text in enumerate(texts):
        processed_text = _preprocess_text(text)
        if not processed_text:
            results.append(np.array([], dtype=np.float32))
            continue

        cache_key = _create_cache_key(processed_text, input_type)
        cached_result = _embedding_cache.get(cache_key)
        if cached_result is not None:
            results.append(cached_result)
        else:
            results.append(None)  # Placeholder
            uncached_texts.append(processed_text)
            uncached_indices.append(i)

    # Process uncached texts in batches
    if uncached_texts:
        batch_size = min(config.VOYAGE_BATCH_SIZE, len(uncached_texts))

        for batch_start in range(0, len(uncached_texts), batch_size):
            batch_texts = uncached_texts[batch_start:batch_start + batch_size]
            batch_indices = uncached_indices[batch_start:batch_start + batch_size]

            try:
                if voyage_sdk_imported_successfully and voyage_client:
                    response = await asyncio.to_thread(
                        voyage_client.embed,
                        texts=batch_texts,
                        model=config.VOYAGE_EMBED_MODEL,
                        input_type=input_type,
                        truncation=True
                    )

                    if response and response.embeddings:
                        for i, embedding in enumerate(response.embeddings):
                            if i < len(batch_indices):
                                embedding_vector = np.array(embedding, dtype=np.float32)
                                result_index = batch_indices[i]
                                results[result_index] = embedding_vector

                                # Cache the result
                                cache_key = _create_cache_key(batch_texts[i], input_type)
                                _embedding_cache.put(cache_key, embedding_vector)

            except Exception as e:
                logger.error(f"Error in batch embedding: {e}")
                # Fill with empty arrays for failed batch
                for idx in batch_indices:
                    if results[idx] is None:
                        results[idx] = np.array([], dtype=np.float32)

    # Ensure all results are filled
    for i in range(len(results)):
        if results[i] is None:
            results[i] = np.array([], dtype=np.float32)

    return results

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Optimized cosine similarity computation."""
    if not isinstance(vec_a, np.ndarray) or not isinstance(vec_b, np.ndarray):
        return 0.0
    if vec_a.size == 0 or vec_b.size == 0 or vec_a.ndim != 1 or vec_b.ndim != 1:
        return 0.0
    if vec_a.shape != vec_b.shape:
        return 0.0

    # Use numpy's optimized operations
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    similarity = dot_product / (norm_a * norm_b)
    return float(np.clip(similarity, -1.0, 1.0))

def cosine_similarity_batch(query_vec: np.ndarray, doc_vecs: List[np.ndarray]) -> List[float]:
    """Compute cosine similarity between query and multiple documents efficiently."""
    if not isinstance(query_vec, np.ndarray) or query_vec.size == 0:
        return [0.0] * len(doc_vecs)

    similarities = []
    query_norm = np.linalg.norm(query_vec)

    if query_norm == 0:
        return [0.0] * len(doc_vecs)

    for doc_vec in doc_vecs:
        if not isinstance(doc_vec, np.ndarray) or doc_vec.size == 0:
            similarities.append(0.0)
            continue

        if doc_vec.shape != query_vec.shape:
            similarities.append(0.0)
            continue

        doc_norm = np.linalg.norm(doc_vec)
        if doc_norm == 0:
            similarities.append(0.0)
            continue

        similarity = np.dot(query_vec, doc_vec) / (query_norm * doc_norm)
        similarities.append(float(np.clip(similarity, -1.0, 1.0)))

    return similarities

def get_cache_stats() -> Dict[str, int]:
    """Get embedding cache statistics."""
    return _embedding_cache.stats()

def clear_embedding_cache():
    """Clear the embedding cache."""
    _embedding_cache.clear()
    logger.info("Embedding cache cleared")

# Self-test with performance monitoring
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    async def run_nlp_tests_main():
        logger.info(f"Running optimized nlp.py self-test. Voyage SDK: {voyage_sdk_imported_successfully}")

        if not (voyage_sdk_imported_successfully and config.VOYAGE_API_KEY):
            logger.error("VOYAGE_API_KEY not set or SDK not available. Aborting self-test.")
            return

        if not config.VOYAGE_EMBEDDING_DIM:
            logger.error("config.VOYAGE_EMBEDDING_DIM not set. Aborting self-test.")
            return

        test_texts = [
            "This is a sample document for testing Eidon's NLP embedding capabilities with Voyage AI.",
            "Another test document with different content for batch processing.",
            "A third document to test the batch embedding functionality."
        ]
        test_query = "Find information about Voyage AI NLP testing."

        # Test single embeddings
        start_time = time.time()
        doc_emb = await get_text_embedding(test_texts[0])
        single_time = time.time() - start_time

        query_emb = await get_query_embedding(test_query)

        logger.info(f"Single embedding took {single_time:.2f}s, shape: {doc_emb.shape}")
        assert doc_emb.size == config.VOYAGE_EMBEDDING_DIM or doc_emb.size == 0

        # Test batch embeddings
        start_time = time.time()
        batch_embs = await get_embeddings_batch(test_texts, "document")
        batch_time = time.time() - start_time

        logger.info(f"Batch embedding took {batch_time:.2f}s for {len(test_texts)} texts")
        assert len(batch_embs) == len(test_texts)

        # Test similarity computations
        if doc_emb.size > 0 and query_emb.size > 0:
            similarity = cosine_similarity(doc_emb, query_emb)
            logger.info(f"Cosine similarity: {similarity:.4f}")
            assert -1.0 <= similarity <= 1.0

            # Test batch similarity
            similarities = cosine_similarity_batch(query_emb, batch_embs)
            logger.info(f"Batch similarities: {similarities}")

        # Test caching
        cache_stats_before = get_cache_stats()
        logger.info(f"Cache stats before re-request: {cache_stats_before}")

        start_time = time.time()
        doc_emb_cached = await get_text_embedding(test_texts[0])
        cache_time = time.time() - start_time

        cache_stats_after = get_cache_stats()
        logger.info(f"Cache stats after re-request: {cache_stats_after}")
        logger.info(f"Cached request took {cache_time:.4f}s")

        assert np.array_equal(doc_emb, doc_emb_cached), "Cached embedding doesn't match original"
        assert cache_stats_after['hits'] > cache_stats_before['hits'], "Cache hit not registered"

        logger.info("Optimized nlp.py self-test completed successfully.")

    asyncio.run(run_nlp_tests_main())