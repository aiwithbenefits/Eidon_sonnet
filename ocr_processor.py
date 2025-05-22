import logging
import voyageai
from typing import List, Optional
import asyncio

import config
from utils import async_retry  # Using centralized retry

logger = logging.getLogger(__name__)


class VoyageClient:
    """Singleton class to manage Voyage AI client for text embeddings."""
    _instance = None
    _client = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VoyageClient, cls).__new__(cls)
            cls._instance._initialize_client()
        return cls._instance
    
    @classmethod
    def _initialize_client(cls):
        """Initialize the Voyage AI client if API key is available."""
        if cls._client is None and config.VOYAGE_API_KEY:
            try:
                cls._client = voyageai.Client(api_key=config.VOYAGE_API_KEY)
                logger.info("Initialized Voyage AI client for text embeddings")
            except Exception as e:
                logger.error(f"Failed to initialize Voyage AI client: {e}")
    
    @classmethod
    def get_client(cls):
        """Get the Voyage AI client instance."""
        if cls._client is None:
            cls._initialize_client()
        return cls._client


def get_voyage_client():
    """Get or create a Voyage AI client instance for text embeddings."""
    return VoyageClient().get_client()


@async_retry(max_retries=config.VOYAGE_API_RETRIES, delay_seconds=2.0)
async def get_embedding_for_text(
    text: str,
    model: str = config.VOYAGE_EMBED_MODEL,
    input_type: Optional[str] = None,
) -> Optional[List[float]]:
    """
    Gets an embedding for the given text using Voyage AI's embedding model.
    
    Args:
        text: The text to embed.
        model: The Voyage AI model to use for embedding.
        input_type: Type of input, either "query" or "document".
        
    Returns:
        A list of floats representing the text embedding, or None if an error occurs.
    """
    if not config.VOYAGE_API_KEY or not model:
        logger.warning("Voyage API key or model not configured. Embedding generation skipped.")
        return None
    if not text or not text.strip():
        logger.warning("Empty text provided for embedding. Skipping.")
        return None

    vo = get_voyage_client()
    if not vo:
        logger.error("Failed to initialize Voyage AI client")
        return None

    try:
        # Prepare the input parameters
        params = {
            "input": [text],
            "model": model,
        }
        
        # Add input_type if provided
        if input_type and input_type in ["query", "document"]:
            params["input_type"] = input_type
        
        # Get the embedding
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, lambda: vo.embed(**params))
        
        if getattr(response, "embeddings", None):
            return response.embeddings[0]
            
        logger.warning("No embedding was returned by Voyage AI")
        return None
        
    except Exception as e:
        logger.error(f"Error getting text embedding from Voyage AI: {e}", exc_info=True)
        return None
