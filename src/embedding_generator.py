import hashlib
import logging
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import MODEL_NAME, MAX_EMBEDDING_CACHE_SIZE

logger = logging.getLogger(__name__)

# Global model singleton & cache store
_model_instance: SentenceTransformer | None = None
_embedding_cache: dict[str, np.ndarray] = {}


def get_embedding_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """
    Lazy singleton loader for the SentenceTransformer embedding model.
    Instantiates the model once and reuses it across calls.
    """
    global _model_instance
    if _model_instance is None:
        logger.info(f"Loading SentenceTransformer model singleton: {model_name}")
        _model_instance = SentenceTransformer(model_name)
    return _model_instance


def reset_model_singleton() -> None:
    """Reset the singleton model instance (useful for testing)."""
    global _model_instance
    _model_instance = None


def compute_dataset_hash(df: pd.DataFrame) -> str:
    """
    Compute MD5 hash fingerprint for DataFrame's topic and message columns.
    """
    if df.empty:
        return "empty_df"

    topics = df.get("topic", pd.Series([""] * len(df))).astype(str)
    messages = df.get("message", pd.Series([""] * len(df))).astype(str)

    combined_text = (topics + "||" + messages).str.cat(sep="\n")
    return hashlib.md5(combined_text.encode("utf-8")).hexdigest()


def clear_embedding_cache() -> None:
    """Clear the in-memory ticket embedding cache."""
    global _embedding_cache
    _embedding_cache.clear()


def get_embedding_cache() -> dict[str, np.ndarray]:
    """Retrieve the current in-memory embedding cache dictionary."""
    return _embedding_cache


class EmbeddingGenerator:
    """
    Handles text and DataFrame encoding using the singleton model and dataset caching.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name

    def encode_texts(self, texts: list[str] | str) -> np.ndarray:
        """
        Encode single text string or list of text strings into normalized L2 vectors.

        Parameters:
            texts (list[str] | str): Text string or list of text strings to encode.

        Returns:
            np.ndarray: 2D array of normalized 384-d float32 embeddings.
        """
        if isinstance(texts, str):
            if not texts.strip():
                raise ValueError("Text to encode cannot be empty or whitespace-only")
            input_texts = [texts.strip()]
        elif isinstance(texts, list):
            if not texts:
                raise ValueError("Text list cannot be empty")
            input_texts = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
            if not input_texts:
                raise ValueError("Text list contains no valid non-empty strings")
        else:
            raise TypeError("texts must be a string or a list of strings")

        model = get_embedding_model(self.model_name)
        embeddings = model.encode(
            input_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)

    def encode_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """
        Encode support ticket DataFrame rows into L2-normalized embeddings,
        reusing cached matrix if dataset fingerprint matches.

        Parameters:
            df (pd.DataFrame): Support tickets DataFrame.

        Returns:
            np.ndarray: 2D matrix of embeddings matching the DataFrame row count.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        if df.empty:
            return np.empty((0, 384), dtype=np.float32)

        # Validate required columns
        required = {"topic", "message"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"DataFrame missing required columns for embedding: {missing}"
            )

        # Check cache hit
        dataset_hash = compute_dataset_hash(df)
        if dataset_hash in _embedding_cache:
            logger.debug(f"Cache hit for dataset hash: {dataset_hash}")
            return _embedding_cache[dataset_hash]

        # Construct combined text strings per ticket
        formatted_texts = [
            f"Topic: {row.get('topic', '')} | Message: {row.get('message', '')}"
            for _, row in df.iterrows()
        ]

        # Generate embeddings
        embeddings = self.encode_texts(formatted_texts)

        # Manage cache size
        if len(_embedding_cache) >= MAX_EMBEDDING_CACHE_SIZE:
            first_key = next(iter(_embedding_cache))
            del _embedding_cache[first_key]

        _embedding_cache[dataset_hash] = embeddings
        return embeddings
