import os

# Embedding Model Configuration
MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# Retrieval Defaults
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("DEFAULT_SIMILARITY_THRESHOLD", "0.2"))

# Cache Configuration
MAX_EMBEDDING_CACHE_SIZE = int(os.getenv("MAX_EMBEDDING_CACHE_SIZE", "50"))

# Evidence Validation
EVIDENCE_SIMILARITY_THRESHOLD = float(os.getenv("EVIDENCE_SIMILARITY_THRESHOLD", "0.60"))
