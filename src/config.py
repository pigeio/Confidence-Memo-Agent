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

# Small Dataset Score Caps
SINGLE_TICKET_SCORE_CAP = int(os.getenv("SINGLE_TICKET_SCORE_CAP", "39"))
TWO_TICKET_SCORE_CAP = int(os.getenv("TWO_TICKET_SCORE_CAP", "69"))
MIN_TICKETS_FOR_HIGH_CONFIDENCE = int(os.getenv("MIN_TICKETS_FOR_HIGH_CONFIDENCE", "3"))

# Evidence Deduplication
DEDUPLICATION_SIMILARITY_THRESHOLD = float(
    os.getenv("DEDUPLICATION_SIMILARITY_THRESHOLD", "0.85")
)

# Evidence Clustering
CLUSTERING_DISTANCE_THRESHOLD = float(
    os.getenv("CLUSTERING_DISTANCE_THRESHOLD", "0.35")
)
CLUSTERING_MIN_SAMPLES = int(os.getenv("CLUSTERING_MIN_SAMPLES", "1"))

# Historical Calibration
DEFAULT_CALIBRATION_STORAGE_PATH = os.getenv(
    "DEFAULT_CALIBRATION_STORAGE_PATH", "data/calibration_history.json"
)

