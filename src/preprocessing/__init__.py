"""
Confidence Memo Agent — Preprocessing Package
Input-Agnostic Data Ingestion & Schema Normalization Layer
"""

from src.preprocessing.loader import load_data
from src.preprocessing.detector import detect_file_type
from src.preprocessing.normalizer import normalize_data
from src.preprocessing.exceptions import (
    PreprocessingError,
    UnsupportedFileTypeError,
    EmptyFileError,
    InvalidSchemaError,
    ParserError,
)

__all__ = [
    "load_data",
    "detect_file_type",
    "normalize_data",
    "PreprocessingError",
    "UnsupportedFileTypeError",
    "EmptyFileError",
    "InvalidSchemaError",
    "ParserError",
]
