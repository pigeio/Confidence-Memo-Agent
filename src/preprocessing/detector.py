import os
from pathlib import Path
from src.preprocessing.exceptions import (
    UnsupportedFileTypeError,
    EmptyFileError,
)

SUPPORTED_EXTENSIONS = {
    ".csv": "csv",
    ".xlsx": "excel",
    ".xls": "excel",
    ".json": "json",
    ".pdf": "pdf",
    ".txt": "txt",
}


def detect_file_type(file_path: str | Path) -> str:
    """
    Determine the input file type based on file extension and existence.

    Parameters:
        file_path (str | Path): Path to the input evidence file.

    Returns:
        str: Canonical file format identifier ('csv', 'excel', 'json', 'pdf', 'txt').

    Raises:
        FileNotFoundError: If the specified file path does not exist.
        EmptyFileError: If the file exists but has 0 bytes.
        UnsupportedFileTypeError: If the file extension is not supported.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found at path: {path}")

    if path.is_file() and path.stat().st_size == 0:
        raise EmptyFileError(f"Input file is empty (0 bytes): {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        supported_str = ", ".join(sorted(SUPPORTED_EXTENSIONS.keys()))
        raise UnsupportedFileTypeError(
            f"Unsupported file extension '{ext}'. Supported formats: {supported_str}"
        )

    return SUPPORTED_EXTENSIONS[ext]
