from pathlib import Path
import pandas as pd

from src.preprocessing.detector import detect_file_type
from src.preprocessing.normalizer import normalize_data
from src.preprocessing.csv_parser import CSVParser
from src.preprocessing.excel_parser import ExcelParser
from src.preprocessing.json_parser import JSONParser
from src.preprocessing.text_parser import TextParser
from src.preprocessing.pdf_parser import PDFParser
from src.preprocessing.exceptions import UnsupportedFileTypeError

# Map format string to parser class
PARSER_REGISTRY = {
    "csv": CSVParser,
    "excel": ExcelParser,
    "json": JSONParser,
    "txt": TextParser,
    "pdf": PDFParser,
}


def load_data(file_path: str | Path) -> pd.DataFrame:
    """
    Public entry point to load, parse, and normalize evidence data from any supported format
    (CSV, Excel, JSON, TXT, PDF) into a standardized pandas DataFrame.

    Parameters:
        file_path (str | Path): Path to the input evidence file.

    Returns:
        pd.DataFrame: Normalized DataFrame with standard columns:
                      ['ticket_id', 'created_at', 'topic', 'message'].

    Raises:
        FileNotFoundError: If the file path does not exist.
        EmptyFileError: If the file is 0 bytes or contains no valid evidence records.
        UnsupportedFileTypeError: If the file format is not supported.
        ParserError: If parsing fails due to syntax or file corruption.
        InvalidSchemaError: If normalized schema cannot be produced.
    """
    path = Path(file_path)

    # 1. Detect file type
    file_type = detect_file_type(path)

    # 2. Select appropriate parser
    parser_cls = PARSER_REGISTRY.get(file_type)
    if not parser_cls:
        raise UnsupportedFileTypeError(f"No registered parser available for format '{file_type}'.")

    parser = parser_cls()

    # 3. Parse raw file contents into DataFrame / raw data
    raw_df = parser.parse(path)

    # 4. Normalize raw data into standardized schema
    normalized_df = normalize_data(raw_df)

    return normalized_df
