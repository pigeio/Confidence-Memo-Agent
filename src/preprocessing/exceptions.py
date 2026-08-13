class PreprocessingError(Exception):
    """Base exception class for all preprocessing and ingestion errors."""
    pass


class UnsupportedFileTypeError(PreprocessingError):
    """Raised when an unsupported file type or extension is provided."""
    pass


class EmptyFileError(PreprocessingError):
    """Raised when the input file is empty or contains no readable evidence records."""
    pass


class InvalidSchemaError(PreprocessingError):
    """Raised when input data cannot be normalized into a valid schema."""
    pass


class ParserError(PreprocessingError):
    """Raised when a specific file parser fails due to format corruption or syntax errors."""
    pass
