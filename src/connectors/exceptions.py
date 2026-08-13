class ConnectorError(Exception):
    """Base exception class for all connector failures."""
    pass


class AuthenticationError(ConnectorError):
    """Raised when API credentials are missing or invalid."""
    pass


class RateLimitError(ConnectorError):
    """Raised when the external API rate-limits the request."""
    pass


class EmptyResponseError(ConnectorError):
    """Raised when the external source returns no usable data."""
    pass


class PaginationError(ConnectorError):
    """Raised when pagination fails mid-stream (e.g. invalid cursor, unexpected response)."""
    pass
