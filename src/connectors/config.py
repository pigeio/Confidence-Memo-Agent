from dataclasses import dataclass


@dataclass
class ConnectorConfig:
    """
    Shared configuration for all external source connectors.

    Provides sensible defaults so connectors work out-of-the-box
    with zero configuration, while remaining fully customizable.

    Attributes:
        timeout (int): HTTP request timeout in seconds.
        retries (int): Maximum retry attempts on transient failures.
        backoff_factor (float): Exponential backoff multiplier (wait = factor * 2^attempt).
        batch_size (int): Records per page/batch for paginated API requests.
    """

    timeout: int = 30
    retries: int = 3
    backoff_factor: float = 1.0
    batch_size: int = 100
