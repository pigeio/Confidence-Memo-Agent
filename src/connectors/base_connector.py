from abc import ABC, abstractmethod
import time

import pandas as pd

from src.connectors.config import ConnectorConfig
from src.connectors.exceptions import ConnectorError


class BaseConnector(ABC):
    """
    Abstract base class for all external source connectors.

    Every connector must implement `fetch()` which returns a **raw** DataFrame.
    Normalization is handled by the orchestration layer (loader.py), not by connectors.

    Provides shared infrastructure:
        - Configurable timeout, retries, backoff, batch_size via ConnectorConfig.
        - `_retry()` for automatic retry with exponential backoff on transient failures.
    """

    def __init__(self, config: ConnectorConfig = None):
        """
        Initialize the connector with shared configuration.

        Parameters:
            config (ConnectorConfig | None): Shared connector settings.
                                              Uses sensible defaults if not provided.
        """
        self.config = config or ConnectorConfig()

    @abstractmethod
    def fetch(self, **kwargs) -> pd.DataFrame:
        """
        Fetch raw data from the external source.

        Returns:
            pd.DataFrame: Raw data as-is from the source. Column names should
                          reflect the source's native schema — the normalizer
                          will handle alias mapping downstream.

        Raises:
            ConnectorError: On any failure during data fetching.
        """
        ...

    def _retry(self, func, *args, **kwargs):
        """
        Execute a callable with automatic retry and exponential backoff.

        Retries up to `self.config.retries` times on ConnectorError exceptions.
        Wait time between retries follows exponential backoff:
            wait = backoff_factor * (2 ^ attempt)

        Parameters:
            func (callable): The function to execute.
            *args: Positional arguments passed to func.
            **kwargs: Keyword arguments passed to func.

        Returns:
            The return value of func on success.

        Raises:
            ConnectorError: The last error if all retries are exhausted.
        """
        last_error = None
        for attempt in range(self.config.retries):
            try:
                return func(*args, **kwargs)
            except ConnectorError as e:
                last_error = e
                if attempt < self.config.retries - 1:
                    self._backoff(attempt)
        raise last_error

    def _backoff(self, attempt: int):
        """
        Sleep for an exponentially increasing duration.

        Wait time: backoff_factor * (2 ^ attempt) seconds.

        Parameters:
            attempt (int): Zero-indexed retry attempt number.
        """
        wait = self.config.backoff_factor * (2 ** attempt)
        time.sleep(wait)
