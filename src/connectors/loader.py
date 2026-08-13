import pandas as pd

from src.connectors.google_sheets import GoogleSheetsConnector
from src.connectors.notion import NotionConnector
from src.connectors.zendesk import ZendeskConnector
from src.connectors.intercom import IntercomConnector
from src.connectors.exceptions import ConnectorError
from src.preprocessing.normalizer import normalize_data

# Map source name to connector class
CONNECTOR_REGISTRY = {
    "google_sheets": GoogleSheetsConnector,
    "notion": NotionConnector,
    "zendesk": ZendeskConnector,
    "intercom": IntercomConnector,
}


def load_connector(source: str, **kwargs) -> pd.DataFrame:
    """
    Public entry point to fetch, parse, and normalize evidence data from an
    external source connector into a standardized pandas DataFrame.

    Mirrors `preprocessing.load_data()` in shape and contract:
        - `load_data("tickets.csv")` → load from file
        - `load_connector("google_sheets", spreadsheet_id="...")` → load from API

    Orchestration flow:
        1. Look up connector class by source name.
        2. Instantiate connector with provided kwargs.
        3. Call connector.fetch() → raw DataFrame.
        4. Pass raw DataFrame through normalize_data() → standard schema.

    Parameters:
        source (str): Connector name. One of: "google_sheets", "notion",
                      "zendesk", "intercom".
        **kwargs: Connector-specific arguments passed to the constructor
                  (e.g., spreadsheet_id, subdomain, api_key, config).

    Returns:
        pd.DataFrame: Normalized DataFrame with standard columns:
                      ['ticket_id', 'created_at', 'topic', 'message'].

    Raises:
        ConnectorError: If the source name is not recognized.
        AuthenticationError: If required credentials are missing.
        EmptyResponseError: If the source returns no data.
        InvalidSchemaError: If normalization fails.
    """
    connector_cls = CONNECTOR_REGISTRY.get(source)
    if not connector_cls:
        available = ", ".join(sorted(CONNECTOR_REGISTRY.keys()))
        raise ConnectorError(
            f"Unknown source '{source}'. Available connectors: {available}"
        )

    connector = connector_cls(**kwargs)
    raw_df = connector.fetch()
    normalized_df = normalize_data(raw_df)
    return normalized_df
