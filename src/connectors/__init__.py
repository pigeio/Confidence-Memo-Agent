"""
Confidence Memo Agent — Connectors Package
External Source Connector Adapter Layer

Provides a unified interface for fetching evidence data from external
services (Google Sheets, Notion, Zendesk, Intercom) and normalizing
it into the same canonical DataFrame schema used by file-based ingestion.

Usage:
    from src.connectors import load_connector

    # Load from Google Sheets
    df = load_connector("google_sheets", spreadsheet_id="...")

    # Load from Zendesk
    df = load_connector("zendesk", subdomain="mycompany")
"""

from src.connectors.loader import load_connector
from src.connectors.base_connector import BaseConnector
from src.connectors.config import ConnectorConfig
from src.connectors.google_sheets import GoogleSheetsConnector
from src.connectors.notion import NotionConnector
from src.connectors.zendesk import ZendeskConnector
from src.connectors.intercom import IntercomConnector
from src.connectors.exceptions import (
    ConnectorError,
    AuthenticationError,
    RateLimitError,
    EmptyResponseError,
    PaginationError,
)

__all__ = [
    "load_connector",
    "BaseConnector",
    "ConnectorConfig",
    "GoogleSheetsConnector",
    "NotionConnector",
    "ZendeskConnector",
    "IntercomConnector",
    "ConnectorError",
    "AuthenticationError",
    "RateLimitError",
    "EmptyResponseError",
    "PaginationError",
]
