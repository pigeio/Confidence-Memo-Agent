"""
Tests for Sprint 3.4 — Source Connectors Package.

All external API calls are mocked. No real credentials needed.
"""

import pytest
import pandas as pd
import time
from unittest.mock import patch, MagicMock, PropertyMock
from dataclasses import dataclass

# ─── Infrastructure imports ───────────────────────────────────────────
from src.connectors.config import ConnectorConfig
from src.connectors.exceptions import (
    ConnectorError,
    AuthenticationError,
    RateLimitError,
    EmptyResponseError,
    PaginationError,
)
from src.connectors.auth import resolve_credential
from src.connectors.base_connector import BaseConnector
from src.connectors.pagination import paginate_offset, paginate_cursor

# ─── Connector imports ────────────────────────────────────────────────
from src.connectors.google_sheets import GoogleSheetsConnector
from src.connectors.notion import NotionConnector
from src.connectors.zendesk import ZendeskConnector
from src.connectors.intercom import IntercomConnector
from src.connectors.loader import load_connector, CONNECTOR_REGISTRY


# ═════════════════════════════════════════════════════════════════════
# SECTION 1: ConnectorConfig
# ═════════════════════════════════════════════════════════════════════


class TestConnectorConfig:
    """Tests for the ConnectorConfig dataclass."""

    def test_default_values(self):
        config = ConnectorConfig()
        assert config.timeout == 30
        assert config.retries == 3
        assert config.backoff_factor == 1.0
        assert config.batch_size == 100

    def test_custom_values(self):
        config = ConnectorConfig(timeout=60, retries=5, backoff_factor=2.0, batch_size=50)
        assert config.timeout == 60
        assert config.retries == 5
        assert config.backoff_factor == 2.0
        assert config.batch_size == 50

    def test_partial_override(self):
        config = ConnectorConfig(timeout=10)
        assert config.timeout == 10
        assert config.retries == 3  # default preserved


# ═════════════════════════════════════════════════════════════════════
# SECTION 2: Exceptions
# ═════════════════════════════════════════════════════════════════════


class TestExceptions:
    """Tests for the connector exception hierarchy."""

    def test_connector_error_is_base(self):
        assert issubclass(AuthenticationError, ConnectorError)
        assert issubclass(RateLimitError, ConnectorError)
        assert issubclass(EmptyResponseError, ConnectorError)
        assert issubclass(PaginationError, ConnectorError)

    def test_connector_error_is_exception(self):
        assert issubclass(ConnectorError, Exception)

    def test_error_messages(self):
        err = AuthenticationError("Missing API key")
        assert str(err) == "Missing API key"

    def test_raise_and_catch_hierarchy(self):
        with pytest.raises(ConnectorError):
            raise AuthenticationError("test")

        with pytest.raises(ConnectorError):
            raise RateLimitError("limit hit")

        with pytest.raises(ConnectorError):
            raise EmptyResponseError("no data")

        with pytest.raises(ConnectorError):
            raise PaginationError("cursor lost")


# ═════════════════════════════════════════════════════════════════════
# SECTION 3: Authentication Helpers
# ═════════════════════════════════════════════════════════════════════


class TestResolveCredential:
    """Tests for the resolve_credential() function."""

    def test_explicit_value_takes_priority(self):
        with patch.dict("os.environ", {"MY_KEY": "env_value"}):
            result = resolve_credential("MY_KEY", explicit_value="explicit_value")
            assert result == "explicit_value"

    def test_falls_back_to_env_var(self):
        with patch.dict("os.environ", {"MY_KEY": "env_value"}):
            result = resolve_credential("MY_KEY")
            assert result == "env_value"

    def test_raises_authentication_error_when_missing(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(AuthenticationError, match="Missing test credential"):
                resolve_credential("NONEXISTENT_VAR", label="test credential")

    def test_error_message_includes_env_var_name(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(AuthenticationError, match="SOME_API_KEY"):
                resolve_credential("SOME_API_KEY", label="API key")

    def test_empty_string_explicit_value_falls_to_env(self):
        """Empty string is falsy, should fall through to env var."""
        with patch.dict("os.environ", {"MY_KEY": "env_value"}):
            result = resolve_credential("MY_KEY", explicit_value="")
            assert result == "env_value"

    def test_none_explicit_value_falls_to_env(self):
        with patch.dict("os.environ", {"MY_KEY": "env_value"}):
            result = resolve_credential("MY_KEY", explicit_value=None)
            assert result == "env_value"


# ═════════════════════════════════════════════════════════════════════
# SECTION 4: BaseConnector (ABC, Retry, Backoff)
# ═════════════════════════════════════════════════════════════════════


class TestBaseConnector:
    """Tests for the BaseConnector abstract base class."""

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseConnector()

    def test_subclass_must_implement_fetch(self):
        class IncompleteConnector(BaseConnector):
            pass

        with pytest.raises(TypeError):
            IncompleteConnector()

    def test_subclass_with_fetch_works(self):
        class SimpleConnector(BaseConnector):
            def fetch(self, **kwargs):
                return pd.DataFrame({"a": [1, 2]})

        connector = SimpleConnector()
        df = connector.fetch()
        assert len(df) == 2

    def test_default_config(self):
        class SimpleConnector(BaseConnector):
            def fetch(self, **kwargs):
                return pd.DataFrame()

        connector = SimpleConnector()
        assert connector.config.timeout == 30
        assert connector.config.retries == 3

    def test_custom_config(self):
        class SimpleConnector(BaseConnector):
            def fetch(self, **kwargs):
                return pd.DataFrame()

        config = ConnectorConfig(timeout=60, retries=5)
        connector = SimpleConnector(config=config)
        assert connector.config.timeout == 60
        assert connector.config.retries == 5


class TestRetryMechanism:
    """Tests for the _retry() and _backoff() methods."""

    def test_retry_succeeds_on_first_attempt(self):
        class SimpleConnector(BaseConnector):
            def fetch(self, **kwargs):
                return pd.DataFrame()

        connector = SimpleConnector()
        mock_func = MagicMock(return_value="success")
        result = connector._retry(mock_func)
        assert result == "success"
        assert mock_func.call_count == 1

    def test_retry_succeeds_after_failures(self):
        class SimpleConnector(BaseConnector):
            def fetch(self, **kwargs):
                return pd.DataFrame()

        config = ConnectorConfig(retries=3, backoff_factor=0.001)
        connector = SimpleConnector(config=config)

        mock_func = MagicMock(
            side_effect=[ConnectorError("fail1"), ConnectorError("fail2"), "success"]
        )
        result = connector._retry(mock_func)
        assert result == "success"
        assert mock_func.call_count == 3

    def test_retry_exhausted_raises_last_error(self):
        class SimpleConnector(BaseConnector):
            def fetch(self, **kwargs):
                return pd.DataFrame()

        config = ConnectorConfig(retries=2, backoff_factor=0.001)
        connector = SimpleConnector(config=config)

        mock_func = MagicMock(
            side_effect=[ConnectorError("fail1"), ConnectorError("fail2")]
        )
        with pytest.raises(ConnectorError, match="fail2"):
            connector._retry(mock_func)

    def test_retry_passes_args_and_kwargs(self):
        class SimpleConnector(BaseConnector):
            def fetch(self, **kwargs):
                return pd.DataFrame()

        connector = SimpleConnector()
        mock_func = MagicMock(return_value="ok")
        connector._retry(mock_func, "arg1", key="val")
        mock_func.assert_called_once_with("arg1", key="val")

    @patch("src.connectors.base_connector.time.sleep")
    def test_backoff_calculates_correct_wait(self, mock_sleep):
        class SimpleConnector(BaseConnector):
            def fetch(self, **kwargs):
                return pd.DataFrame()

        config = ConnectorConfig(backoff_factor=1.0)
        connector = SimpleConnector(config=config)

        connector._backoff(0)  # 1.0 * 2^0 = 1.0
        mock_sleep.assert_called_with(1.0)

        connector._backoff(1)  # 1.0 * 2^1 = 2.0
        mock_sleep.assert_called_with(2.0)

        connector._backoff(2)  # 1.0 * 2^2 = 4.0
        mock_sleep.assert_called_with(4.0)

    @patch("src.connectors.base_connector.time.sleep")
    def test_backoff_respects_factor(self, mock_sleep):
        class SimpleConnector(BaseConnector):
            def fetch(self, **kwargs):
                return pd.DataFrame()

        config = ConnectorConfig(backoff_factor=0.5)
        connector = SimpleConnector(config=config)

        connector._backoff(0)  # 0.5 * 2^0 = 0.5
        mock_sleep.assert_called_with(0.5)

        connector._backoff(2)  # 0.5 * 2^2 = 2.0
        mock_sleep.assert_called_with(2.0)


# ═════════════════════════════════════════════════════════════════════
# SECTION 5: Pagination Helpers
# ═════════════════════════════════════════════════════════════════════


class TestPaginateOffset:
    """Tests for the offset-based pagination helper."""

    @patch("src.connectors.pagination.requests.get")
    def test_single_page(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [{"id": 1}, {"id": 2}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        results = paginate_offset(
            url="https://api.example.com/data",
            headers={"Auth": "token"},
            batch_size=100,
            timeout=30,
            results_key="results",
        )
        assert len(results) == 2
        assert results[0]["id"] == 1

    @patch("src.connectors.pagination.requests.get")
    def test_multiple_pages(self, mock_get):
        page1 = MagicMock()
        page1.json.return_value = {"results": [{"id": i} for i in range(100)]}
        page1.raise_for_status = MagicMock()

        page2 = MagicMock()
        page2.json.return_value = {"results": [{"id": 100 + i} for i in range(50)]}
        page2.raise_for_status = MagicMock()

        mock_get.side_effect = [page1, page2]

        results = paginate_offset(
            url="https://api.example.com/data",
            headers={},
            batch_size=100,
            timeout=30,
            results_key="results",
        )
        assert len(results) == 150

    @patch("src.connectors.pagination.requests.get")
    def test_empty_results(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        results = paginate_offset(
            url="https://api.example.com/data",
            headers={},
            batch_size=100,
            timeout=30,
            results_key="results",
        )
        assert len(results) == 0

    @patch("src.connectors.pagination.requests.get")
    def test_http_error_raises_pagination_error(self, mock_get):
        import requests
        mock_get.side_effect = requests.RequestException("connection failed")

        with pytest.raises(PaginationError, match="connection failed"):
            paginate_offset(
                url="https://api.example.com/data",
                headers={},
                batch_size=100,
                timeout=30,
                results_key="results",
            )


class TestPaginateCursor:
    """Tests for the cursor-based pagination helper."""

    @patch("src.connectors.pagination.requests.post")
    def test_single_page_post(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [{"id": 1}],
            "next_cursor": None,
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        results = paginate_cursor(
            url="https://api.example.com/query",
            headers={},
            body={},
            batch_size=100,
            timeout=30,
            results_key="results",
            cursor_path=("next_cursor",),
            method="POST",
        )
        assert len(results) == 1

    @patch("src.connectors.pagination.requests.post")
    def test_multiple_pages_cursor(self, mock_post):
        page1 = MagicMock()
        page1.json.return_value = {
            "results": [{"id": 1}],
            "next_cursor": "cursor_abc",
        }
        page1.raise_for_status = MagicMock()

        page2 = MagicMock()
        page2.json.return_value = {
            "results": [{"id": 2}],
            "next_cursor": None,
        }
        page2.raise_for_status = MagicMock()

        mock_post.side_effect = [page1, page2]

        results = paginate_cursor(
            url="https://api.example.com/query",
            headers={},
            body={},
            batch_size=100,
            timeout=30,
            results_key="results",
            cursor_path=("next_cursor",),
            method="POST",
        )
        assert len(results) == 2

    @patch("src.connectors.pagination.requests.get")
    def test_get_method(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [{"id": 1}],
            "next_cursor": None,
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        results = paginate_cursor(
            url="https://api.example.com/data",
            headers={},
            batch_size=50,
            timeout=15,
            results_key="results",
            cursor_path=("next_cursor",),
            method="GET",
        )
        assert len(results) == 1

    @patch("src.connectors.pagination.requests.post")
    def test_nested_cursor_path(self, mock_post):
        page1 = MagicMock()
        page1.json.return_value = {
            "results": [{"id": 1}],
            "pages": {"next": "cursor_xyz"},
        }
        page1.raise_for_status = MagicMock()

        page2 = MagicMock()
        page2.json.return_value = {
            "results": [{"id": 2}],
            "pages": {"next": None},
        }
        page2.raise_for_status = MagicMock()

        mock_post.side_effect = [page1, page2]

        results = paginate_cursor(
            url="https://api.example.com/query",
            headers={},
            body={},
            batch_size=100,
            timeout=30,
            results_key="results",
            cursor_path=("pages", "next"),
            method="POST",
        )
        assert len(results) == 2


# ═════════════════════════════════════════════════════════════════════
# SECTION 6: GoogleSheetsConnector
# ═════════════════════════════════════════════════════════════════════


class TestGoogleSheetsConnector:
    """Tests for GoogleSheetsConnector with mocked gspread."""

    def test_missing_credentials_raises_auth_error(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(AuthenticationError, match="Google service account"):
                GoogleSheetsConnector(spreadsheet_id="test_id")

    @patch.dict("os.environ", {"GOOGLE_SHEETS_CREDENTIALS": "/path/to/creds.json"})
    @patch("src.connectors.google_sheets.gspread")
    @patch("src.connectors.google_sheets.Credentials")
    def test_fetch_returns_raw_dataframe(self, mock_creds_cls, mock_gspread):
        """Mock the gspread library to test fetch logic."""
        # Setup mock chain: Credentials → authorize → open_by_key → worksheet → get_all_records
        mock_creds = MagicMock()
        mock_creds_cls.from_service_account_file.return_value = mock_creds

        mock_worksheet = MagicMock()
        mock_worksheet.get_all_records.return_value = [
            {"id": 1, "subject": "Bug", "text": "App crashes on login"},
            {"id": 2, "subject": "Feature", "text": "Add dark mode"},
        ]
        mock_spreadsheet = MagicMock()
        mock_spreadsheet.worksheet.return_value = mock_worksheet
        mock_client = MagicMock()
        mock_client.open_by_key.return_value = mock_spreadsheet
        mock_gspread.authorize.return_value = mock_client

        connector = GoogleSheetsConnector(spreadsheet_id="test_id")
        df = connector.fetch()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "id" in df.columns
        assert "subject" in df.columns
        assert "text" in df.columns

    @patch.dict("os.environ", {"GOOGLE_SHEETS_CREDENTIALS": "/path/to/creds.json"})
    @patch("src.connectors.google_sheets.gspread")
    @patch("src.connectors.google_sheets.Credentials")
    def test_empty_sheet_raises_empty_response(self, mock_creds_cls, mock_gspread):
        mock_creds = MagicMock()
        mock_creds_cls.from_service_account_file.return_value = mock_creds

        mock_worksheet = MagicMock()
        mock_worksheet.get_all_records.return_value = []
        mock_spreadsheet = MagicMock()
        mock_spreadsheet.worksheet.return_value = mock_worksheet
        mock_client = MagicMock()
        mock_client.open_by_key.return_value = mock_spreadsheet
        mock_gspread.authorize.return_value = mock_client

        connector = GoogleSheetsConnector(spreadsheet_id="test_id")
        with pytest.raises(EmptyResponseError, match="no data rows"):
            connector.fetch()

    @patch.dict("os.environ", {"GOOGLE_SHEETS_CREDENTIALS": "/path/to/creds.json"})
    def test_custom_sheet_name(self):
        connector = GoogleSheetsConnector(
            spreadsheet_id="test_id", sheet_name="Feedback"
        )
        assert connector.sheet_name == "Feedback"

    @patch.dict("os.environ", {"GOOGLE_SHEETS_CREDENTIALS": "/path/to/creds.json"})
    def test_custom_config(self):
        config = ConnectorConfig(timeout=60, retries=5)
        connector = GoogleSheetsConnector(
            spreadsheet_id="test_id", config=config
        )
        assert connector.config.timeout == 60
        assert connector.config.retries == 5


# ═════════════════════════════════════════════════════════════════════
# SECTION 7: NotionConnector
# ═════════════════════════════════════════════════════════════════════


class TestNotionConnector:
    """Tests for NotionConnector with mocked HTTP."""

    def test_missing_api_key_raises_auth_error(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(AuthenticationError, match="Notion API key"):
                NotionConnector(database_id="test_db")

    @patch.dict("os.environ", {"NOTION_API_KEY": "secret_key"})
    @patch("src.connectors.pagination.requests.post")
    def test_fetch_returns_raw_dataframe(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "id": "page1",
                    "created_time": "2025-01-01T00:00:00.000Z",
                    "properties": {
                        "Name": {
                            "type": "title",
                            "title": [{"plain_text": "Bug Report"}],
                        },
                        "Description": {
                            "type": "rich_text",
                            "rich_text": [{"plain_text": "App crashes on startup"}],
                        },
                        "Status": {
                            "type": "select",
                            "select": {"name": "Open"},
                        },
                    },
                }
            ],
            "next_cursor": None,
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        connector = NotionConnector(database_id="test_db")
        df = connector.fetch()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0]["Name"] == "Bug Report"
        assert df.iloc[0]["Description"] == "App crashes on startup"
        assert df.iloc[0]["Status"] == "Open"

    @patch.dict("os.environ", {"NOTION_API_KEY": "secret_key"})
    @patch("src.connectors.pagination.requests.post")
    def test_empty_database_raises_empty_response(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [],
            "next_cursor": None,
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        connector = NotionConnector(database_id="test_db")
        with pytest.raises(EmptyResponseError, match="no pages"):
            connector.fetch()

    def test_extract_properties_handles_all_types(self):
        page = {
            "id": "page123",
            "created_time": "2025-06-15T12:00:00.000Z",
            "properties": {
                "Title": {"type": "title", "title": [{"plain_text": "Test"}]},
                "Notes": {"type": "rich_text", "rich_text": [{"plain_text": "Hello"}]},
                "Count": {"type": "number", "number": 42},
                "Priority": {"type": "select", "select": {"name": "High"}},
                "Tags": {"type": "multi_select", "multi_select": [{"name": "bug"}, {"name": "ui"}]},
                "Due": {"type": "date", "date": {"start": "2025-07-01"}},
                "Done": {"type": "checkbox", "checkbox": True},
                "Link": {"type": "url", "url": "https://example.com"},
                "Email": {"type": "email", "email": "test@test.com"},
                "Phone": {"type": "phone_number", "phone_number": "+1234567890"},
                "Created": {"type": "created_time", "created_time": "2025-01-01T00:00:00.000Z"},
            },
        }

        row = NotionConnector._extract_properties(page)

        assert row["Title"] == "Test"
        assert row["Notes"] == "Hello"
        assert row["Count"] == 42
        assert row["Priority"] == "High"
        assert row["Tags"] == "bug, ui"
        assert row["Due"] == "2025-07-01"
        assert row["Done"] is True
        assert row["Link"] == "https://example.com"
        assert row["Email"] == "test@test.com"
        assert row["Phone"] == "+1234567890"
        assert row["_page_id"] == "page123"

    def test_extract_properties_handles_null_select(self):
        page = {
            "id": "page123",
            "created_time": "",
            "properties": {
                "Status": {"type": "select", "select": None},
                "Date": {"type": "date", "date": None},
            },
        }
        row = NotionConnector._extract_properties(page)
        assert row["Status"] is None
        assert row["Date"] is None


# ═════════════════════════════════════════════════════════════════════
# SECTION 8: ZendeskConnector
# ═════════════════════════════════════════════════════════════════════


class TestZendeskConnector:
    """Tests for ZendeskConnector with mocked HTTP."""

    def test_missing_credentials_raises_auth_error(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(AuthenticationError):
                ZendeskConnector(subdomain="test")

    @patch.dict("os.environ", {"ZENDESK_EMAIL": "agent@test.com", "ZENDESK_TOKEN": "zd_token"})
    @patch("src.connectors.zendesk.requests.get")
    def test_fetch_returns_raw_dataframe(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "tickets": [
                {
                    "id": 101,
                    "created_at": "2025-01-15T10:00:00Z",
                    "subject": "Login broken",
                    "description": "Cannot log in with SSO",
                },
                {
                    "id": 102,
                    "created_at": "2025-01-16T11:00:00Z",
                    "subject": "Slow response",
                    "description": "API takes 10 seconds to respond",
                },
            ],
            "next_page": None,
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        connector = ZendeskConnector(subdomain="testco")
        df = connector.fetch()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["ticket_id", "created_at", "topic", "message"]
        assert df.iloc[0]["ticket_id"] == 101
        assert df.iloc[0]["topic"] == "Login broken"
        assert df.iloc[0]["message"] == "Cannot log in with SSO"

    @patch.dict("os.environ", {"ZENDESK_EMAIL": "agent@test.com", "ZENDESK_TOKEN": "zd_token"})
    @patch("src.connectors.zendesk.requests.get")
    def test_pagination_follows_next_page(self, mock_get):
        page1 = MagicMock()
        page1.json.return_value = {
            "tickets": [{"id": 1, "created_at": "2025-01-01", "subject": "A", "description": "Desc A"}],
            "next_page": "https://testco.zendesk.com/api/v2/tickets.json?page=2",
        }
        page1.raise_for_status = MagicMock()

        page2 = MagicMock()
        page2.json.return_value = {
            "tickets": [{"id": 2, "created_at": "2025-01-02", "subject": "B", "description": "Desc B"}],
            "next_page": None,
        }
        page2.raise_for_status = MagicMock()

        mock_get.side_effect = [page1, page2]

        connector = ZendeskConnector(subdomain="testco")
        df = connector.fetch()

        assert len(df) == 2
        assert mock_get.call_count == 2

    @patch.dict("os.environ", {"ZENDESK_EMAIL": "agent@test.com", "ZENDESK_TOKEN": "zd_token"})
    @patch("src.connectors.zendesk.requests.get")
    def test_empty_tickets_raises_empty_response(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {"tickets": [], "next_page": None}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        connector = ZendeskConnector(subdomain="testco")
        with pytest.raises(EmptyResponseError, match="no tickets"):
            connector.fetch()

    @patch.dict("os.environ", {"ZENDESK_EMAIL": "agent@test.com", "ZENDESK_TOKEN": "zd_token"})
    @patch("src.connectors.zendesk.requests.get")
    def test_http_error_raises_connector_error(self, mock_get):
        import requests
        mock_get.side_effect = requests.RequestException("timeout")

        config = ConnectorConfig(retries=1, backoff_factor=0.001)
        connector = ZendeskConnector(subdomain="testco", config=config)
        with pytest.raises(ConnectorError):
            connector.fetch()


# ═════════════════════════════════════════════════════════════════════
# SECTION 9: IntercomConnector
# ═════════════════════════════════════════════════════════════════════


class TestIntercomConnector:
    """Tests for IntercomConnector with mocked HTTP."""

    def test_missing_token_raises_auth_error(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(AuthenticationError, match="Intercom API token"):
                IntercomConnector()

    @patch.dict("os.environ", {"INTERCOM_TOKEN": "ic_token"})
    @patch("src.connectors.intercom.requests.get")
    def test_fetch_returns_raw_dataframe(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "conversations": [
                {
                    "id": "conv1",
                    "created_at": 1705000000,
                    "title": "Help needed",
                    "source": {
                        "subject": "Cannot export data",
                        "body": "Export button does nothing when clicked",
                    },
                },
            ],
            "pages": {"next": None},
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        connector = IntercomConnector()
        df = connector.fetch()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0]["ticket_id"] == "conv1"
        assert df.iloc[0]["topic"] == "Cannot export data"
        assert df.iloc[0]["message"] == "Export button does nothing when clicked"

    @patch.dict("os.environ", {"INTERCOM_TOKEN": "ic_token"})
    @patch("src.connectors.intercom.requests.get")
    def test_cursor_pagination(self, mock_get):
        page1 = MagicMock()
        page1.json.return_value = {
            "conversations": [
                {"id": "c1", "created_at": 1705000000, "title": "", "source": {"subject": "A", "body": "Body A"}},
            ],
            "pages": {"next": {"starting_after": "cursor123"}},
        }
        page1.raise_for_status = MagicMock()

        page2 = MagicMock()
        page2.json.return_value = {
            "conversations": [
                {"id": "c2", "created_at": 1705100000, "title": "", "source": {"subject": "B", "body": "Body B"}},
            ],
            "pages": {"next": None},
        }
        page2.raise_for_status = MagicMock()

        mock_get.side_effect = [page1, page2]

        connector = IntercomConnector()
        df = connector.fetch()

        assert len(df) == 2
        assert mock_get.call_count == 2

    @patch.dict("os.environ", {"INTERCOM_TOKEN": "ic_token"})
    @patch("src.connectors.intercom.requests.get")
    def test_empty_conversations_raises_empty_response(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "conversations": [],
            "pages": {"next": None},
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        connector = IntercomConnector()
        with pytest.raises(EmptyResponseError, match="no conversations"):
            connector.fetch()

    @patch.dict("os.environ", {"INTERCOM_TOKEN": "ic_token"})
    @patch("src.connectors.intercom.requests.get")
    def test_fallback_topic_from_title(self, mock_get):
        """If source.subject is empty, falls back to conversation title."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "conversations": [
                {
                    "id": "c1",
                    "created_at": 1705000000,
                    "title": "Billing question",
                    "source": {"subject": "", "body": "How do I upgrade?"},
                },
            ],
            "pages": {"next": None},
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        connector = IntercomConnector()
        df = connector.fetch()
        assert df.iloc[0]["topic"] == "Billing question"


# ═════════════════════════════════════════════════════════════════════
# SECTION 10: Connector Loader (load_connector)
# ═════════════════════════════════════════════════════════════════════


class TestConnectorLoader:
    """Tests for the load_connector() entry point and registry."""

    def test_registry_contains_all_connectors(self):
        assert "google_sheets" in CONNECTOR_REGISTRY
        assert "notion" in CONNECTOR_REGISTRY
        assert "zendesk" in CONNECTOR_REGISTRY
        assert "intercom" in CONNECTOR_REGISTRY

    def test_registry_maps_to_correct_classes(self):
        assert CONNECTOR_REGISTRY["google_sheets"] is GoogleSheetsConnector
        assert CONNECTOR_REGISTRY["notion"] is NotionConnector
        assert CONNECTOR_REGISTRY["zendesk"] is ZendeskConnector
        assert CONNECTOR_REGISTRY["intercom"] is IntercomConnector

    def test_unknown_source_raises_connector_error(self):
        with pytest.raises(ConnectorError, match="Unknown source 'twitter'"):
            load_connector("twitter")

    @patch.dict("os.environ", {"ZENDESK_EMAIL": "a@b.com", "ZENDESK_TOKEN": "tok"})
    @patch("src.connectors.zendesk.requests.get")
    def test_load_connector_returns_normalized_dataframe(self, mock_get):
        """End-to-end: load_connector() → fetch → normalize → standard schema."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "tickets": [
                {
                    "id": 1,
                    "created_at": "2025-01-15T10:00:00Z",
                    "subject": "Login issue",
                    "description": "Users cannot log in after password reset",
                },
            ],
            "next_page": None,
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        df = load_connector("zendesk", subdomain="testco")

        # Verify standard normalized schema
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["ticket_id", "created_at", "topic", "message"]
        assert len(df) == 1
        assert df.iloc[0]["message"] == "Users cannot log in after password reset"

    @patch.dict("os.environ", {"INTERCOM_TOKEN": "ic_token"})
    @patch("src.connectors.intercom.requests.get")
    def test_load_connector_intercom_normalized(self, mock_get):
        """Verify Intercom data also normalizes to the standard schema."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "conversations": [
                {
                    "id": "conv1",
                    "created_at": 1705000000,
                    "title": "",
                    "source": {
                        "subject": "Export issue",
                        "body": "CSV export fails on large datasets",
                    },
                },
            ],
            "pages": {"next": None},
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        df = load_connector("intercom")

        assert list(df.columns) == ["ticket_id", "created_at", "topic", "message"]
        assert df.iloc[0]["message"] == "CSV export fails on large datasets"


# ═════════════════════════════════════════════════════════════════════
# SECTION 11: Package-Level Imports
# ═════════════════════════════════════════════════════════════════════


class TestPackageImports:
    """Verify all public names are importable from the package."""

    def test_public_api_imports(self):
        from src.connectors import (
            load_connector,
            BaseConnector,
            ConnectorConfig,
            GoogleSheetsConnector,
            NotionConnector,
            ZendeskConnector,
            IntercomConnector,
            ConnectorError,
            AuthenticationError,
            RateLimitError,
            EmptyResponseError,
            PaginationError,
        )
        # Verify they are the correct types
        assert callable(load_connector)
        assert issubclass(GoogleSheetsConnector, BaseConnector)
        assert issubclass(NotionConnector, BaseConnector)
        assert issubclass(ZendeskConnector, BaseConnector)
        assert issubclass(IntercomConnector, BaseConnector)
