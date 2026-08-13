import requests
import pandas as pd

from src.connectors.base_connector import BaseConnector
from src.connectors.config import ConnectorConfig
from src.connectors.auth import resolve_credential
from src.connectors.exceptions import ConnectorError, EmptyResponseError
from src.connectors.pagination import paginate_cursor


class NotionConnector(BaseConnector):
    """
    Connector for Notion databases via the Notion API v1.

    Queries a Notion database, extracts page properties from Notion's
    nested JSON structure, and returns a flat raw pandas DataFrame.

    Credentials Resolution:
        1. Explicit `api_key` argument.
        2. NOTION_API_KEY environment variable in .env.

    Example:
        connector = NotionConnector(
            database_id="a1b2c3d4e5f6...",
        )
        raw_df = connector.fetch()
    """

    NOTION_API_VERSION = "2022-06-28"
    NOTION_BASE_URL = "https://api.notion.com/v1"

    def __init__(
        self,
        database_id: str,
        api_key: str = None,
        config: ConnectorConfig = None,
    ):
        """
        Initialize the Notion connector.

        Parameters:
            database_id (str): The Notion database ID to query.
            api_key (str | None): Notion integration API key (internal integration token).
            config (ConnectorConfig | None): Shared connector configuration.
        """
        super().__init__(config)
        self.database_id = database_id
        self.api_key = resolve_credential(
            "NOTION_API_KEY", api_key, "Notion API key"
        )

    def fetch(self, **kwargs) -> pd.DataFrame:
        """
        Query the Notion database and return all pages as a raw DataFrame.

        Each page's properties are flattened into a single row.
        Handles cursor-based pagination automatically.

        Returns:
            pd.DataFrame: Raw data from the Notion database.

        Raises:
            ConnectorError: If the API request fails.
            EmptyResponseError: If the database contains no pages.
        """
        return self._retry(self._fetch_database)

    def _fetch_database(self) -> pd.DataFrame:
        """Internal method that performs the actual Notion API calls."""
        url = f"{self.NOTION_BASE_URL}/databases/{self.database_id}/query"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Notion-Version": self.NOTION_API_VERSION,
            "Content-Type": "application/json",
        }

        try:
            records = paginate_cursor(
                url=url,
                headers=headers,
                body={},
                batch_size=self.config.batch_size,
                timeout=self.config.timeout,
                results_key="results",
                cursor_path=("next_cursor",),
                page_size_key="page_size",
                method="POST",
            )
        except Exception as e:
            raise ConnectorError(
                f"Failed to query Notion database '{self.database_id}': {e}"
            ) from e

        if not records:
            raise EmptyResponseError(
                f"Notion database '{self.database_id}' contains no pages."
            )

        # Flatten Notion page properties into simple key-value rows
        rows = [self._extract_properties(page) for page in records]
        df = pd.DataFrame(rows)
        return df

    @staticmethod
    def _extract_properties(page: dict) -> dict:
        """
        Extract property values from a Notion page object.

        Notion stores properties in a deeply nested format. This method
        flattens them into a simple {property_name: value} dictionary.

        Parameters:
            page (dict): A single Notion page object from the API response.

        Returns:
            dict: Flattened property key-value pairs.
        """
        row = {}
        properties = page.get("properties", {})

        for prop_name, prop_data in properties.items():
            prop_type = prop_data.get("type", "")
            value = None

            if prop_type == "title":
                title_parts = prop_data.get("title", [])
                value = "".join(
                    part.get("plain_text", "") for part in title_parts
                )

            elif prop_type == "rich_text":
                text_parts = prop_data.get("rich_text", [])
                value = "".join(
                    part.get("plain_text", "") for part in text_parts
                )

            elif prop_type == "number":
                value = prop_data.get("number")

            elif prop_type == "select":
                select = prop_data.get("select")
                value = select.get("name", "") if select else None

            elif prop_type == "multi_select":
                selections = prop_data.get("multi_select", [])
                value = ", ".join(s.get("name", "") for s in selections)

            elif prop_type == "date":
                date_obj = prop_data.get("date")
                value = date_obj.get("start", "") if date_obj else None

            elif prop_type == "checkbox":
                value = prop_data.get("checkbox", False)

            elif prop_type == "url":
                value = prop_data.get("url")

            elif prop_type == "email":
                value = prop_data.get("email")

            elif prop_type == "phone_number":
                value = prop_data.get("phone_number")

            elif prop_type == "created_time":
                value = prop_data.get("created_time")

            elif prop_type == "last_edited_time":
                value = prop_data.get("last_edited_time")

            elif prop_type == "status":
                status = prop_data.get("status")
                value = status.get("name", "") if status else None

            else:
                # Unsupported types stored as-is for transparency
                value = str(prop_data.get(prop_type, ""))

            row[prop_name] = value

        # Include page-level metadata
        row["_page_id"] = page.get("id", "")
        row["_created_time"] = page.get("created_time", "")

        return row
