import requests
import pandas as pd

from src.connectors.base_connector import BaseConnector
from src.connectors.config import ConnectorConfig
from src.connectors.auth import resolve_credential
from src.connectors.exceptions import ConnectorError, EmptyResponseError


class IntercomConnector(BaseConnector):
    """
    Connector for Intercom conversations via the Intercom REST API v2.11.

    Fetches conversations using bearer token authentication, handles
    cursor-based pagination, and returns a raw pandas DataFrame.

    Credentials Resolution:
        1. Explicit `api_token` argument.
        2. INTERCOM_TOKEN environment variable in .env.

    Example:
        connector = IntercomConnector()
        raw_df = connector.fetch()
    """

    INTERCOM_API_VERSION = "2.11"
    INTERCOM_BASE_URL = "https://api.intercom.io"

    def __init__(
        self,
        api_token: str = None,
        config: ConnectorConfig = None,
    ):
        """
        Initialize the Intercom connector.

        Parameters:
            api_token (str | None): Intercom API access token.
            config (ConnectorConfig | None): Shared connector configuration.
        """
        super().__init__(config)
        self.api_token = resolve_credential(
            "INTERCOM_TOKEN", api_token, "Intercom API token"
        )

    def fetch(self, **kwargs) -> pd.DataFrame:
        """
        Fetch all conversations from Intercom and return as a raw DataFrame.

        Handles cursor-based pagination (Intercom uses `starting_after` cursors).
        Extracts conversation source data including body, author, and timestamps.

        Returns:
            pd.DataFrame: Raw conversation data from Intercom.

        Raises:
            ConnectorError: If the API request fails.
            EmptyResponseError: If no conversations are found.
        """
        return self._retry(self._fetch_conversations)

    def _fetch_conversations(self) -> pd.DataFrame:
        """Internal method that performs the actual Intercom API calls."""
        url = f"{self.INTERCOM_BASE_URL}/conversations"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Accept": "application/json",
            "Intercom-Version": self.INTERCOM_API_VERSION,
        }

        try:
            records = self._paginate_intercom(url, headers)
        except ConnectorError:
            raise
        except Exception as e:
            raise ConnectorError(
                f"Failed to fetch Intercom conversations: {e}"
            ) from e

        if not records:
            raise EmptyResponseError(
                "Intercom returned no conversations."
            )

        # Extract conversation data into flat rows
        rows = []
        for conversation in records:
            source = conversation.get("source", {})
            rows.append(
                {
                    "ticket_id": conversation.get("id"),
                    "created_at": conversation.get("created_at"),
                    "topic": source.get("subject", "")
                    or conversation.get("title", ""),
                    "message": source.get("body", ""),
                }
            )

        df = pd.DataFrame(rows)
        return df

    def _paginate_intercom(
        self, url: str, headers: dict
    ) -> list[dict]:
        """
        Intercom-specific cursor-based pagination.

        Intercom returns a `pages.next.starting_after` cursor in each response.
        We follow it until the cursor is exhausted.

        Parameters:
            url (str): Initial API endpoint URL.
            headers (dict): HTTP headers (including auth).

        Returns:
            list[dict]: All conversation records.

        Raises:
            ConnectorError: If an HTTP request fails.
        """
        all_conversations = []
        params = {"per_page": self.config.batch_size}

        while True:
            try:
                response = requests.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=self.config.timeout,
                )
                response.raise_for_status()
            except requests.RequestException as e:
                raise ConnectorError(
                    f"Intercom API request failed: {e}"
                ) from e

            data = response.json()
            conversations = data.get("conversations", [])
            all_conversations.extend(conversations)

            # Intercom cursor: pages → next → starting_after
            pages = data.get("pages", {})
            next_page = pages.get("next")
            if not next_page:
                break

            starting_after = next_page.get("starting_after")
            if not starting_after:
                break

            params = {
                "per_page": self.config.batch_size,
                "starting_after": starting_after,
            }

        return all_conversations
