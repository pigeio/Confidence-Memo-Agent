import requests
import pandas as pd

from src.connectors.base_connector import BaseConnector
from src.connectors.config import ConnectorConfig
from src.connectors.auth import resolve_credential
from src.connectors.exceptions import ConnectorError, EmptyResponseError
from src.connectors.pagination import paginate_offset


class ZendeskConnector(BaseConnector):
    """
    Connector for Zendesk Support tickets via the Zendesk REST API v2.

    Fetches tickets from a Zendesk subdomain using email/token basic auth,
    handles offset-based pagination, and returns a raw pandas DataFrame.

    Credentials Resolution:
        1. Explicit `email` / `api_token` arguments.
        2. ZENDESK_EMAIL / ZENDESK_TOKEN environment variables in .env.

    Example:
        connector = ZendeskConnector(
            subdomain="mycompany",
        )
        raw_df = connector.fetch()
    """

    def __init__(
        self,
        subdomain: str,
        email: str = None,
        api_token: str = None,
        config: ConnectorConfig = None,
    ):
        """
        Initialize the Zendesk connector.

        Parameters:
            subdomain (str): Zendesk account subdomain (e.g. "mycompany"
                              for mycompany.zendesk.com).
            email (str | None): Zendesk agent email address.
            api_token (str | None): Zendesk API token.
            config (ConnectorConfig | None): Shared connector configuration.
        """
        super().__init__(config)
        self.subdomain = subdomain
        self.email = resolve_credential(
            "ZENDESK_EMAIL", email, "Zendesk email"
        )
        self.api_token = resolve_credential(
            "ZENDESK_TOKEN", api_token, "Zendesk API token"
        )

    def fetch(self, **kwargs) -> pd.DataFrame:
        """
        Fetch all tickets from the Zendesk instance and return as a raw DataFrame.

        Handles offset-based pagination (max 100 tickets per page).
        Maps Zendesk ticket fields to raw columns:
            id → ticket_id, created_at → created_at,
            subject → topic, description → message.

        Returns:
            pd.DataFrame: Raw ticket data from Zendesk.

        Raises:
            ConnectorError: If the API request fails.
            EmptyResponseError: If no tickets are found.
        """
        return self._retry(self._fetch_tickets)

    def _fetch_tickets(self) -> pd.DataFrame:
        """Internal method that performs the actual Zendesk API calls."""
        url = f"https://{self.subdomain}.zendesk.com/api/v2/tickets.json"
        headers = {"Content-Type": "application/json"}

        # Zendesk uses basic auth: {email}/token:{api_token}
        auth = (f"{self.email}/token", self.api_token)

        try:
            records = self._paginate_zendesk(url, headers, auth)
        except ConnectorError:
            raise
        except Exception as e:
            raise ConnectorError(
                f"Failed to fetch Zendesk tickets from '{self.subdomain}': {e}"
            ) from e

        if not records:
            raise EmptyResponseError(
                f"Zendesk instance '{self.subdomain}' returned no tickets."
            )

        # Map Zendesk fields to raw column names
        rows = []
        for ticket in records:
            rows.append(
                {
                    "ticket_id": ticket.get("id"),
                    "created_at": ticket.get("created_at"),
                    "topic": ticket.get("subject", ""),
                    "message": ticket.get("description", ""),
                }
            )

        df = pd.DataFrame(rows)
        return df

    def _paginate_zendesk(
        self, url: str, headers: dict, auth: tuple
    ) -> list[dict]:
        """
        Zendesk-specific pagination using the `next_page` URL pattern.

        Zendesk returns a `next_page` URL in each response. We follow it
        until it becomes null, collecting all ticket records.

        Parameters:
            url (str): Initial API endpoint URL.
            headers (dict): HTTP headers.
            auth (tuple): Basic auth credentials (email/token, api_token).

        Returns:
            list[dict]: All ticket records.

        Raises:
            ConnectorError: If an HTTP request fails.
        """
        all_tickets = []
        current_url = url
        params = {"per_page": self.config.batch_size}

        while current_url:
            try:
                response = requests.get(
                    current_url,
                    headers=headers,
                    auth=auth,
                    params=params,
                    timeout=self.config.timeout,
                )
                response.raise_for_status()
            except requests.RequestException as e:
                raise ConnectorError(
                    f"Zendesk API request failed: {e}"
                ) from e

            data = response.json()
            tickets = data.get("tickets", [])
            all_tickets.extend(tickets)

            # Zendesk provides the full next URL or null
            current_url = data.get("next_page")
            # Only send per_page on first request; next_page URL includes it
            params = None

        return all_tickets
