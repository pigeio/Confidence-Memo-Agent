import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

from src.connectors.base_connector import BaseConnector
from src.connectors.config import ConnectorConfig
from src.connectors.auth import resolve_credential
from src.connectors.exceptions import ConnectorError, EmptyResponseError


class GoogleSheetsConnector(BaseConnector):
    """
    Connector for Google Sheets via the Google Sheets API v4.

    Authenticates using a Google service account JSON credentials file,
    opens the specified spreadsheet, reads all rows from the target sheet,
    and returns a raw pandas DataFrame.

    Credentials Resolution:
        1. Explicit `credentials_path` argument.
        2. GOOGLE_SHEETS_CREDENTIALS environment variable in .env.

    Example:
        connector = GoogleSheetsConnector(
            spreadsheet_id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms",
            sheet_name="Sheet1",
        )
        raw_df = connector.fetch()
    """

    def __init__(
        self,
        spreadsheet_id: str,
        sheet_name: str = "Sheet1",
        credentials_path: str = None,
        config: ConnectorConfig = None,
    ):
        """
        Initialize the Google Sheets connector.

        Parameters:
            spreadsheet_id (str): The Google Sheets spreadsheet ID
                                   (from the URL: /spreadsheets/d/<ID>/edit).
            sheet_name (str): Name of the worksheet tab to read. Defaults to "Sheet1".
            credentials_path (str | None): Path to the Google service account JSON file.
            config (ConnectorConfig | None): Shared connector configuration.
        """
        super().__init__(config)
        self.spreadsheet_id = spreadsheet_id
        self.sheet_name = sheet_name
        self.credentials_path = resolve_credential(
            "GOOGLE_SHEETS_CREDENTIALS",
            credentials_path,
            "Google service account credentials path",
        )

    def fetch(self, **kwargs) -> pd.DataFrame:
        """
        Authenticate with Google, open the spreadsheet, read all rows,
        and return a raw DataFrame.

        The first row of the sheet is used as column headers.

        Returns:
            pd.DataFrame: Raw data from the Google Sheet.

        Raises:
            ConnectorError: If authentication or sheet access fails.
            EmptyResponseError: If the sheet contains no data rows.
        """
        return self._retry(self._fetch_sheet)

    def _fetch_sheet(self) -> pd.DataFrame:
        """Internal method that performs the actual Google Sheets API call."""
        try:
            scopes = [
                "https://www.googleapis.com/auth/spreadsheets.readonly",
                "https://www.googleapis.com/auth/drive.readonly",
            ]
            credentials = Credentials.from_service_account_file(
                self.credentials_path, scopes=scopes
            )
            client = gspread.authorize(credentials)

            spreadsheet = client.open_by_key(self.spreadsheet_id)
            worksheet = spreadsheet.worksheet(self.sheet_name)
            records = worksheet.get_all_records()

        except FileNotFoundError:
            raise ConnectorError(
                f"Service account file not found at: {self.credentials_path}"
            )
        except Exception as e:
            raise ConnectorError(
                f"Failed to fetch Google Sheet '{self.spreadsheet_id}': {e}"
            ) from e

        if not records:
            raise EmptyResponseError(
                f"Google Sheet '{self.spreadsheet_id}' sheet '{self.sheet_name}' "
                f"contains no data rows."
            )

        df = pd.DataFrame(records)
        return df
