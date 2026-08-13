import json
import pandas as pd
from pathlib import Path
from src.preprocessing.exceptions import ParserError, EmptyFileError


class JSONParser:
    """
    Dedicated parser for JSON (.json) evidence files.
    Supports both list of dictionaries and nested object key wrapper structures.
    """

    def parse(self, file_path: str | Path) -> pd.DataFrame:
        """
        Parse JSON evidence data into a pandas DataFrame.

        Parameters:
            file_path (str | Path): Path to the JSON file.

        Returns:
            pd.DataFrame: Raw parsed DataFrame.

        Raises:
            EmptyFileError: If the JSON file contains no data records.
            ParserError: If JSON syntax or parsing fails.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            records = self._extract_records(data)
            if not records:
                raise EmptyFileError(f"JSON file '{file_path}' contains no valid data records.")

            df = pd.DataFrame(records)
            if df.empty:
                raise EmptyFileError(f"JSON file '{file_path}' produced an empty DataFrame.")

            return df

        except EmptyFileError:
            raise
        except json.JSONDecodeError as e:
            raise ParserError(f"Malformed JSON syntax in file '{file_path}': {e}") from e
        except Exception as e:
            raise ParserError(f"Failed to parse JSON file '{file_path}': {e}") from e

    def _extract_records(self, data: object) -> list[dict]:
        """Extract a list of records from either a top-level list or nested object wrapper."""
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]

        if isinstance(data, dict):
            # Check for common wrapper keys like "tickets", "data", "records", "items", "evidence"
            for key in ("tickets", "data", "records", "items", "evidence", "support_tickets"):
                if key in data and isinstance(data[key], list):
                    return [item for item in data[key] if isinstance(item, dict)]

            # If dict values are dictionaries, flatten values
            sub_dicts = [v for v in data.values() if isinstance(v, dict)]
            if sub_dicts:
                return sub_dicts

            # Single record dictionary
            return [data]

        return []
