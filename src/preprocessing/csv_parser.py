import pandas as pd
from pathlib import Path
from src.preprocessing.exceptions import ParserError, EmptyFileError


class CSVParser:
    """
    Dedicated parser for CSV (.csv) evidence files.
    """

    def parse(self, file_path: str | Path) -> pd.DataFrame:
        """
        Parse CSV evidence data into a pandas DataFrame.

        Parameters:
            file_path (str | Path): Path to the CSV file.

        Returns:
            pd.DataFrame: Raw parsed DataFrame.

        Raises:
            EmptyFileError: If the CSV file contains no data rows.
            ParserError: If pandas fails to read or parse the CSV file.
        """
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                raise EmptyFileError(f"CSV file '{file_path}' contains no data rows.")
            return df
        except EmptyFileError:
            raise
        except Exception as e:
            raise ParserError(f"Failed to parse CSV file '{file_path}': {e}") from e
