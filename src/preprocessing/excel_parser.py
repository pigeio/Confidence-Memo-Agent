import pandas as pd
from pathlib import Path
from src.preprocessing.exceptions import ParserError, EmptyFileError


class ExcelParser:
    """
    Dedicated parser for Excel (.xlsx, .xls) evidence files.
    Reads the first worksheet by default using openpyxl.
    """

    def parse(self, file_path: str | Path) -> pd.DataFrame:
        """
        Parse Excel evidence data into a pandas DataFrame.

        Parameters:
            file_path (str | Path): Path to the Excel file.

        Returns:
            pd.DataFrame: Raw parsed DataFrame.

        Raises:
            EmptyFileError: If the Excel sheet contains no data rows.
            ParserError: If openpyxl or pandas fails to read the Excel file.
        """
        try:
            df = pd.read_excel(file_path, sheet_name=0, engine="openpyxl")
            if df.empty:
                raise EmptyFileError(f"Excel file '{file_path}' contains no data rows.")
            return df
        except EmptyFileError:
            raise
        except Exception as e:
            raise ParserError(f"Failed to parse Excel file '{file_path}': {e}") from e
