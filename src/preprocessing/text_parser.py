import pandas as pd
from pathlib import Path
from src.preprocessing.exceptions import ParserError, EmptyFileError


class TextParser:
    """
    Dedicated parser for plain text (.txt) evidence files.
    Treats each non-empty line (or paragraph) as an individual evidence record.
    Generates sequential IDs automatically.
    """

    def parse(self, file_path: str | Path) -> pd.DataFrame:
        """
        Parse text evidence data into a pandas DataFrame.

        Parameters:
            file_path (str | Path): Path to the TXT file.

        Returns:
            pd.DataFrame: Raw parsed DataFrame with generated ticket_id, topic, and message.

        Raises:
            EmptyFileError: If the text file has no non-empty lines.
            ParserError: If reading the file fails.
        """
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            records = []
            rec_id = 1
            for line in lines:
                cleaned = line.strip()
                if cleaned and not cleaned.startswith("#"):
                    # Use first few words as topic, full line as message
                    words = cleaned.split()
                    topic = " ".join(words[:4]) if words else "Customer Note"
                    records.append({
                        "ticket_id": rec_id,
                        "topic": topic,
                        "message": cleaned,
                    })
                    rec_id += 1

            if not records:
                raise EmptyFileError(f"Text file '{file_path}' contains no valid text lines.")

            return pd.DataFrame(records)

        except EmptyFileError:
            raise
        except Exception as e:
            raise ParserError(f"Failed to parse text file '{file_path}': {e}") from e
