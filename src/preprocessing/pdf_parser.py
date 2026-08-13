import pypdf
import pandas as pd
from pathlib import Path
from src.preprocessing.exceptions import ParserError, EmptyFileError


class PDFParser:
    """
    Dedicated parser for PDF (.pdf) evidence files.
    Extracts text using pypdf.PdfReader page-by-page and converts non-empty paragraphs
    or pages into evidence records with auto-generated IDs.
    """

    def parse(self, file_path: str | Path) -> pd.DataFrame:
        """
        Parse PDF evidence data into a pandas DataFrame.

        Parameters:
            file_path (str | Path): Path to the PDF file.

        Returns:
            pd.DataFrame: Raw parsed DataFrame with ticket_id, topic, and message.

        Raises:
            EmptyFileError: If the PDF contains no extractable text pages.
            ParserError: If pypdf fails to read or parse the PDF document.
        """
        try:
            reader = pypdf.PdfReader(str(file_path))
            if not reader.pages:
                raise EmptyFileError(f"PDF file '{file_path}' has 0 pages.")

            records = []
            rec_id = 1

            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if not text:
                    continue

                lines = [line.strip() for line in text.splitlines() if line.strip()]
                for line in lines:
                    if len(line) < 5:
                        continue
                    words = line.split()
                    topic = " ".join(words[:4]) if words else f"PDF Page {page_num}"
                    records.append({
                        "ticket_id": rec_id,
                        "topic": topic,
                        "message": line,
                    })
                    rec_id += 1

            if not records:
                raise EmptyFileError(f"PDF file '{file_path}' contains no extractable text content.")

            return pd.DataFrame(records)

        except EmptyFileError:
            raise
        except Exception as e:
            raise ParserError(f"Failed to parse PDF file '{file_path}': {e}") from e
