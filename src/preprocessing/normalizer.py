import pandas as pd
import numpy as np
from src.preprocessing.exceptions import InvalidSchemaError, EmptyFileError

# Column mapping aliases
ID_ALIASES = ["ticket_id", "id", "ticket", "record_id", "uuid", "index"]
DATE_ALIASES = ["created_at", "date", "timestamp", "time", "created_date", "datetime"]
TOPIC_ALIASES = ["topic", "title", "subject", "category", "issue", "header"]
MESSAGE_ALIASES = ["message", "description", "text", "body", "comment", "content", "feedback", "detail"]


def normalize_data(raw_data: pd.DataFrame | list[dict]) -> pd.DataFrame:
    """
    Normalize raw parsed data (DataFrame or list of dicts) into a standard schema:
    [ticket_id, created_at, topic, message].

    Parameters:
        raw_data (pd.DataFrame | list[dict]): Parsed raw evidence data.

    Returns:
        pd.DataFrame: Normalized DataFrame with standard columns.

    Raises:
        EmptyFileError: If raw_data is empty or contains no records.
        InvalidSchemaError: If no message column or valid text evidence exists.
    """
    if raw_data is None:
        raise EmptyFileError("Input raw data is None.")

    if isinstance(raw_data, list):
        if not raw_data:
            raise EmptyFileError("Input dictionary list is empty.")
        df = pd.DataFrame(raw_data)
    elif isinstance(raw_data, pd.DataFrame):
        df = raw_data.copy()
    else:
        raise InvalidSchemaError(f"Unsupported data structure type: {type(raw_data)}")

    if df.empty:
        raise EmptyFileError("Parsed DataFrame contains no rows.")

    # Match column aliases to canonical column names
    col_map = {}
    lower_cols = {str(c).strip().lower(): c for c in df.columns}

    # Find ID column
    id_col = _find_matching_column(lower_cols, ID_ALIASES)
    if id_col:
        col_map[id_col] = "ticket_id"

    # Find Date column
    date_col = _find_matching_column(lower_cols, DATE_ALIASES)
    if date_col:
        col_map[date_col] = "created_at"

    # Find Topic column
    topic_col = _find_matching_column(lower_cols, TOPIC_ALIASES)
    if topic_col:
        col_map[topic_col] = "topic"

    # Find Message column
    message_col = _find_matching_column(lower_cols, MESSAGE_ALIASES)
    if message_col:
        col_map[message_col] = "message"

    # Rename mapped columns
    df = df.rename(columns=col_map)

    # Ensure required 'message' column exists
    if "message" not in df.columns:
        # Fallback: if there is any string/object column, use it as message
        string_cols = df.select_dtypes(include=["object", "string"]).columns
        if not string_cols.empty:
            df["message"] = df[string_cols[0]]
        else:
            raise InvalidSchemaError("Could not find a valid text 'message' column in input data.")

    # Clean message text and drop empty message rows
    df["message"] = df["message"].astype(str).str.strip()
    df = df[df["message"].str.len() > 0].copy()

    if df.empty:
        raise EmptyFileError("No non-empty text evidence messages found after normalization.")

    # Populate missing 'topic' column
    if "topic" not in df.columns or df["topic"].isna().all():
        # Derive topic from first 4 words of message
        df["topic"] = df["message"].apply(lambda m: " ".join(m.split()[:4]))
    else:
        df["topic"] = df["topic"].fillna("General Feedback").astype(str).str.strip()

    # Populate missing 'ticket_id' column
    if "ticket_id" not in df.columns or df["ticket_id"].isna().all():
        df["ticket_id"] = np.arange(1, len(df) + 1)
    else:
        df["ticket_id"] = df["ticket_id"].fillna(pd.Series(np.arange(1, len(df) + 1), index=df.index))

    # Populate missing 'created_at' column
    if "created_at" not in df.columns:
        df["created_at"] = None

    # Final canonical column order
    result = df[["ticket_id", "created_at", "topic", "message"]].copy()
    result = result.reset_index(drop=True)
    return result


def _find_matching_column(lower_cols: dict[str, str], aliases: list[str]) -> str | None:
    """Find original column name that matches one of the alias candidate strings."""
    for alias in aliases:
        if alias in lower_cols:
            return lower_cols[alias]
    return None
