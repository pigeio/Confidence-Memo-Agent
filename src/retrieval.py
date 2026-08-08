import pandas as pd
import re


def retrieve_tickets(df: pd.DataFrame, keywords: list[str]) -> pd.DataFrame:
    """
    Retrieve support tickets that match any of the given keywords
    in the topic or message columns.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the support tickets.
                           Must have 'topic' and 'message' columns.
        keywords (list[str]): A list of keywords (strings) to search for.

    Returns:
        pd.DataFrame: A new DataFrame containing only the matching tickets.

    Raises:
        TypeError: If df is not a pandas DataFrame or keywords is not a list.
        ValueError: If required columns are missing, the keywords list is empty,
                    or keywords contains non-string/empty values.
    """
    # Validate DataFrame type
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    # Validate required columns
    required_cols = {"topic", "message"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"DataFrame is missing required columns: {missing_cols}")

    # Validate keywords list type
    if not isinstance(keywords, list):
        raise TypeError("keywords must be a list of strings")

    # Validate keywords contents
    if not keywords:
        raise ValueError("keywords list cannot be empty")

    for kw in keywords:
        if not isinstance(kw, str):
            raise TypeError("All keywords must be strings")
        if not kw.strip():
            raise ValueError("Keywords cannot be empty or whitespace-only strings")

    # Build regex pattern for keyword matching
    pattern = "|".join(re.escape(kw.strip()) for kw in keywords)

    # Perform case-insensitive search in topic and message columns
    mask = (
        df["topic"].str.contains(pattern, case=False, na=False)
        |
        df["message"].str.contains(pattern, case=False, na=False)
    )

    return df.loc[mask].copy()