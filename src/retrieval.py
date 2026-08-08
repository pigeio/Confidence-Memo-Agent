import pandas as pd
from src.semantic_search import search_tickets


def retrieve_tickets(df: pd.DataFrame, keywords: list[str]) -> pd.DataFrame:
    """
    Retrieve support tickets that semantically match any of the given keywords
    or search query in the topic or message columns.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the support tickets.
                           Must have 'topic' and 'message' columns.
        keywords (list[str]): A list of keywords or query strings to search for.

    Returns:
        pd.DataFrame: A new DataFrame containing only the matching tickets.

    Raises:
        TypeError: If df is not a pandas DataFrame or keywords is not a list.
        ValueError: If required columns are missing, the keywords list is empty,
                    or keywords contains non-string/empty values.
    """
    return search_tickets(df, keywords)
