import pytest
import numpy as np
import pandas as pd
from src.retrieval import retrieve_tickets


@pytest.fixture
def sample_tickets_df() -> pd.DataFrame:
    """
    Pytest fixture providing a small synthetic DataFrame of support tickets
    for testing keyword retrieval scenarios.
    """
    return pd.DataFrame([
        {
            "ticket_id": 1,
            "topic": "Dark mode support",
            "message": "The bright background hurts my eyes at night."
        },
        {
            "ticket_id": 2,
            "topic": "Export feature request",
            "message": "Can we export transaction reports to CSV?"
        },
        {
            "ticket_id": 3,
            "topic": "C++ SDK Integration",
            "message": "Need help setting up C++ libraries."
        },
        {
            "ticket_id": 4,
            "topic": "General Feedback",
            "message": "Great product! Node.js and C# documentation is helpful."
        }
    ])


# 1. Normal keyword match
def test_normal_keyword_match(sample_tickets_df):
    """Verify that a standard keyword matches relevant tickets."""
    result, scores = retrieve_tickets(sample_tickets_df, ["export"])
    assert len(result) == 1
    assert result.iloc[0]["ticket_id"] == 2
    assert len(scores) == 1
    assert scores[0] > 0.0


# 2. Case-insensitive matching
def test_case_insensitive_matching(sample_tickets_df):
    """Verify that keyword matching is case-insensitive (e.g. 'DARK' matches 'Dark')."""
    result_upper, scores_upper = retrieve_tickets(sample_tickets_df, ["DARK"])
    result_lower, scores_lower = retrieve_tickets(sample_tickets_df, ["dark"])

    assert len(result_upper) == 1
    assert len(result_lower) == 1
    assert result_upper.iloc[0]["ticket_id"] == result_lower.iloc[0]["ticket_id"] == 1


# 3. Matching in the topic column
def test_matching_in_topic_column(sample_tickets_df):
    """Verify matching occurs when the keyword is present only in the 'topic' column."""
    result, scores = retrieve_tickets(sample_tickets_df, ["support"])
    assert len(result) == 1
    assert result.iloc[0]["ticket_id"] == 1


# 4. Matching in the message column
def test_matching_in_message_column(sample_tickets_df):
    """Verify matching occurs when the keyword is present only in the 'message' column."""
    result, scores = retrieve_tickets(sample_tickets_df, ["background"])
    assert len(result) == 1
    assert result.iloc[0]["ticket_id"] == 1


# 5. Multiple keywords
def test_multiple_keywords_match(sample_tickets_df):
    """Verify that providing multiple keywords matches any ticket containing at least one keyword."""
    result, scores = retrieve_tickets(sample_tickets_df, ["dark", "export"])
    assert len(result) == 2
    assert set(result["ticket_id"]) == {1, 2}
    assert len(scores) == 2


# 6. No matching tickets
def test_no_matching_tickets(sample_tickets_df):
    """Verify that searching for a non-existent keyword returns an empty DataFrame."""
    result, scores = retrieve_tickets(sample_tickets_df, ["nonexistent_keyword"])
    assert isinstance(result, pd.DataFrame)
    assert result.empty
    assert list(result.columns) == list(sample_tickets_df.columns)
    assert len(scores) == 0


# 7. Empty DataFrame
def test_empty_dataframe():
    """Verify that searching on an empty DataFrame with required columns returns an empty DataFrame."""
    empty_df = pd.DataFrame(columns=["ticket_id", "topic", "message"])
    result, scores = retrieve_tickets(empty_df, ["dark"])
    assert isinstance(result, pd.DataFrame)
    assert result.empty
    assert len(scores) == 0


# 8. Empty keyword list
def test_empty_keyword_list(sample_tickets_df):
    """Verify that passing an empty list of keywords raises a ValueError."""
    with pytest.raises(ValueError, match="cannot be empty"):
        retrieve_tickets(sample_tickets_df, [])


# 9. Special-character keywords (e.g. C++, C#, Node.js)
def test_special_character_keywords(sample_tickets_df):
    """Verify that special character keywords (C++, C#, Node.js) retrieve semantically relevant technical tickets."""
    result_cpp, _ = retrieve_tickets(sample_tickets_df, ["C++"])
    assert len(result_cpp) >= 1
    assert 3 in list(result_cpp["ticket_id"])

    result_csharp, _ = retrieve_tickets(sample_tickets_df, ["C#"])
    assert len(result_csharp) >= 1
    assert 4 in list(result_csharp["ticket_id"])

    result_nodejs, _ = retrieve_tickets(sample_tickets_df, ["Node.js"])
    assert len(result_nodejs) >= 1
    assert 4 in list(result_nodejs["ticket_id"])


# 10. Missing "topic" column should raise an informative exception
def test_missing_topic_column():
    """Verify that a DataFrame missing the 'topic' column raises a ValueError."""
    invalid_df = pd.DataFrame({"ticket_id": [1], "message": ["Hello world"]})
    with pytest.raises(ValueError, match="missing required columns"):
        retrieve_tickets(invalid_df, ["hello"])


# 11. Missing "message" column should raise an informative exception
def test_missing_message_column():
    """Verify that a DataFrame missing the 'message' column raises a ValueError."""
    invalid_df = pd.DataFrame({"ticket_id": [1], "topic": ["Dark Mode"]})
    with pytest.raises(ValueError, match="missing required columns"):
        retrieve_tickets(invalid_df, ["dark"])


# 12. Invalid input types raise appropriate exceptions
def test_invalid_input_types(sample_tickets_df):
    """Verify that invalid argument types for 'df' and 'keywords' raise TypeError or ValueError."""
    # Invalid df type
    with pytest.raises(TypeError, match="df must be a pandas DataFrame"):
        retrieve_tickets("not_a_dataframe", ["dark"])

    # Single string is valid in semantic search
    result_str_query, scores = retrieve_tickets(sample_tickets_df, "dark")
    assert isinstance(result_str_query, pd.DataFrame)
    assert isinstance(scores, np.ndarray)

    # Invalid query type (int instead of string/list)
    with pytest.raises(TypeError, match="must be a string or a list of strings"):
        retrieve_tickets(sample_tickets_df, 12345)

    # Non-string element inside keywords list
    with pytest.raises(ValueError, match="contains no valid non-empty strings"):
        retrieve_tickets(sample_tickets_df, ["   "])


# 13. Similarity scores are returned alongside tickets
def test_similarity_scores_returned(sample_tickets_df):
    """Verify that retrieve_tickets returns similarity scores as a numpy array."""
    result, scores = retrieve_tickets(sample_tickets_df, ["dark mode"])
    assert isinstance(scores, np.ndarray)
    assert len(scores) == len(result)
    # Scores should be in descending order
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1]
