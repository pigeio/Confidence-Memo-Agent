import pytest
import pandas as pd
from src.evaluation.adapters import (
    GooglePlayAdapter,
    GitHubIssuesAdapter,
    CustomerSupportAdapter,
    AmazonReviewsAdapter,
    DatasetRegistry,
    STANDARD_COLUMNS,
)


def test_google_play_adapter():
    adapter = GooglePlayAdapter()
    raw_df = pd.DataFrame(
        [
            {
                "reviewId": "gp_1",
                "userName": "Alice",
                "content": "Please add dark mode.",
                "score": 1,
                "at": "2026-01-01",
            },
            {
                "reviewId": "gp_2",
                "userName": "Bob",
                "content": "Awesome app!",
                "score": 5,
                "at": "2026-01-02",
            },
        ]
    )

    df = adapter.adapt(raw_df)
    assert list(df.columns) == STANDARD_COLUMNS
    assert len(df) == 2
    assert df.iloc[0]["ticket_id"] == "gp_1"
    assert "App Review" in df.iloc[0]["topic"]
    assert df.iloc[0]["message"] == "Please add dark mode."


def test_github_issues_adapter():
    adapter = GitHubIssuesAdapter()
    raw_df = pd.DataFrame(
        [
            {
                "number": 101,
                "title": "Bug: Crash on startup",
                "body": "App crashes immediately with SIGSEGV.",
                "created_at": "2026-01-03",
            }
        ]
    )

    df = adapter.adapt(raw_df)
    assert list(df.columns) == STANDARD_COLUMNS
    assert len(df) == 1
    assert df.iloc[0]["ticket_id"] == "GH-101"
    assert df.iloc[0]["topic"] == "Bug: Crash on startup"
    assert df.iloc[0]["message"] == "App crashes immediately with SIGSEGV."


def test_customer_support_adapter():
    adapter = CustomerSupportAdapter()
    raw_df = pd.DataFrame(
        [
            {
                "ticket_id": "CS_99",
                "created_at": "2026-01-04",
                "text": "Cannot login with SSO.",
                "category": "Auth",
            }
        ]
    )

    df = adapter.adapt(raw_df)
    assert list(df.columns) == STANDARD_COLUMNS
    assert len(df) == 1
    assert df.iloc[0]["ticket_id"] == "CS_99"
    assert df.iloc[0]["topic"] == "Auth"
    assert df.iloc[0]["message"] == "Cannot login with SSO."


def test_amazon_reviews_adapter():
    adapter = AmazonReviewsAdapter()
    raw_df = pd.DataFrame(
        [
            {
                "reviewerID": "U123",
                "asin": "B001",
                "summary": "Battery issues",
                "reviewText": "Battery died after 1 week.",
                "reviewTime": "2026-01-05",
            }
        ]
    )

    df = adapter.adapt(raw_df)
    assert list(df.columns) == STANDARD_COLUMNS
    assert len(df) == 1
    assert df.iloc[0]["ticket_id"] == "U123_B001"
    assert df.iloc[0]["topic"] == "Battery issues"
    assert df.iloc[0]["message"] == "Battery died after 1 week."


def test_dataset_registry_discovery_and_loading():
    datasets = DatasetRegistry.list_datasets()
    assert len(datasets) >= 4
    names = [d["name"] for d in datasets]
    assert "google_play_reviews" in names
    assert "github_issues" in names
    assert "customer_support_tickets" in names
    assert "amazon_product_reviews" in names

    df, meta = DatasetRegistry.load_dataset("google_play_reviews")
    assert list(df.columns) == STANDARD_COLUMNS
    assert len(df) > 0
    assert meta["adapter"] == "GooglePlayAdapter"


def test_dataset_registry_errors():
    with pytest.raises(KeyError, match="not found in registry"):
        DatasetRegistry.get_adapter("NonExistentAdapter")

    with pytest.raises(KeyError, match="not found in"):
        DatasetRegistry.load_dataset("non_existent_dataset")
