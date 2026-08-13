import os
import json
import logging
from abc import ABC, abstractmethod
import pandas as pd

logger = logging.getLogger(__name__)

STANDARD_COLUMNS = ["ticket_id", "created_at", "topic", "message"]


class BaseDatasetAdapter(ABC):
    """
    Abstract Base Class for evaluation dataset adapters.
    Transforms arbitrary public/customer dataset schemas into the canonical project schema:
    [ticket_id, created_at, topic, message].
    """

    @abstractmethod
    def adapt(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform a raw DataFrame into the standard schema.

        Parameters:
            raw_df (pd.DataFrame): Raw ingested DataFrame.

        Returns:
            pd.DataFrame: Standardized DataFrame with columns [ticket_id, created_at, topic, message].
        """
        pass

    def validate_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and normalize DataFrame to conform to standard schema contracts:
        [ticket_id, created_at, topic, message].
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Adapted output must be a pandas DataFrame")

        clean_df = df.copy()

        # Add missing standard columns with safe defaults
        if "ticket_id" not in clean_df.columns:
            clean_df["ticket_id"] = pd.Series(range(len(clean_df))).astype(str)
        if "created_at" not in clean_df.columns:
            clean_df["created_at"] = ""
        if "topic" not in clean_df.columns:
            clean_df["topic"] = "General Feedback"
        if "message" not in clean_df.columns:
            clean_df["message"] = ""

        clean_df = clean_df[STANDARD_COLUMNS].copy()

        # Handle NaNs and non-string types
        clean_df["ticket_id"] = clean_df["ticket_id"].fillna("").astype(str)
        clean_df["created_at"] = clean_df["created_at"].fillna("").astype(str)
        clean_df["topic"] = clean_df["topic"].fillna("General Feedback").astype(str).str.strip()
        clean_df["message"] = clean_df["message"].fillna("").astype(str).str.strip()

        # Filter out rows with completely empty messages
        clean_df = clean_df[clean_df["message"].str.len() > 0].reset_index(drop=True)

        return clean_df


class GooglePlayAdapter(BaseDatasetAdapter):
    """
    Adapter for Google Play Store & Mobile App Store review datasets.
    Raw fields typically: [reviewId, userName, content, score, thumbsUpCount, at, replyContent]
    """

    def adapt(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        if raw_df.empty:
            return pd.DataFrame(columns=STANDARD_COLUMNS)

        ticket_id = raw_df.get("reviewId", pd.Series(range(len(raw_df)))).astype(str)
        created_at = raw_df.get("at", pd.Series([""] * len(raw_df))).astype(str)
        message = raw_df.get("content", raw_df.get("reviewText", pd.Series([""] * len(raw_df)))).astype(str)

        # Derive topic from score or rating if present
        if "score" in raw_df.columns:
            topic = raw_df["score"].apply(lambda s: f"App Review (Rating {s}/5)")
        else:
            topic = pd.Series(["Mobile App Review"] * len(raw_df))

        adapted_df = pd.DataFrame(
            {
                "ticket_id": ticket_id,
                "created_at": created_at,
                "topic": topic,
                "message": message,
            }
        )
        return self.validate_schema(adapted_df)


class GitHubIssuesAdapter(BaseDatasetAdapter):
    """
    Adapter for GitHub Issues & Developer issue tracker datasets.
    Raw fields typically: [number / id, title, body, created_at, labels, state]
    """

    def adapt(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        if raw_df.empty:
            return pd.DataFrame(columns=STANDARD_COLUMNS)

        ticket_id = (
            raw_df.get("number", raw_df.get("id", pd.Series(range(len(raw_df)))))
            .astype(str)
            .apply(lambda n: f"GH-{n}")
        )
        created_at = raw_df.get("created_at", pd.Series([""] * len(raw_df))).astype(str)
        topic = raw_df.get("title", pd.Series(["Issue Report"] * len(raw_df))).astype(str)
        message = raw_df.get("body", raw_df.get("message", pd.Series([""] * len(raw_df)))).astype(str)

        adapted_df = pd.DataFrame(
            {
                "ticket_id": ticket_id,
                "created_at": created_at,
                "topic": topic,
                "message": message,
            }
        )
        return self.validate_schema(adapted_df)


class CustomerSupportAdapter(BaseDatasetAdapter):
    """
    Adapter for Customer Support ticket datasets (e.g. Zendesk, Twitter Support, Intercom export).
    Raw fields typically: [ticket_id / tweet_id, created_at, text / message, category]
    """

    def adapt(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        if raw_df.empty:
            return pd.DataFrame(columns=STANDARD_COLUMNS)

        ticket_id = raw_df.get("ticket_id", raw_df.get("tweet_id", pd.Series(range(len(raw_df))))).astype(str)
        created_at = raw_df.get("created_at", pd.Series([""] * len(raw_df))).astype(str)
        topic = raw_df.get("category", raw_df.get("topic", pd.Series(["Customer Support"] * len(raw_df)))).astype(str)
        message = raw_df.get("text", raw_df.get("message", pd.Series([""] * len(raw_df)))).astype(str)

        adapted_df = pd.DataFrame(
            {
                "ticket_id": ticket_id,
                "created_at": created_at,
                "topic": topic,
                "message": message,
            }
        )
        return self.validate_schema(adapted_df)


class AmazonReviewsAdapter(BaseDatasetAdapter):
    """
    Adapter for Amazon product reviews & E-commerce feedback datasets.
    Raw fields typically: [reviewerID, asin, summary, reviewText, overall, reviewTime]
    """

    def adapt(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        if raw_df.empty:
            return pd.DataFrame(columns=STANDARD_COLUMNS)

        reviewer_id = raw_df.get("reviewerID", pd.Series(["REV"] * len(raw_df))).astype(str)
        asin = raw_df.get("asin", pd.Series(["PROD"] * len(raw_df))).astype(str)
        ticket_id = reviewer_id + "_" + asin

        created_at = raw_df.get("reviewTime", raw_df.get("created_at", pd.Series([""] * len(raw_df)))).astype(str)
        topic = raw_df.get("summary", pd.Series(["Product Feedback"] * len(raw_df))).astype(str)
        message = raw_df.get("reviewText", raw_df.get("message", pd.Series([""] * len(raw_df)))).astype(str)

        adapted_df = pd.DataFrame(
            {
                "ticket_id": ticket_id,
                "created_at": created_at,
                "topic": topic,
                "message": message,
            }
        )
        return self.validate_schema(adapted_df)


class DatasetRegistry:
    """
    Config-driven registry for discovering, loading, and adapting evaluation datasets.
    """

    _ADAPTERS: dict[str, type[BaseDatasetAdapter]] = {
        "GooglePlayAdapter": GooglePlayAdapter,
        "GitHubIssuesAdapter": GitHubIssuesAdapter,
        "CustomerSupportAdapter": CustomerSupportAdapter,
        "AmazonReviewsAdapter": AmazonReviewsAdapter,
    }

    @classmethod
    def register_adapter(cls, name: str, adapter_cls: type[BaseDatasetAdapter]) -> None:
        cls._ADAPTERS[name] = adapter_cls

    @classmethod
    def get_adapter(cls, adapter_name: str) -> BaseDatasetAdapter:
        if adapter_name not in cls._ADAPTERS:
            raise KeyError(
                f"Adapter '{adapter_name}' not found in registry. Registered: {list(cls._ADAPTERS.keys())}"
            )
        return cls._ADAPTERS[adapter_name]()

    @classmethod
    def list_datasets(cls, config_path: str = "data/evaluation/datasets.json") -> list[dict]:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Datasets config file not found at: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("datasets", [])

    @classmethod
    def load_dataset(
        cls, dataset_name: str, config_path: str = "data/evaluation/datasets.json"
    ) -> tuple[pd.DataFrame, dict]:
        """
        Load a dataset by registered name, parse format, and adapt to standard schema.

        Returns:
            tuple: (standard_df, dataset_metadata)
        """
        datasets = cls.list_datasets(config_path)
        matching = [d for d in datasets if d.get("name") == dataset_name]
        if not matching:
            available = [d.get("name") for d in datasets]
            raise KeyError(f"Dataset '{dataset_name}' not found in {config_path}. Available: {available}")

        meta = matching[0]
        file_path = meta.get("path")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file does not exist at: {file_path}")

        fmt = meta.get("format", "csv").lower()
        if fmt == "csv":
            raw_df = pd.read_csv(file_path)
        elif fmt == "json":
            raw_df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported format '{fmt}' for dataset '{dataset_name}'")

        adapter_name = meta.get("adapter", "")
        adapter = cls.get_adapter(adapter_name)
        adapted_df = adapter.adapt(raw_df)

        return adapted_df, meta
