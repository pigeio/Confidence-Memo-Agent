import re
import string
import logging
import numpy as np
import pandas as pd
from src.config import DEDUPLICATION_SIMILARITY_THRESHOLD
from src.embedding_generator import EmbeddingGenerator

logger = logging.getLogger(__name__)


def _normalize_text(text: str) -> str:
    """Normalize text for exact duplicate detection by lowercasing, stripping punctuation, and collapsing whitespace."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.lower().strip()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


class EvidenceDeduplicator:
    """
    Deterministic Evidence Deduplication Engine.
    Detects both exact and semantic duplicate customer feedback tickets,
    selects the best representative ticket for each duplicate group,
    and reports detailed deduplication statistics.
    """

    def __init__(
        self,
        similarity_threshold: float = DEDUPLICATION_SIMILARITY_THRESHOLD,
        embedding_generator: EmbeddingGenerator | None = None,
    ):
        """
        Initialize the EvidenceDeduplicator.

        Parameters:
            similarity_threshold (float): Cosine similarity threshold (0.0 - 1.0) above which
                                          two tickets are considered semantic duplicates.
            embedding_generator (EmbeddingGenerator): Optional embedding generator instance.
        """
        if not (0.0 <= similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")

        self.similarity_threshold = float(similarity_threshold)
        self.embedding_generator = embedding_generator or EmbeddingGenerator()

    def deduplicate(
        self,
        df: pd.DataFrame,
        similarity_scores: np.ndarray | None = None,
        representative_strategy: str = "longest",
    ) -> tuple[pd.DataFrame, np.ndarray | None, dict]:
        """
        Deduplicate a DataFrame of support tickets using exact text normalization and
        dense semantic embeddings.

        Parameters:
            df (pd.DataFrame): Support tickets DataFrame.
            similarity_scores (np.ndarray): Optional 1D array of retrieval/relevance scores aligned with df rows.
            representative_strategy (str): Strategy for selecting representative ticket:
                                          'longest' (default): ticket with most comprehensive text
                                          'highest_similarity': ticket with highest retrieval score
                                          'earliest': first ticket in order of appearance / timestamp

        Returns:
            tuple: (
                deduplicated_df (pd.DataFrame): DataFrame with duplicate rows removed,
                deduplicated_scores (np.ndarray | None): Aligned similarity scores for deduplicated rows,
                stats (dict): Deduplication statistics and telemetry breakdown
            )

        Raises:
            TypeError: If df is not a DataFrame or similarity_scores is invalid.
            ValueError: If strategy is invalid or df lacks required columns.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        valid_strategies = {"longest", "highest_similarity", "earliest"}
        if representative_strategy not in valid_strategies:
            raise ValueError(
                f"representative_strategy must be one of {valid_strategies}, got '{representative_strategy}'"
            )

        if similarity_scores is not None:
            if not isinstance(similarity_scores, (np.ndarray, list)):
                raise TypeError("similarity_scores must be a numpy ndarray or list")
            similarity_scores = np.array(similarity_scores, dtype=np.float32)
            if len(similarity_scores) != len(df):
                raise ValueError(
                    f"similarity_scores length ({len(similarity_scores)}) must match DataFrame length ({len(df)})"
                )

        total_input = len(df)
        if total_input <= 1:
            stats = {
                "total_input_count": total_input,
                "unique_count": total_input,
                "duplicate_count": 0,
                "duplicate_rate": 0.0,
                "exact_duplicates_count": 0,
                "semantic_duplicates_count": 0,
                "duplicate_groups": [],
            }
            return df.copy(), (similarity_scores.copy() if similarity_scores is not None else None), stats

        # Required columns validation
        required_cols = {"topic", "message"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"df is missing required columns: {missing_cols}")

        # Step 1: Exact duplicate detection via normalization
        exact_keys = []
        for _, row in df.iterrows():
            norm_topic = _normalize_text(row.get("topic", ""))
            norm_msg = _normalize_text(row.get("message", ""))
            exact_keys.append(f"{norm_topic}||{norm_msg}")

        # Disjoint set / Union-Find structure to group duplicate indices
        parent = list(range(total_input))
        duplicate_type_map = {}  # index pair -> 'exact' or 'semantic'

        def find(i: int) -> int:
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]

        def union(i: int, j: int, dup_type: str) -> None:
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j
                duplicate_type_map[(min(i, j), max(i, j))] = dup_type

        # Group exact duplicates
        seen_keys: dict[str, int] = {}
        exact_duplicate_pairs = 0
        for i, key in enumerate(exact_keys):
            if key in seen_keys:
                union(i, seen_keys[key], "exact")
                exact_duplicate_pairs += 1
            else:
                seen_keys[key] = i

        # Step 2: Semantic duplicate detection using dense vector cosine similarity
        ticket_texts = [
            f"Topic: {row.get('topic', '')} | Message: {row.get('message', '')}"
            for _, row in df.iterrows()
        ]
        embeddings = self.embedding_generator.encode_texts(ticket_texts)

        # Pairwise cosine similarity matrix (embeddings are pre-normalized to L2 unit norm)
        sim_matrix = np.dot(embeddings, embeddings.T)
        sim_matrix = np.clip(sim_matrix, 0.0, 1.0)

        semantic_duplicate_pairs = 0
        for i in range(total_input):
            for j in range(i + 1, total_input):
                # If not already unified by exact match and similarity exceeds threshold
                if find(i) != find(j) and sim_matrix[i, j] >= self.similarity_threshold:
                    union(i, j, "semantic")
                    semantic_duplicate_pairs += 1

        # Step 3: Group tickets by connected component root
        groups: dict[int, list[int]] = {}
        for i in range(total_input):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)

        # Step 4: Representative Selection
        selected_indices = []
        group_metadata_list = []

        exact_dups_count = 0
        semantic_dups_count = 0

        for root, member_indices in groups.items():
            if len(member_indices) == 1:
                selected_indices.append(member_indices[0])
                continue

            # Multi-member duplicate group: select representative
            rep_idx = self._select_representative(
                df, member_indices, similarity_scores, representative_strategy
            )
            selected_indices.append(rep_idx)

            # Determine duplicate types within this group
            other_indices = [idx for idx in member_indices if idx != rep_idx]
            has_exact = any(
                exact_keys[idx] == exact_keys[rep_idx] for idx in other_indices
            )
            has_semantic = any(
                exact_keys[idx] != exact_keys[rep_idx] for idx in other_indices
            )

            exact_in_group = sum(
                1 for idx in other_indices if exact_keys[idx] == exact_keys[rep_idx]
            )
            semantic_in_group = len(other_indices) - exact_in_group

            exact_dups_count += exact_in_group
            semantic_dups_count += semantic_in_group

            rep_ticket_id = (
                df.iloc[rep_idx].get("ticket_id", rep_idx)
                if "ticket_id" in df.columns
                else rep_idx
            )
            dup_ticket_ids = [
                df.iloc[idx].get("ticket_id", idx) if "ticket_id" in df.columns else idx
                for idx in other_indices
            ]

            group_metadata_list.append(
                {
                    "representative_index": rep_idx,
                    "representative_ticket_id": rep_ticket_id,
                    "duplicate_indices": other_indices,
                    "duplicate_ticket_ids": dup_ticket_ids,
                    "group_size": len(member_indices),
                    "has_exact": has_exact,
                    "has_semantic": has_semantic,
                }
            )

        # Sort selected indices to preserve original order
        selected_indices.sort()

        # Step 5: Build deduplicated DataFrame and aligned scores
        dedup_df = df.iloc[selected_indices].copy().reset_index(drop=True)
        dedup_scores = (
            similarity_scores[selected_indices] if similarity_scores is not None else None
        )

        duplicate_count = total_input - len(selected_indices)
        duplicate_rate = round(duplicate_count / total_input, 4) if total_input > 0 else 0.0

        stats = {
            "total_input_count": total_input,
            "unique_count": len(dedup_df),
            "duplicate_count": duplicate_count,
            "duplicate_rate": duplicate_rate,
            "exact_duplicates_count": exact_dups_count,
            "semantic_duplicates_count": semantic_dups_count,
            "duplicate_groups": group_metadata_list,
        }

        return dedup_df, dedup_scores, stats

    def _select_representative(
        self,
        df: pd.DataFrame,
        indices: list[int],
        similarity_scores: np.ndarray | None,
        strategy: str,
    ) -> int:
        """Select representative ticket index among candidate indices based on the chosen strategy."""
        if strategy == "highest_similarity" and similarity_scores is not None:
            # Pick the ticket with highest similarity score
            best_idx = indices[0]
            best_score = similarity_scores[best_idx]
            for idx in indices[1:]:
                if similarity_scores[idx] > best_score:
                    best_score = similarity_scores[idx]
                    best_idx = idx
            return best_idx

        if strategy == "earliest":
            # If created_at is present, try date sorting; else index order
            if "created_at" in df.columns:
                try:
                    sub_df = df.iloc[indices]
                    earliest_pos = pd.to_datetime(
                        sub_df["created_at"], errors="coerce"
                    ).argmin()
                    return indices[earliest_pos]
                except Exception:
                    pass
            return min(indices)

        # Default strategy: 'longest' (most informative message text)
        best_idx = indices[0]
        max_len = len(str(df.iloc[best_idx].get("message", "")))
        for idx in indices[1:]:
            msg_len = len(str(df.iloc[idx].get("message", "")))
            if msg_len > max_len:
                max_len = msg_len
                best_idx = idx
        return best_idx
