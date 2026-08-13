import os
import sys
import glob
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Ensure project root directory is in the Python search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memo_service import MemoService
from src.retrieval import retrieve_tickets
from src.evidence_validation import validate_retrieved_evidence
from src.config import EVIDENCE_SIMILARITY_THRESHOLD
from src.preprocessing import load_data, detect_file_type


def detect_sample_dataset(sample_dir: str) -> str:
    """
    Detect sample dataset file inside the data/sample/ directory.

    Parameters:
        sample_dir (str): Path to the sample data directory.

    Returns:
        str: Absolute path to the detected dataset file.
    """
    for ext in ["*.csv", "*.xlsx", "*.json", "*.txt", "*.pdf"]:
        found = glob.glob(os.path.join(sample_dir, ext))
        if found:
            return found[0]
    raise FileNotFoundError(f"No valid sample dataset found in directory: {sample_dir}")


def main():
    print("==========================================================")
    print("  Confidence Memo Agent — Universal Ingestion Pipeline    ")
    print("==========================================================\n")

    # Load environment variables (.env file containing GEMINI_API_KEY)
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        print("[WARNING] GEMINI_API_KEY is not set in environment or .env file.")
        print("API requests to Gemini may fail if credentials are not configured.\n")

    # Set UTF-8 encoding for stdout on Windows if supported
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # Stage 1: Load sample support ticket dataset via Universal Ingestion Preprocessing Layer
    print("Loading evidence via Universal Preprocessing Layer...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sample_dir = os.path.join(base_dir, "data", "sample")

    try:
        file_path = detect_sample_dataset(sample_dir)
        format_type = detect_file_type(file_path)
        # Load and normalize evidence into a standard pandas DataFrame
        df = load_data(file_path)
        print(f"[OK] Ingested and normalized format '{format_type.upper()}' ({len(df)} evidence records from '{os.path.basename(file_path)}')\n")
    except Exception as e:
        print(f"[ERROR] Failed to load evidence dataset: {e}")
        return

    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return

    # Stage 2: Initialize orchestration MemoService
    # MemoService delegates to retrieval, prompt building, and Gemini client
    memo_service = MemoService()

    # Stage 3: Define realistic feature proposals and search queries/keywords
    proposals_to_test = [
        {
            "proposal": "Should we add Dark Mode?",
            "keywords": ["dark mode", "eye strain", "night mode", "dark theme"],
        },
        {
            "proposal": "Should we support CSV Export?",
            "keywords": ["export", "csv", "download data"],
        },
        {
            "proposal": "Should we build Offline Mode?",
            "keywords": ["offline mode", "no internet connection", "airplane mode"],
        },
    ]

    # Stage 4: Run proposals sequentially through the pipeline
    for item in proposals_to_test:
        proposal = item["proposal"]
        keywords = item["keywords"]

        print("----------------------------------------")
        print(f"Running query: {proposal}")
        print("----------------------------------------")

        try:
            # 1. Retrieve candidate tickets and similarity scores
            matching_tickets, similarity_scores = retrieve_tickets(df, keywords)

            # 2. Validate evidence using similarity threshold
            validated_tickets = validate_retrieved_evidence(
                matching_tickets, similarity_scores, threshold=EVIDENCE_SIMILARITY_THRESHOLD
            )

            retrieved_count = len(matching_tickets)
            validated_count = len(validated_tickets)
            rejected_count = retrieved_count - validated_count
            avg_similarity = float(np.mean(similarity_scores)) if len(similarity_scores) > 0 else 0.0

            # Sprint 3.3 — Retrieval & Validation Transparency Display
            print("Retrieval Summary\n")
            print(f"Retrieved Tickets    : {retrieved_count}")
            print(f"Validated Tickets    : {validated_count}")
            print(f"Rejected Tickets     : {rejected_count}")
            print(f"Similarity Threshold : {EVIDENCE_SIMILARITY_THRESHOLD:.2f}")
            print(f"Average Similarity   : {avg_similarity:.2f}\n")

            # Print Ticket Details Table
            if retrieved_count > 0:
                print(f"{'Ticket':<10}{'Similarity':<14}{'Status'}")
                for idx, (_, row) in enumerate(matching_tickets.iterrows()):
                    t_id = row.get("ticket_id", idx + 1)
                    sim = similarity_scores[idx]
                    status = "Accepted" if sim >= EVIDENCE_SIMILARITY_THRESHOLD else "Rejected"
                    print(f"{t_id:<10}{sim:<14.2f}{status}")
                print()
            else:
                print("No candidate tickets retrieved.\n")

            print("Building memo...")
            # Call the public API generate_evidence_memo from MemoService with pre-validated tickets
            memo = memo_service.generate_evidence_memo(
                df=df,
                keywords=keywords,
                proposal=proposal,
                validated_tickets=validated_tickets,
            )
            print("[OK] Done\n")

            print("=== GENERATED EVIDENCE MEMO ===")
            print(memo)
            print("=================================\n")

        except Exception as e:
            # Catch exceptions gracefully so remaining queries continue executing
            print(f"[ERROR] Pipeline execution failed for '{proposal}': {e}\n")

        # Sleep briefly between queries to avoid hitting Gemini free-tier rate limits
        import time
        time.sleep(3)

    print("========================================")
    print("Pipeline Demo Execution Complete")
    print("========================================")


if __name__ == "__main__":
    main()
