import os
import sys
import glob
import pandas as pd
from dotenv import load_dotenv

# Ensure project root directory is in the Python search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memo_service import MemoService


def detect_sample_dataset(sample_dir: str) -> str:
    """
    Detect the sample dataset CSV file inside the data/sample/ directory.

    Parameters:
        sample_dir (str): Path to the sample data directory.

    Returns:
        str: Absolute path to the detected CSV file.

    Raises:
        FileNotFoundError: If no CSV file is found in data/sample/.
    """
    csv_files = glob.glob(os.path.join(sample_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV dataset found in directory: {sample_dir}")
    # Return the first detected CSV file
    return csv_files[0]


def main():
    print("========================================")
    print("  Confidence Memo Agent Pipeline Demo  ")
    print("========================================\n")

    # Load environment variables (.env file containing GEMINI_API_KEY)
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        print("[WARNING] GEMINI_API_KEY is not set in environment or .env file.")
        print("API requests to Gemini may fail if credentials are not configured.\n")

    # Set UTF-8 encoding for stdout on Windows if supported
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # Stage 1: Load sample support ticket dataset
    print("Loading dataset...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sample_dir = os.path.join(base_dir, "data", "sample")

    try:
        csv_path = detect_sample_dataset(sample_dir)
        df = pd.read_csv(csv_path)
        print(f"[OK] Dataset loaded successfully ({len(df)} support tickets from '{os.path.basename(csv_path)}')\n")
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
            print("Retrieving tickets...")
            print("[OK] Done")

            print("Building memo...")
            # Call the public API generate_evidence_memo from MemoService
            memo = memo_service.generate_evidence_memo(
                df=df,
                keywords=keywords,
                proposal=proposal
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
