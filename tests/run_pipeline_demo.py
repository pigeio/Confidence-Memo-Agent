import os
import sys
import pandas as pd
from dotenv import load_dotenv

# Ensure the root of the project is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memo_service import MemoService
from src.gemini_client import GeminiClient


def main():
    print("=== Confidence Memo Agent Pipeline Demo ===")

    # 1. Load environment variables
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[WARNING] GEMINI_API_KEY environment variable is not set.")
        print(
            "Real calls to Gemini API will fail. Please set it in a .env file or your environment."
        )
        print("Attempting to run client using default setup...\n")

    # 2. Define files and parameters
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "data", "sample", "synthetic_tickets.csv")
    template_path = os.path.join(base_dir, "prompts", "evidence_prompt.txt")

    # 3. Load synthetic dataset
    if not os.path.exists(csv_path):
        print(f"[ERROR] Synthetic dataset CSV not found at: {csv_path}")
        return

    print(f"Loading synthetic tickets from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} support tickets.")

    # 4. Scenario: High Evidence (Dark Mode request)
    print("\n--------------------------------------------------")
    print("SCENARIO 1: High Evidence Feature Proposal")
    print("Proposal: 'Should we implement Dark Mode support?'")
    print("Keywords: ['dark', 'eye strain', 'theme']")
    print("--------------------------------------------------")

    keywords_dark = ["dark", "eye strain", "theme"]
    proposal_dark = "Should we implement Dark Mode support to reduce eye strain for our night-time users?"

    # Initialize services
    try:
        # We configure Client to use the API key explicitly
        client = GeminiClient(api_key=api_key)
        service = MemoService(gemini_client=client)

        print("Running orchestration pipeline...")
        memo_dark = service.generate_evidence_memo(
            df=df,
            keywords=keywords_dark,
            proposal=proposal_dark,
            template_path=template_path,
        )
        print("\n=== GENERATED CONFIDENCE MEMO ===")
        print(memo_dark)
        print("=================================\n")

    except Exception as e:
        print(f"[ERROR] Scenario 1 failed: {e}")

    # 5. Scenario: Moderate Evidence (CSV Export request)
    print("\n--------------------------------------------------")
    print("SCENARIO 2: Moderate Evidence Feature Proposal")
    print("Proposal: 'Should we add CSV data export capabilities?'")
    print("Keywords: ['export', 'csv']")
    print("--------------------------------------------------")

    keywords_export = ["export", "csv"]
    proposal_export = "Should we add a button to export dashboard transactions data into CSV files?"

    try:
        print("Running orchestration pipeline...")
        memo_export = service.generate_evidence_memo(
            df=df,
            keywords=keywords_export,
            proposal=proposal_export,
            template_path=template_path,
        )
        print("\n=== GENERATED CONFIDENCE MEMO ===")
        print(memo_export)
        print("=================================\n")

    except Exception as e:
        print(f"[ERROR] Scenario 2 failed: {e}")


if __name__ == "__main__":
    main()
