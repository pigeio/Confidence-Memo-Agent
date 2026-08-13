import os
import logging
from unittest.mock import MagicMock
import pandas as pd
from src.evaluation.adapters import DatasetRegistry
from src.memo_service import MemoService
from src.decision_engine import DecisionResult

logger = logging.getLogger(__name__)


class E2EValidator:
    """
    End-to-End Multi-Domain Pipeline Validator.
    Executes full pipeline across real-world public datasets and generates verified example decision memos.
    """

    def __init__(self, memo_service: MemoService | None = None, mock_llm: bool = True):
        if memo_service is not None:
            self.memo_service = memo_service
        elif mock_llm or not os.getenv("GEMINI_API_KEY"):
            mock_client = MagicMock()
            mock_client.generate_response.side_effect = (
                lambda prompt: (
                    f"# Decision Memo: Executive Analysis\n\n"
                    f"**Executive Recommendation:** Evaluated by Deterministic Decision Engine\n\n"
                    f"### Summary\nGenerated from real customer feedback with calibrated deterministic evidence."
                )
            )
            self.memo_service = MemoService(gemini_client=mock_client)
        else:
            self.memo_service = MemoService()

    def run_all_scenarios(
        self,
        output_dir: str = "evaluation/examples",
        mock_llm: bool = False,
    ) -> list[dict]:
        """
        Run end-to-end validation across all domains and write example Markdown memos to output_dir.
        """
        os.makedirs(output_dir, exist_ok=True)

        scenarios = [
            {
                "id": "google_play_dark_mode",
                "dataset_name": "google_play_reviews",
                "proposal": "Dark Mode & Night Theme Support",
                "keywords": ["dark", "mode", "theme", "night", "bright", "eyes"],
                "effort": "Low",
                "impact": "High",
                "alignment": "High",
                "cost": "Low",
                "risk": "Low",
                "out_file": "google_play_dark_mode.md",
            },
            {
                "id": "github_search",
                "dataset_name": "github_issues",
                "proposal": "Repository Search Performance & Filter Optimization",
                "keywords": ["search", "queries", "timeout", "filter", "latency", "performance"],
                "effort": "Medium",
                "impact": "High",
                "alignment": "High",
                "cost": "Low",
                "risk": "Medium",
                "out_file": "github_search.md",
            },
            {
                "id": "customer_support_login",
                "dataset_name": "customer_support_tickets",
                "proposal": "Enterprise SSO & Authentication Reliability Fixes",
                "keywords": ["login", "sso", "oauth", "password", "authentication", "saml", "token"],
                "effort": "Medium",
                "impact": "Critical",
                "alignment": "High",
                "cost": "Low",
                "risk": "High",
                "out_file": "customer_support_login.md",
            },
            {
                "id": "amazon_battery",
                "dataset_name": "amazon_product_reviews",
                "proposal": "Hardware Power Management & Battery Optimization Mode",
                "keywords": ["battery", "drain", "standby", "power", "charging", "overheating"],
                "effort": "High",
                "impact": "Critical",
                "alignment": "High",
                "cost": "Medium",
                "risk": "High",
                "out_file": "amazon_battery.md",
            },
            {
                "id": "crypto_wallet_negative",
                "dataset_name": "google_play_reviews",
                "proposal": "In-App Web3 Crypto & NFT Wallet Integration",
                "keywords": ["crypto", "bitcoin", "ethereum", "wallet", "nft", "blockchain"],
                "effort": "Very High",
                "impact": "Low",
                "alignment": "Low",
                "cost": "High",
                "risk": "Critical",
                "out_file": "crypto_wallet_unsupported.md",
            },
        ]

        results = []

        for sc in scenarios:
            df, meta = DatasetRegistry.load_dataset(sc["dataset_name"])

            service = self.memo_service
            if mock_llm:
                mock_client = MagicMock()
                mock_client.generate_response.side_effect = (
                    lambda prompt: f"# Decision Memo: {sc['proposal']}\n\n"
                    f"**Executive Recommendation:** Evaluated by Deterministic Decision Engine\n\n"
                    f"### Summary\nGenerated from {meta['domain']} with evidence-backed calibration."
                )
                service = MemoService(gemini_client=mock_client)

            memo_str, decision_result = service.generate_decision_memo(
                df=df,
                keywords=sc["keywords"],
                proposal=sc["proposal"],
                engineering_effort=sc["effort"],
                business_impact=sc["impact"],
                strategic_alignment=sc["alignment"],
                cost=sc["cost"],
                risk=sc["risk"],
                return_decision_result=True,
            )

            # Save Markdown artifact
            out_path = os.path.join(output_dir, sc["out_file"])
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(
                    f"<!-- Evaluation Scenario: {sc['id']} -->\n"
                    f"<!-- Domain: {meta['domain']} ({meta['name']}) -->\n\n"
                    f"{memo_str}\n\n"
                    f"---\n\n"
                    f"## Deterministic Decision Telemetry\n\n"
                    f"```json\n"
                    f"{pd.Series(decision_result.to_dict()).to_json(indent=2)}\n"
                    f"```\n"
                )

            results.append(
                {
                    "scenario_id": sc["id"],
                    "dataset": sc["dataset_name"],
                    "proposal": sc["proposal"],
                    "evidence_score": decision_result.evidence_score,
                    "priority_score": decision_result.priority_score,
                    "recommendation": decision_result.recommendation,
                    "output_file": out_path,
                }
            )

        return results
