import unittest
import tempfile
import os
import pandas as pd
from src.prompt_builder import build_evidence_prompt, format_tickets


class TestPromptBuilder(unittest.TestCase):
    def setUp(self):
        self.sample_data = {
            "ticket_id": [1, 2],
            "topic": ["Dark Mode", "CSV Export"],
            "message": ["Please add dark mode", "We want CSV export"],
        }
        self.df = pd.DataFrame(self.sample_data)
        self.proposal = "Add dark mode support"

    def test_format_tickets_empty(self):
        empty_df = pd.DataFrame(columns=["ticket_id", "topic", "message"])
        result = format_tickets(empty_df)
        self.assertEqual(result, "No tickets found matching the keyword criteria.")

    def test_format_tickets_valid(self):
        result = format_tickets(self.df)
        self.assertIn("Ticket ID: 1", result)
        self.assertIn("Topic: Dark Mode", result)
        self.assertIn("Message: Please add dark mode", result)
        self.assertIn("---", result)

    def test_build_prompt_default_template(self):
        prompt = build_evidence_prompt(self.proposal, self.df)
        self.assertIn("Feature Proposal:\nAdd dark mode support", prompt)
        self.assertIn("Ticket ID: 1", prompt)
        self.assertIn("Ticket ID: 2", prompt)

    def test_build_prompt_file_template(self):
        # We can use the actual template file from prompts/evidence_prompt.txt
        template_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "evidence_prompt.txt")
        # Ensure it exists (it was created in the previous step)
        if os.path.exists(template_path):
            prompt = build_evidence_prompt(self.proposal, self.df, template_path=template_path)
            self.assertIn("Feature Proposal:\nAdd dark mode support", prompt)
            self.assertIn("Ticket ID: 1", prompt)
        else:
            self.skipTest("prompts/evidence_prompt.txt not found")

    def test_invalid_proposal_type(self):
        with self.assertRaises(ValueError):
            build_evidence_prompt(123, self.df)

    def test_empty_proposal(self):
        with self.assertRaises(ValueError):
            build_evidence_prompt("", self.df)
        with self.assertRaises(ValueError):
            build_evidence_prompt("   ", self.df)

    def test_invalid_df_type(self):
        with self.assertRaises(TypeError):
            build_evidence_prompt(self.proposal, "not a dataframe")

    def test_df_missing_columns(self):
        invalid_df = pd.DataFrame({"id": [1], "msg": ["Hello"]})
        with self.assertRaises(ValueError):
            build_evidence_prompt(self.proposal, invalid_df)

    def test_missing_file_template(self):
        with self.assertRaises(FileNotFoundError):
            build_evidence_prompt(self.proposal, self.df, template_path="nonexistent_template.txt")

    def test_bad_template_placeholders(self):
        # Create a temporary file with bad placeholders
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp:
            tmp.write("This is a bad template with {wrong_placeholder}")
            tmp_path = tmp.name

        try:
            with self.assertRaises(KeyError):
                build_evidence_prompt(self.proposal, self.df, template_path=tmp_path)
        finally:
            os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
