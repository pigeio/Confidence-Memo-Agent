import unittest
import pandas as pd
from src.retrieval import retrieve_tickets


class TestRetrieval(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame for testing
        self.sample_data = {
            "ticket_id": [1, 2, 3, 4],
            "topic": ["Dark mode support", "Export dashboard data", "App crashes", "Eye strain"],
            "message": [
                "The screen is too bright. We need a dark theme.",
                "Is there a way to export to CSV format?",
                "The login page crashes with an error.",
                "I get severe eye strain when working late hours.",
            ],
        }
        self.df = pd.DataFrame(self.sample_data)

    def test_successful_single_keyword_match(self):
        # Search for "dark"
        result = retrieve_tickets(self.df, ["dark"])
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["ticket_id"], 1)

    def test_successful_multiple_keywords_match(self):
        # Search for "crashes" and "strain"
        result = retrieve_tickets(self.df, ["crashes", "strain"])
        self.assertEqual(len(result), 2)
        matched_ids = set(result["ticket_id"])
        self.assertEqual(matched_ids, {3, 4})

    def test_case_insensitivity(self):
        # Search with mixed case "DaRk"
        result = retrieve_tickets(self.df, ["DaRk"])
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["ticket_id"], 1)

    def test_no_match(self):
        # Search for a keyword that does not exist
        result = retrieve_tickets(self.df, ["billing"])
        self.assertTrue(result.empty)

    def test_regex_special_characters_escaping(self):
        # Search with special regex characters
        result = retrieve_tickets(self.df, ["mode?"])
        self.assertTrue(result.empty)  # "mode?" matches exactly "mode?" not "mode" optional

        df_special = pd.DataFrame({
            "ticket_id": [1],
            "topic": ["Special Characters"],
            "message": ["Does this support dark mode? Yes, it does."]
        })
        result_special = retrieve_tickets(df_special, ["mode?"])
        self.assertEqual(len(result_special), 1)

    def test_invalid_df_type(self):
        with self.assertRaises(TypeError):
            retrieve_tickets("not a dataframe", ["dark"])

    def test_missing_columns(self):
        invalid_df = pd.DataFrame({"id": [1], "message": ["Hello"]})
        with self.assertRaises(ValueError):
            retrieve_tickets(invalid_df, ["hello"])

    def test_invalid_keywords_type(self):
        with self.assertRaises(TypeError):
            retrieve_tickets(self.df, "not a list")

    def test_empty_keywords_list(self):
        with self.assertRaises(ValueError):
            retrieve_tickets(self.df, [])

    def test_non_string_keyword(self):
        with self.assertRaises(TypeError):
            retrieve_tickets(self.df, ["dark", 123])

    def test_empty_keyword_string(self):
        with self.assertRaises(ValueError):
            retrieve_tickets(self.df, ["dark", ""])
        with self.assertRaises(ValueError):
            retrieve_tickets(self.df, ["  "])


if __name__ == "__main__":
    unittest.main()
