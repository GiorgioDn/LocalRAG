# tests/test_main_text.py
import sys
import unittest
from unittest.mock import patch, MagicMock
from src.rag_app.main_text import main

class TestMainText(unittest.TestCase):
    def setUp(self):
        # We need to mock the pipeline functions as they are imported 
        # in the module where 'main' is defined.
        self.patch_ingest = patch('src.rag_app.main_text.ingest_text_path')
        self.patch_answer = patch('src.rag_app.main_text.answer_question')

        self.mock_ingest = self.patch_ingest.start()
        self.mock_answer = self.patch_answer.start()

    def tearDown(self):
        self.patch_ingest.stop()
        self.patch_answer.stop()

    def test_main_success_full_args(self):
        """Test full flow: path, collection, and question."""
        # Mock the response object as expected by main_text.py (attribute access)
        mock_response = MagicMock()
        mock_response.answer = "This is a test answer."
        mock_response.sources = ["source1", "source2"]
        self.mock_answer.return_value = mock_response

        # Simulate CLI args
        test_args = ["main_text.py", "data/test.txt", "--collection", "my_collection", "--question", "What is test?"]
        with patch('sys.argv', test_args):
            # Capture stdout if needed, but check calls mainly
            main()

        self.mock_ingest.assert_called_once_with(path="data/test.txt", collection_name="my_collection")
        self.mock_answer.assert_called_once_with(question="What is test?", collection_name="my_collection")

    def test_main_no_question(self):
        """Test flow when no question is provided (only ingestion)."""
        test_args = ["main_text.py", "data/test.txt", "--collection", "my_collection"]
        with patch('sys.argv', test_args):
            main()

        self.mock_ingest.assert_called_once_with(path="data/test.txt", collection_name="my_collection")
        self.mock_answer.assert_not_called()

    def test_invalid_arguments(self):
        """Test that invalid arguments do not call the pipeline and exit."""
        test_args = ["main_text.py"] # Missing 'path' argument
        with patch('sys.argv', test_args):
            with self.assertRaises(SystemExit):
                main()

        self.mock_ingest.assert_not_called()