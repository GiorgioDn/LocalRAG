# tests/test_main_pdf.py
import sys
import unittest
from unittest.mock import patch, MagicMock
from src.rag_app.main_pdf import main

class TestMainPdf(unittest.TestCase):
    def setUp(self):
        # Mock the pipeline functions as they are imported in the module
        self.patch_ingest = patch('src.rag_app.main_pdf.ingest_pdf_path')
        self.patch_answer = patch('src.rag_app.main_pdf.answer_question')

        self.mock_ingest = self.patch_ingest.start()
        self.mock_answer = self.patch_answer.start()

    def tearDown(self):
        self.patch_ingest.stop()
        self.patch_answer.stop()

    def test_main_success_full_args(self):
        """Test full flow: pdf path, collection, and question."""
        # Mock the response as a dict as expected by main_pdf.py (key access)
        mock_result = {
            "answer": "This is a PDF test answer.",
            "sources": [{"metadata": {"source": "page1"}}]
        }
        self.mock_answer.return_value = mock_result

        test_args = ["main_pdf.py", "data/doc.pdf", "--collection", "pdf_col", "--question", "Tell me more."]
        with patch('sys.argv', test_args):
            main()

        self.mock_ingest.assert_called_once_with(
            pdf_path="data/doc.pdf", 
            collection_name="pdf_col"
        )
        self.mock_answer.assert_called_once_with(
            question="Tell me more.", 
            collection_name="pdf_col"
        )

    def test_main_no_question(self):
        """Test flow when no question is provided."""
        test_args = ["main_pdf.py", "data/doc.pdf"]
        with patch('sys.argv', test_args):
            main()

        self.mock_ingest.assert_called_once_with(
            pdf_path="data/doc.pdf", 
            collection_name="default_collection" # Default from argparse in main_pdf.py
        )
        self.mock_answer.assert_not_called()

    def test_invalid_arguments(self):
        """Test that invalid arguments result in a system exit."""
        test_args = ["main_pdf.py"] # Missing pdf_path
        with patch('sys.argv', test_args):
            with self.assertRaises(SystemExit):
                main()

        self.mock_ingest.assert_not_called()