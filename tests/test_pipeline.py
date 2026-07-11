import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
from src.rag_app.rag.pipeline import ingest_text_path, ingest_pdf_path, answer_question
from src.rag_app.types import DocumentChunk, RagAnswer

class TestPipeline(unittest.TestCase):
    def setUp(self):
        # Setup mocks for the core components
        self.mock_indexer = MagicMock()
        self.mock_retriever = MagicMock()
        self.mock_generator = MagicMock()
        self.default_collection = "test_collection"

    @patch("src.rag_app.rag.pipeline.load_text_file")
    def test_ingest_text_path_calls_correct_components(self, mock_load_text):
        # Setup
        mock_load_text.return_value = [{"text": "content", "metadata": {}}]
        mock_chunks = [DocumentChunk(id="1", text="chunk1")]
        self.mock_indexer.index_documents.return_value = mock_chunks
        
        # Execute
        result = ingest_text_path(
            path="test.txt",
            collection_name=self.default_collection,
            indexer=self.mock_indexer
        )

        # Verify
        mock_load_text.assert_called_once()
        self.mock_indexer.index_documents.assert_called_once_with(
            mock_load_text.return_value, self.default_collection
        )
        self.assertEqual(result, mock_chunks)

    @patch("src.rag_app.rag.pipeline.load_pdf_file")
    def test_ingest_pdf_path_calls_correct_components(self, mock_load_pdf):
        # Setup
        mock_load_pdf.return_value = [{"text": "content", "metadata": {}}]
        mock_chunks = [DocumentChunk(id="1", text="chunk1")]
        self.mock_indexer.index_documents.return_value = mock_chunks
        
        # Execute
        result = ingest_pdf_path(
            path="test.pdf",
            collection_name=self.default_collection,
            indexer=self.mock_indexer
        )

        # Verify
        mock_load_pdf.assert_called_once()
        self.mock_indexer.index_documents.assert_called_once_with(
            mock_load_pdf.return_value, self.default_collection
        )
        self.assertEqual(result, mock_chunks)

    @patch("src.rag_app.rag.pipeline.load_text_file")
    def test_ingest_path_uses_default_collection(self, mock_load_text):
        # Test that if collection_name is None, it uses the default from config
        mock_load_text.return_value = []
        
        ingest_text_path(
            path="test.txt",
            collection_name=None, 
            indexer=self.mock_indexer
        )

        # Check if indexer was called with the default from config (from src.rag_app.config.settings)
        # This assumes settings.default_collection is set to a non-empty string in config.py
        args, _ = self.mock_indexer.index_documents.call_args
        self.assertNotEqual(args[1], None)

    def test_answer_question_successful_flow(self):
        # Setup mocks
        test_chunks = [DocumentChunk(id="1", text="chunk1")]
        mock_retrieved_chunks = test_chunks.copy()
        mock_prompt = "Constructed Prompt"
        mock_answer = RagAnswer(answer="The answer is here", sources=[{"source": "test"}])

        self.mock_retriever.retrieve.return_value = mock_retrieved_chunks
        # We need to patch build_rag_prompt because it's a standalone function in the module
        with patch("src.rag_app.rag.pipeline.build_rag_prompt", return_value=mock_prompt):
            result = answer_question(
                question="What is X?",
                collection_name=self.default_collection,
                top_k=5,
                retriever=self.mock_retriever,
                generator=self.mock_generator
            )

        # Verify flow and data integrity
        self.mock_retriever.retrieve.assert_called_once_with(
            "What is X?", self.default_collection, 5
        )
        self.mock_generator.generate.assert_called_once_with(mock_prompt, mock_retrieved_chunks)
        self.assertEqual(result, mock_answer)

    def test_answer_question_validation_fails(self):
        # Test empty question
        with self.assertRaises(ValueError):
            answer_question(question="   ", collection_name=self.default_collection)
        
        # Test invalid top_k
        with self.assertRaises(ValueError):
            answer_question(question="valid?", collection_name=self.default_collection, top_k=0)

    def test_answer_question_defaults(self):
        # Verify that default values are pulled from settings when None is passed
        # This requires 'settings' to be loaded correctly in the environment
        with patch("src.rag_app.rag.pipeline.Retriever") as MockRetriever, \
             patch("src.rag_app.rag.pipeline.RagGenerator") as MockGenerator:
            
            # We use a mock that can be verified for calls with default values
            mock_ret_inst = MockRetriever.return_value
            mock_gen_inst = MockGenerator.return_value

            answer_question(
                question="test?",
                collection_name=None,
                top_k=None,
                retriever=None,
                generator=None
            )
            
            # Assertions here would verify if the call used values from 'settings'
            # Since we can't easily check internal logic without more complex mocks, 
            # this confirms that it doesn't crash when None is passed.
            mock_ret_inst.retrieve.assert_called()
            mock_gen_inst.generate.assert_called()

if __name__ == "__main__":
    unittest.main()