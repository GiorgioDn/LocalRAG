import unittest
from unittest.mock import MagicMock
from src.rag_app.rag.retriever import Retriever
from src.rag_app.types import RetrievedChunk

class MockEmbeddingClient:
    def embed_query(self, query: str) -> list[float]:
        return [0.1, 0.2, 0.3]

class MockChromaConnector:
    def search(self, collection_name: str, query_embedding: list[float], top_k: int) -> list[dict]:
        # This will be mocked/overridden in specific test cases if needed,
        # but the base mock is for dependency injection.
        return []

class TestRetriever(unittest.TestCase):
    def setUp(self):
        self.mock_embedding = MagicMock()
        self.mock_vector_store = MagicMock()
        self.retriever = Retriever(
            embedding_client=self.mock_embedding, 
            vector_store=self.mock_vector_store
        )

    def test_retrieve_success(self):
        """Test a successful retrieval with valid inputs."""
        self.mock_embedding.embed_query.return_value = [0.1, 0.2]
        self.mock_vector_store.search.return_value = [
            {"text": "chunk1", "metadata": {"source": "a"}, "score": 0.9},
            {"text": "chunk2", "metadata": {"source": "b"}, "score": 0.8}
        ]

        results = self.retriever.retrieve(query="hello", top_k=2)

        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], RetrievedChunk)
        self.assertEqual(results[0].text, "chunk1")
        self.assertEqual(results[0].score, 0.9)
        self.mock_embedding.embed_query.assert_called_once_with("hello")
        self.mock_vector_store.search.assert_called_once()

    def test_retrieve_empty_query(self):
        """Test that an empty or whitespace-only query raises ValueError."""
        with self.assertRaisesRegex(ValueError, "Query must not be empty"):
            self.retriever.retrieve(query="   ")
        
        with self.assertRaisesRegex(ValueError, "Query must not be empty"):
            self.retriever.retrieve(query="")

    def test_retrieve_invalid_top_k(self):
        """Test that an invalid top_k value raises ValueError."""
        with self.assertRaisesRegex(ValueError, "top_k must be a positive integer"):
            self.retriever.retrieve(query="hello", top_k=0)
        
        with self.assertRaisesRegex(ValueError, "top_k must be a positive integer"):
            self.retriever.retrieve(query="hello", top_k=-5)

    def test_retrieve_no_results(self):
        """Test retrieval when the vector store returns an empty list."""
        self.mock_embedding.embed_query.return_value = [0.1, 0.2]
        self.mock_vector_store.search.return_value = []

        results = self.retriever.retrieve(query="no results", top_k=5)
        self.assertEqual(results, [])

    def test_retrieve_preserves_order(self):
        """Test that the order of results from the vector store is preserved."""
        self.mock_embedding.embed_query.return_value = [0.1]
        search_results = [
            {"text": "first", "metadata": {}, "score": 0.9},
            {"text": "second", "metadata": {}, "score": 0.8}
        ]
        self.mock_vector_store.search.return_value = search_results

        results = self.retriever.retrieve(query="order check")
        self.assertEqual(results[0].text, "first")
        self.assertEqual(results[1].text, "second")

    def test_retrieve_malformed_results(self):
        """Test that malformed results from the vector store raise RuntimeError."""
        self.mock_embedding.embed_query.return_value = [0.1]
        # Missing 'score' field
        self.mock_vector_store.search.return_value = [
            {"text": "bad_record", "metadata": {}} 
        ]

        with self.assertRaises(RuntimeError) as cm:
            self.retriever.retrieve(query="malformed")
        self.assertIn("malformed record", str(cm.exception))

    def test_metadata_is_copied(self):
        """Ensure that metadata is a copy and not the original reference."""
        original_metadata = {"source": "origin"}
        self.mock_embedding.embed_query.return_value = [0.1]
        self.mock_vector_store.search.return_value = [
            {"text": "chunk", "metadata": original_metadata, "score": 0.9}
        ]

        results = self.retriever.retrieve(query="meta test")
        
        # Modify the result metadata and ensure it doesn't change the source object
        results[0].metadata["source"] = "modified"
        self.assertNotEqual(results[0].metadata["source"], original_metadata["source"])
        self.assertEqual(original_metadata["source"], "origin")

if __name__ == "__main__":
    unittest.main()