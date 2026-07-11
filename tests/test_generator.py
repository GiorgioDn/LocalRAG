import unittest
from unittest.mock import MagicMock
from src.rag_app.rag.generator import RagGenerator
from src.rag_app.types import RagAnswer, RetrievedChunk

class MockChatClient:
    """A mock chat client for testing purposes."""
    def __init__(self):
        self.call_count = 0

    def answer(self, prompt: str) -> str:
        self.call_count += 1
        return f"Response to: {prompt}"

class TestRagGenerator(unittest.TestCase):
    def setUp(self):
        self.mock_client = MockChatClient()
        self.generator = RagGenerator(chat_client=self.mock_client)

    def test_successful_generation(self):
        """Verify that a valid prompt and chunks produce a RagAnswer."""
        prompt = "Tell me about the company."
        chunks = [
            RetrievedChunk(content="Some content", metadata={"source": "doc1", "page": 1})
        ]
        
        result = self.generator.generate(prompt, chunks)
        
        self.assertIsInstance(result, RagAnswer)
        self.assertEqual(result.answer, f"Response to: {prompt}")
        self.assertEqual(len(result.sources), 1)
        self.assertEqual(self.mock_client.call_count, 1)

    def test_metadata_copying(self):
        """Verify that source metadata is returned as a copy of the dictionary."""
        prompt = "Tell me about the company."
        original_metadata = {"source": "doc1", "page": 1}
        chunks = [RetrievedChunk(content="Some content", metadata=original_metadata)]
        
        result = self.generator.generate(prompt, chunks)
        
        # Ensure it's a new dictionary object
        self.assertIsNot(result.sources[0], original_metadata)
        self.assertEqual(result.sources[0], original_metadata)

    def test_deduplication_and_ordering(self):
        """Verify that sources are deduplicated correctly based on (source, page)."""
        prompt = "Tell me about the company."
        chunks = [
            RetrievedChunk(content="c1", metadata={"source": "doc1", "page": 1}),  # First occurrence of doc1 p1
            RetrievedChunk(content="c2", metadata={"source": "doc1", "page": 1}),  # Duplicate of doc1 p1
            RetrievedChunk(content="c3", metadata={"source": "doc1", "page": 2}),  # Different page of same doc
            RetrievedChunk(content="c4", metadata={"source": "doc2", "page": 1}),  # New source
            RetrievedChunk(content="c5", metadata={"source": "text_file"}),        # Text source (page=None)
            RetrievedChunk(content="c6", metadata={"source": "text_file"}),        # Duplicate of text_file
        ]

        result = self.generator.generate(prompt, chunks)
        
        # Expected unique sources: 
        # 1. doc1 p1
        # 2. doc1 p2
        # 3. doc2 p1
        # 4. text_file (where page is None)
        self.assertEqual(len(result.sources), 4)
        
        source_keys = [("doc1", 1), ("doc1", 2), ("doc2", 1), ("text_file", None)]
        for i, chunk in enumerate(chunks):
            # Only some indices should appear in the result because of deduplication
            pass # The logic is already handled by unique keys.

        self.assertEqual(result.sources[0]["source"], "doc1")
        self.assertEqual(result.sources[0]["page"], 1)
        self.assertEqual(result.sources[1]["source"], "doc1")
        self.assertEqual(result.sources[1]["page"], 2)
        self.assertEqual(result.sources[2]["source"], "doc2")
        self.assertEqual(result.sources[3]["source"], "text_file")

    def test_empty_prompt_validation(self):
        """Verify that empty or whitespace-only prompts raise a ValueError."""
        chunks = [RetrievedChunk(content="c1", metadata={"source": "d1"})]
        
        with self.assertRaises(ValueError) as cm:
            self.generator.generate("", chunks)
        self.assertEqual(str(cm.exception), "Prompt must not be empty or whitespace-only.")

        with self.assertRaises(ValueError):
            self.generator.generate("   ", chunks)

    def test_chat_client_failure_propagation(self):
        """Verify that exceptions from the chat client are propagated."""
        # Replace our mock with one that raises an error
        faulty_client = MockChatClient()
        def faulty_answer(prompt):
            raise RuntimeError("API Connection Failed")
        
        faulty_client.answer = faulty_answer
        generator = RagGenerator(chat_client=faulty_client)

        with self.assertRaises(RuntimeError) as cm:
            generator.generate("Hello", [])
        self.assertEqual(str(cm.exception), "API Connection Failed")