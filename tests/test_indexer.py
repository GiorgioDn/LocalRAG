import unittest
from unittest.mock import MagicMock, patch
from rag_app.rag.indexer import DocumentIndexer
from rag_app.types import DocumentChunk, LoadedDocument

class TestDocumentIndexer(unittest.TestCase):
    def setUp(self):
        # Create mocks for the dependencies
        self.mock_embedding_client = MagicMock()
        self.mock_vector_store = MagicMock()
        
        # We override the chunking function's behavior via a mock 
        # if we were to inject it, but since DocumentIndexer uses a class 
        # attribute/property or a local reference that we can target:
        # For this test suite, we will initialize with mocks.
        self.indexer = DocumentIndexer(
            embedding_client=self.mock_embedding_client,
            vector_store=self.mock_vector_store
        )
        
        # We'll mock the chunking function specifically for tests where 
        # we want to control its output directly.
        # Since it is imported as a module-level function in indexer.py:
        self.patcher = patch("rag_app.rag.indexer.chunk_documents")
        self.mock_chunk_func = self.patcher.start()
        # Update the instance's reference to our mock
        self.indexer._chunk_func = self.mock_chunk_func

    def tearDown(self):
        self.patcher.stop()

    def test_successful_indexing_flow(self):
        """1 & 2: Test normal flow where data is correctly passed to downstream systems."""
        docs = [LoadedDocument(text="Sample text", metadata={"source": "test.txt"})]
        chunks = [DocumentChunk(id="chunk-1", text="chunk_content", metadata={"source": "test.txt"})]
        
        self.mock_chunk_func.return_value = chunks
        self.mock_embedding_client.embed_texts.return_value = [[0.1, 0.2]]
        
        # Use a fixed collection name
        result = self.indexer.index_documents(docs, collection_name="test_collection")

        # Assertions for logic flow
        self.assertEqual(result, chunks)
        self.mock_chunk_func.assert_called_once_with(docs)
        self.mock_embedding_client.embed_texts.assert_called_once_with(["chunk_content"])
        self.mock_vector_store.upsert_documents.assert_called_once_with(
            collection_name="test_collection",
            chunks=chunks,
            embeddings=[[0.1, 0.2]]
        )

    def test_empty_input_returns_empty_list(self):
        """3: If input is empty, no downstream calls are made."""
        result = self.indexer.index_documents([], collection_name="test_collection")
        
        self.assertEqual(result, [])
        self.mock_chunk_func.assert_not_called()
        self.mock_embedding_client.embed_texts.assert_not_called()
        self.mock_vector_store.upsert_documents.assert_not_called()

    def test_no_chunks_produced_returns_empty_list(self):
        """4: If chunker returns nothing, no embeddings or upserts occur."""
        docs = [LoadedDocument(text="Unchunkable text", metadata={})]
        self.mock_chunk_func.return_value = []

        result = self.indexer.index_documents(docs, collection_name="test_collection")

        self.assertEqual(result, [])
        self.mock_embedding_client.embed_texts.assert_not_called()
        self.mock_vector_store.upsert_documents.assert_not_called()

    def test_mismatched_embeddings_raises_runtime_error(self):
        """5: Raise RuntimeError if embedding count doesn't match chunk count."""
        docs = [LoadedDocument(text="Some text", metadata={})]
        chunks = [
            DocumentChunk(id="1", text="chunk 1", metadata={}),
            DocumentChunk(id="2", text="chunk 2", metadata={})
        ]
        self.mock_chunk_func.return_value = chunks
        # Only 1 embedding returned for 2 chunks
        self.mock_embedding_client.embed_texts.return_value = [[0.1]]

        with self.assertRaises(RuntimeError) as cm:
            self.indexer.index_documents(docs, collection_name="test_collection")
        
        self.assertIn("mismatch", str(cm.exception))
        # Ensure upsert was never called because of the error
        self.mock_vector_store.upsert_documents.assert_not_called()

    def test_explicit_collection_name_override(self):
        """6: Explicitly provided collection names are used."""
        docs = [LoadedDocument(text="test", metadata={})]
        self.mock_chunk_func.return_value = [DocumentChunk(id="1", text="t", metadata={})]
        self.mock_embedding_client.embed_texts.return_value = [[0.1]]

        self.indexer.index_documents(docs, collection_name="custom_collection")
        
        # Verify the name passed to the vector store is the custom one
        args, kwargs = self.mock_vector_store.upsert_documents.call_args
        self.assertEqual(kwargs['collection_name'], "custom_collection")

    def test_invalid_collection_name_raises_value_error(self):
        """7: Empty or whitespace collection names raise ValueError."""
        docs = [LoadedDocument(text="test", metadata={})]
        
        with self.assertRaises(ValueError):
            self.indexer.index_documents(docs, collection_name="   ")

    def test_metadata_preservation(self):
        """8: Metadata fields are passed to vector store untouched."""
        original_metadata = {
            "source": "test.pdf", 
            "source_type": "pdf", 
            "page": 10, 
            "chunk_index": 5
        }
        docs = [LoadedDocument(text="content", metadata=original_metadata)]
        chunks = [DocumentChunk(id="c1", text="content", metadata=original_metadata)]
        
        self.mock_chunk_func.return_value = chunks
        self.mock_embedding_client.embed_texts.return_value = [[0.1]]

        self.indexer.index_documents(docs, collection_name="test")

        # Verify the exact dictionary was passed to upsert
        args, kwargs = self.mock_vector_store.upsert_documents.call_args
        passed_metadata = kwargs['chunks'][0].metadata
        self.assertEqual(passed_metadata, original_metadata)
        self.assertEqual(passed_metadata["source"], "test.pdf")
        self.assertEqual(passed_metadata["page"], 10)

if __name__ == "__main__":
    unittest.main()