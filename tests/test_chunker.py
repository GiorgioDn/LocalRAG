import pytest
from rag_app.ingestion.chunker import chunk_documents
from rag_app.types import DocumentChunk, LoadedDocument

def test_chunk_size_and_determinism():
    """Verify that text is split into chunks based on size and produces identical results."""
    docs = [
        LoadedDocument(
            text="a" * 100 + " " + "b" * 100, # Long string to ensure multiple chunks or specific sizing
            metadata={"source": "doc1", "source_type": "text"}
        )
    ]
    # With size=50 and overlap=0, we expect roughly two segments if separated by space.
    chunks = chunk_documents(docs, chunk_size=50, chunk_overlap=0)
    
    assert len(chunks) > 1
    assert len(chunks[0].text) <= 50
    # Determinism check: run again
    chunks_again = chunk_documents(docs, chunk_size=50, chunk_overlap=0)
    assert chunks == chunks_again

def test_chunk_overlap():
    """Verify overlap exists between adjacent chunks."""
    text = "the quick brown_fox jumps over the lazy dog" # underline for clarity in logic
    # We use a small size and significant overlap to force overlapping content.
    doc = LoadedDocument(
        text="1234567890", 
        metadata={"source": "test", "source_type": "text"}
    )
    # chunk_size=5, chunk_overlap=3.
    # Chunk 0: [0:5] -> "12345"
    # Next position: 5 - 3 = 2.
    # Chunk 1: [2:7] -> "34567" (Overlaps with "345")
    chunks = chunk_documents([doc], chunk_size=5, chunk_overlap=3)
    
    assert len(chunks) == 2
    assert chunks[0].text in ("12345", "12345 ") # Depending on strip() logic
    assert chunks[1].text in ("34567", "34567 ")
    # Check overlapping part
    common = set(chunks[0].text) & set(chunks[1].text)
    assert len(common) > 0

def test_no_empty_chunks():
    """Verify that no empty chunks are emitted."""
    docs = [
        LoadedDocument(
            text="   \n  \t  content", # Leading/trailing whitespace or just whitespace
            metadata={"source": "test", "source_type": "text"}
        )
    ]
    chunks = chunk_documents(docs, chunk_size=10, chunk_overlap=2)
    for c in chunks:
        assert len(c.text.strip()) > 0

def test_metadata_copying():
    """Verify metadata from the loader is copied rather than mutated."""
    original_meta = {"source": "orig", "source_type": "doc", "extra": "keep"}
    docs = [LoadedDocument(text="some text", metadata=original_meta)]
    
    chunks = chunk_documents(docs, chunk_size=10)
    
    # Ensure the original dict wasn't changed (e.g., by adding METADATA_CHUNK_INDEX)
    assert "extra" in original_meta
    assert len(original_meta) == 3
    # Check that the chunk has its own copy
    assert chunks[0].metadata["source"] == "orig"
    assert "chunk_index" in chunks[0].metadata # Note: ensure key name matches constant

def test_chunk_indexing():
    """Verify chunk_index starts from zero for each input document."""
    docs = [
        LoadedDocument(text="doc1 text", metadata={"source": "d1"}),
        LoadedDocument(text="doc2 text", metadata={"source": "d2"})
    ]
    chunks = chunk_documents(docs, chunk_size=5)
    
    # Check indices for doc 1
    assert chunks[0].metadata["chunk_index"] == 0
    # Check indices for doc 2 (should restart at 0)
    assert chunks[1].metadata["chunk_index"] == 0

def test_pdf_metadata_persistence():
    """Verify PDF metadata survives chunking."""
    docs = [
        LoadedDocument(
            text="page one content",
            metadata={"source": "file.pdf", "source_type": "pdf", "page": 1}
        )
    ]
    chunks = chunk_documents(docs, chunk_size=5)
    
    for c in chunks:
        assert c.metadata["source"] == "file.pdf"
        assert c.metadata["source_type"] == "pdf"
        assert c.metadata["page"] == 1

def test_stable_ids():
    """Verify that IDs are stable and follow the naming convention."""
    # Test text source without page
    doc_text = LoadedDocument(
        text="some content", 
        metadata={"source": "guide.md", "source_type": "text"}
    )
    chunks = chunk_documents([doc_text])
    assert chunks[0].id == "guide.md-pnone-c0"

    # Test PDF source with page
    doc_pdf = LoadedDocument(
        text="some content", 
        metadata={"source": "/path/to/manual.pdf", "source_type": "pdf", "page": 12}
    )
    chunks_pdf = chunk_documents([doc_pdf])
    assert chunks_pdf[0].id == "manual.pdf-p12-c0"

def test_id_determinism():
    """Verify that repeated calls with identical input produce identical IDs."""
    doc = LoadedDocument(
        text="stable", 
        metadata={"source": "data.md"}
    )
    chunks1 = chunk_documents([doc])
    chunks2 = chunk_documents([doc])
    assert chunks1[0].id == chunks2[0].id

def test_invalid_settings():
    """Verify invalid chunk settings raise ValueError."""
    docs = [LoadedDocument(text="test", metadata={})]
    
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_documents(docs, chunk_size=0)
    
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_documents(docs, chunk_overlap=-1)
        
    with pytest.raises(ValueError, match="must be less than"):
        chunk_documents(docs, chunk_size=10, chunk_overlap=10)

def test_document_isolation():
    """Verify content from separate documents is never combined."""
    docs = [
        LoadedDocument(text="doc1", metadata={"source": "d1"}),
        LoadedDocument(text="doc2", metadata={"source": "d2"})
    ]
    # Even with small chunks, they shouldn't bleed into each other
    chunks = chunk_documents(docs, chunk_size=5)
    assert len(chunks) == 2
    assert "doc1" in chunks[0].text
    assert "doc1" not in chunks[1].text
    assert "doc2" in chunks[1].text
    assert "doc2" not in chunks[0].text