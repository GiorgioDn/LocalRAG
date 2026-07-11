import pytest
from typing import Any, Dict
from src.rag_app.types import RetrievedChunk
from src.rag_app.rag.prompt_builder import build_rag_prompt

def test_build_rag_prompt_empty_query():
    """Test that an empty or whitespace-only query raises a ValueError."""
    with pytest.raises(ValueError, match="Question cannot be empty."):
        build_rag_prompt("", [])
    with pytest.raises(ValueError, match="Question cannot be empty."):
        build_rag_prompt("   ", [])

def test_source_and_page_metadata_in_prompt():
    """Test that source and page metadata are correctly formatted in the prompt."""
    chunks = [
        RetrievedChunk(
            content="Sample content.",
            metadata={"source": "manual.pdf", "page": 12, "source_type": "pdf"}
        )
    ]
    prompt = build_rag_prompt("What is the main point?", chunks)
    assert "[Context 1 | source: manual.pdf | page: 12 | type: pdf]" in prompt
    assert "Sample content." in prompt

def test_text_file_no_page_metadata():
    """Test that text files (missing page metadata) don't include a meaningless page number."""
    chunks = [
        RetrievedChunk(
            content="Text content.",
            metadata={"source": "notes.txt"}
        )
    ]
    prompt = build_rag_prompt("What is this?", chunks)
    # Should not contain 'page' if it's missing from metadata
    assert "source: notes.txt" in prompt
    assert "page" not in prompt

def test_retrieved_chunk_order_preserved():
    """Test that the order of retrieved chunks is preserved in the final prompt."""
    chunks = [
        RetrievedChunk(content="First chunk", metadata={"source": "1.txt"}),
        RetrievedChunk(content="Second chunk", metadata={"source": "2.txt"}),
    ]
    prompt = build_rag_prompt("Question?", chunks)
    assert prompt.find("First chunk") < prompt.find("Second chunk")
    assert "Context 1" in prompt
    assert "Context 2" in prompt

def test_no_context_fabrication_warning():
    """Test that even with no context, the model is warned not to fabricate."""
    prompt = build_rag_prompt("What is X?", [])
    assert "Do not invent facts or sources." in prompt
    assert "No relevant context was found." in prompt

def test_incomplete_metadata_handling():
    """Test that missing keys in metadata do not cause a crash."""
    chunks = [
        RetrievedChunk(content="Content", metadata={}) # Empty metadata
    ]
    # Should not raise KeyError
    prompt = build_rag_prompt("Question?", chunks)
    assert "source: unknown source" in prompt

def test_metadata_immutability():
    """Test that the input metadata dictionaries are not modified by the builder."""
    original_metadata = {"source": "test.pdf", "page": 5}
    chunks = [RetrievedChunk(content="Content", metadata=original_metadata)]
    
    build_rag_prompt("Question?", chunks)
    
    assert original_metadata == {"source": "test.pdf", "page": 5}
    # Ensure we didn't just modify a copy but the actual passed dictionary (if it were shared)