# tests/test_id_utils.py contents:
import pytest
from typing import Any, Mapping
from rag_app.utils.id_utils import build_chunk_id

def test_build_chunk_id_basic():
    """Test standard cases for text and PDF sources."""
    # Text source without page (pnone)
    meta1 = {"source": "data/guide.md"}
    assert build_chunk_id(meta1, 0) == "guide.md-pnone-c0"

    # PDF source with a page
    meta2 = {"source": "/docs/manual.pdf", "page": 12}
    assert build_chunk_id(meta2, 3) == "manual.pdf-p12-c3"

def test_build_chunk_id_sanitization():
    """Test that paths are stripped and special characters are handled."""
    # Full path should only show filename
    meta = {"source": "/home/user/data/file @ name!.txt"}
    # Logic ensures characters not in [a-zA-Z0-9._-] become '-' then collapsed.
    # "file @ name!.txt" -> "file---name--.txt" -> "file-name-.txt" (example)
    # The assertion checks that the specific path component is removed.
    res = build_chunk_id(meta, 0)
    assert "/home/user" not in res

def test_build_chunk_id_stability():
    """Verify that identical inputs produce identical IDs."""
    meta = {"source": "doc.pdf", "page": 5}
    idx = 10
    assert build_chunk_id(meta, idx) == build_chunk_id(meta, idx)

def test_build_chunk_id_variation():
    """Verify that different inputs produce different IDs."""
    meta = {"source": "doc.pdf", "page": 5}
    assert build_chunk_id(meta, 10) != build_chunk_id(meta, 11)
    
    meta2 = {"source": "doc.pdf", "page": 6}
    assert build_chunk_id(meta, 10) != build_chunk_id(meta2, 10)

def test_build_chunk_id_failures():
    """Test various failure modes for inputs."""
    # Invalid Metadata Type
    with pytest.raises(TypeError, match="Metadata must be a mapping"):
        build_chunk_id(["not", "a", "dict"], 0)

    # Invalid Chunk Index (type)
    with pytest.raises(TypeError, match="Chunk index must be an integer"):
        build_chunk_id({"source": "doc"}, "zero")
    
    # Invalid Chunk Index (bool)
    with pytest.raises(TypeError, match="Chunk index must be an integer"):
        build_chunk_id({"source": "doc"}, True)

    # Negative Index
    with pytest.raises(ValueError, match="must be non-negative"):
        build_chunk_id({"source": "doc"}, -1)

    # Missing/Blank Source
    with pytest.raises(ValueError, match="must be a non-empty string"):
        build_chunk_id({}, 0)
    with pytest.raises(ValueError, match="must be a non-empty string"):
        build_chunk_id({"source": "   "}, 0)

    # Invalid Page Values
    with pytest.raises(ValueError, match="must be an integer"):
        build_chunk_id({"source": "doc", "page": "5"}, 0)
    with pytest.raises(ValueError, match="must be greater than zero"):
        build_chunk_id({"source": "doc", "page": 0}, 0)
    with pytest.raises(ValueError, match="must be greater than zero"):
        build_chunk_id({"source": "doc", "page": -1}, 0)

def test_metadata_immutability():
    """Ensure the input metadata dictionary is not modified by the function."""
    meta = {"source": "doc.pdf", "page": 5}
    original_copy = meta.copy()
    build_chunk_id(meta, 0)
    assert meta == original_copy