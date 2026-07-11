import pytest
from pathlib import Path
from unittest.mock import patch
from rag_app.ingestion.text_loader import load_text_file
from rag_app.types import LoadedDocument
from rag_app.constants import METADATA_SOURCE, METADATA_SOURCE_TYPE, SOURCE_TYPE_TEXT

def test_load_text_file_success(tmp_path):
    """Test that a .txt file loads successfully and returns one LoadedDocument."""
    file_path = tmp_path / "test.txt"
    content = "Hello world\nThis is a test."
    file_path.write_text(content, encoding="utf-8")

    results = load_text_file(str(file_path))

    assert len(results) == 1
    doc = results[0]
    assert isinstance(doc, LoadedDocument)
    assert doc.text == content
    assert doc.metadata[METADATA_SOURCE] == str(file_path)
    assert doc.metadata[METADATA_SOURCE_TYPE] == SOURCE_TYPE_TEXT
    assert "page" not in doc.metadata
    assert "chunk_index" not in doc.metadata

@pytest.mark.parametrize("extension", [".md", ".csv", ".json"])
def test_load_text_file_supported_extensions(tmp_path, extension):
    """Test that .md, .csv, and .json files are accepted as plain text."""
    file_path = tmp_path / f"test{extension}"
    content = '{"key": "value", "list": [1, 2, 3]}'  # JSON content to ensure no parsing
    file_path.write_text(content, encoding="utf-8")

    results = load_text_file(str(file_path))

    assert len(results) == 1
    doc = results[0]
    assert doc.text == content
    assert doc.metadata[METADATA_SOURCE] == str(file_path)

def test_load_text_file_case_insensitive_extension(tmp_path):
    """Verify that uppercase suffixes like .TXT are accepted."""
    file_path = tmp_path / "test.TXT"
    content = "Case insensitive check."
    file_path.write_text(content, encoding="utf-8")

    results = load_text_file(str(file_path))
    assert len(results) == 1
    assert results[0].text == content

def test_load_text_file_not_found():
    """Test that missing files raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_text_file("non_existent_file_path_12345.txt")

def test_load_text_file_is_directory(tmp_path):
    """Test that directory input raises IsADirectoryError."""
    dir_path = tmp_path / "some_directory"
    dir_path.mkdir()
    with pytest.raises(IsADirectoryError):
        load_text_file(str(dir_path))

def test_load_text_file_unsupported_extension(tmp_path):
    """Test that unsupported extensions raise ValueError."""
    file_path = tmp_path / "test.pdf"  # .pdf is not in SUPPORTED_TEXT_EXTENSIONS
    file_path.write_text("dummy content", encoding="utf-8")
    
    with pytest.raises(ValueError, match="Unsupported file extension"):
        load_text_file(str(file_path))

def test_load_text_file_invalid_encoding(tmp_path):
    """Test that invalid UTF-8 results in a chained ValueError."""
    file_path = tmp_path / "bad_utf8.txt"
    # Write some bytes that are not valid UTF-8
    with open(file_path, "wb") as f:
        f.write(b, "\xff\xfe\xfd")

    with pytest.raises(ValueError) as excinfo:
        load_text_file(str(file_path))
    # The error is caught and re-raised as a ValueError with a specific message
    assert "Failed to decode" in str(excinfo.value)

def test_load_text_file_empty_after_normalization(tmp_path):
    """Test that empty or whitespace-only files raise ValueError."""
    file_path = tmp_path / "empty.txt"
    file_path.write_text("   \n   ", encoding="utf-8")
    
    with pytest.raises(ValueError, match="empty or only whitespace"):
        load_text_file(str(file_path))

def test_load_text_file_distinct_metadata_dicts(tmp_path):
    """Verify that every call returns a fresh metadata dictionary."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("content", encoding="utf-8")

    res1 = load_text_file(str(file_path))
    res2 = load_text_file(str(file_path))

    # Ensure they are different dictionary objects in memory
    assert res1[0].metadata is not res2[0].metadata