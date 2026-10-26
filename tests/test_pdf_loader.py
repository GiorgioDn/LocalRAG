import pytest
import fitz
from pathlib import Path
from typing import List
from unittest.mock import patch

from src.rag_app.ingestion.pdf_loader import load_pdf_file
from src.rag_app.constants import (
    METADATA_SOURCE,
    METADATA_SOURCE_TYPE,
    METADATA_PAGE,
)
from src.rag_app.types import LoadedDocument

@pytest.fixture
def temp_pdf_paths(tmp_path):
    """Creates temporary PDF files for testing."""
    # 1. Text on page 1, blank page 2, text on page 3
    mixed_pdf = tmp_path / "mixed_pages.pdf"
    doc1 = fitz.open()
    page1 = doc1.new_page()
    page1.insert_text((50, 50), "Content for Page 1")
    
    doc1.new_page() # Page 2 (blank)
    
    page3 = doc1.new_page()
    page3.insert_text((50, 50), "Content for Page 3")
    doc1.save(str(mixed_pdf))
    doc1.close()

    # 2. Only blank pages (no extractable text)
    blank_pdf = tmp_path / "blank_pages.pdf"
    doc2 = fitz.open()
    doc2.new_page()
    doc2.new_page()
    doc2.save(str(blank_pdf))
    doc2.close()

    # 3. Non-PDF file (e.g., a text file with .pdf extension or different content)
    invalid_ext_file = tmp_path / "not_a_pdf.txt"
    invalid_ext_file.write_text("This is not a PDF content.")

    return {
        "mixed": mixed_pdf,
        "blank": blank_pdf,
        "invalid_ext": invalid_ext_file,
        "missing": tmp_path / "non_existent.pdf"
    }

def test_load_pdf_file_filtered_pages(temp_pdf_paths):
    """
    Test 1 & 2: Verify that only pages with text are returned and 
    metadata is correctly preserved.
    """
    docs = load_pdf_file(str(temp_pdf_paths["mixed"]))
    
    # Should only have 2 documents (page 1 and page 3)
    assert len(docs) == 2
    
    # Check content and metadata for first doc (Page 1)
    assert "Content for Page 1" in docs[0].text
    assert docs[0].metadata[METADATA_PAGE] == 1
    assert docs[0].metadata[METADATA_SOURCE_TYPE] == "pdf" # Should match constant

    # Check content and metadata for second doc (Page 3)
    assert "Content for Page 3" in docs[1].text
    assert docs[1].metadata[METADATA_PAGE] == 3
    assert docs[1].metadata[METADATA_SOURCE_TYPE] == "pdf"

def test_load_pdf_no_extractable_text(temp_pdf_paths):
    """
    Test 4: Assert that a PDF with no extractable text raises the specific error.
    """
    with pytest.raises(ValueError, match="contains no extractable text"):
        load_pdf_file(str(temp_pdf_paths["blank"]))

def test_load_pdf_invalid_paths(temp_pdf_paths):
    """
    Test 5: Test non-existent files and invalid formats.
    """
    # Missing file (should trigger FileNotFoundError or the ValueError from open failure)
    with pytest.raises((ValueError, FileNotFoundError)):
        load_pdf_file(str(temp_pdf_paths["missing"]))

    # Check if it handles a file that isn't a valid PDF structure correctly 
    # Note: fitz.open() might raise its own exception which is caught and re-raised as ValueError in the loader.
    with pytest.raises(ValueError, match="Failed to open PDF|Unsupported file extension"):
        load_pdf_file(str(temp_pdf_paths["invalid_ext"]))
