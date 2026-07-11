"""
This module contains static project-wide constants. 
Values here are fixed application conventions and should not be replaced by 
environment-dependent configurations found in config.py.
"""

from typing import Final


# Source Type Constants
SOURCE_TYPE_TEXT: Final[str] = "text"
SOURCE_TYPE_PDF: Final[str] = "pdf"

# Supported File Extensions
SUPPORTED_TEXT_EXTENSIONS: Final[frozenset[str]] = frozenset({
    ".txt",
    ".md",
    ".csv",
    ".json",
})
SUPPORTED_PDF_EXTENSIONS: Final[frozenset[str]] = frozenset({".pdf"})

# Metadata Key Constants
METADATA_SOURCE: Final[str] = "source"
METADATA_SOURCE_TYPE: Final[str] = "source_type"
METADATA_PAGE: Final[str] = "page"
METADATA_CHUNK_INDEX: Final[str] = "chunk_index"

# Prompt Template Keys
PROMPT_KEY_QUESTION: Final[str] = "question"
PROMPT_KEY_CONTEXT: Final[str] = "context"