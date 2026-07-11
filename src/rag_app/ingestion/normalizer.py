"""
This module provides text normalization utilities for the ingestion pipeline.
It ensures that raw text from various sources is cleaned and formatted consistently
before it is passed to the chunking process.
"""

import re


def normalize_text(text: str) -> str:
    """
    Normalizes a string by standardizing line endings, collapsing horizontal 
    whitespace, and merging excessive newlines while preserving structure.

    Args:
        text (str): The raw input text to normalize.

    Returns:
        str: The normalized text.

    Raises:
        TypeError: If the input is not a string.
    """
    if not isinstance(text, str):
        raise TypeError(f"Input must be of type 'str', but got {type(text).__name__}")

    # 1. Standardize line endings (Windows \r\n and old-Mac \r to \n)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Process line by line: strip trailing whitespace and collapse internal horizontal whitespace
    lines = text.split("\n")
    processed_lines = []
    for line in lines:
        # Remove trailing spaces/tabs
        line = line.rstrip()
        # Collapse runs of spaces and tabs into a single space
        line = re.sub(r"[ \t]+", " ", line)
        processed_lines.append(line)

    # 3. Join lines back together
    result = "\n".join(processed_lines)

    # 4. Collapse runs of three or more consecutive newlines into exactly two
    result = re.sub(r"\n{3,}", "\n\n", result)

    # 5. Final trim of leading/trailing whitespace
    return result.strip()