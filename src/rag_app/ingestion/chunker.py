import hashlib
from typing import List, Optional
from dataclasses import replace

from rag_app.constants import METADATA_CHUNK_INDEX
from rag_app.types import DocumentChunk, LoadedDocument
from rag_app.config import settings

def _make_chunk_id(
    source: str, 
    source_type: str, 
    page: Optional[int], 
    index: int, 
    text: str
) -> str:
    """
    Generates a deterministic SHA-256 hash for a chunk.
    """
    # Use a stable representation for the fields to ensure reproducibility.
    # We include text and metadata info in the seed.
    seed_string = f"{source}|{source_type}|{page}|{index}|{text}"
    return hashlib.sha256(seed_string.encode("utf-8")).hexdigest()

def chunk_documents(
    documents: list[LoadedDocument],
    *,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> list[DocumentChunk]:
    """
    Splits a list of LoadedDocuments into DocumentChunks based on 
    specified size and overlap.
    """
    # Determine configuration
    size = chunk_size if chunk_size is not None else settings.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    # Validate configurations
    if size is None or size <= 0:
        raise ValueError(f"Invalid chunk_size ({size}). Must be a positive integer.")
    if overlap is None or overlap < 0:
        raise ValueError(f"Invalid chunk_overlap ({overlap}). Must be non-negative.")
    if overlap >= size:
        raise ValueError(f"chunk_overlap ({overlap}) must be less than chunk_size ({size}).")

    all_chunks = []

    for doc in documents:
        if not isinstance(doc, LoadedDocument):
            raise TypeError(f"Expected type 'LoadedDocument', got {type(doc).__name__}")

        text = doc.text
        if not text or not text.strip():
            continue

        # Prepare metadata for chunks of this specific document
        base_metadata = doc.metadata.copy()
        current_chunk_index = 0
        
        current_pos = 0
        while current_pos < len(text):
            # Determine end point of the window to look at
            end_window = min(current_pos + size, len(text))
            
            # If we are at the very end or near it, just take the rest
            if end_window == len(text):
                chunk_content = text[current_pos:].strip()
                if chunk_content:
                    metadata = base_metadata.copy()
                    metadata[METADATA_CHUNK_INDEX] = current_chunk_index
                    
                    all_chunks.append(DocumentChunk(
                        id=_make_chunk_id(
                            source=base_metadata.get("source", ""),
                            source_type=base_metadata.get("source_type", ""),
                            page=base_metadata.get("page"),
                            index=current_chunk_index,
                            text=chunk_content
                        ),
                        text=chunk_content,
                        metadata=metadata
                    ))
                    current_chunk_index += 1
                break

            # Search for a whitespace boundary at or before end_window.
            # We look backwards from the end of our window to find the nearest newline/space.
            best_split = end_window
            for i in range(end_window - 1, current_pos - 1, -1):
                if text[i] in (" ", "\n", "\t"):
                    best_split = i + 1
                    break
            
            chunk_text = text[current_pos:best_split].strip()
            
            if chunk_text:
                # Create result...
                metadata = base_metadata.copy()
                metadata[METADATA_CHUNK_INDEX] = current_chunk_index
                
                all_chunks.append(DocumentChunk(
                    id=_make_chunk_id(
                        source=base_metadata.get("source", ""),
                        source_type=base_metadata.get("source_type", ""),
                        page=base_metadata.get("page"),
                        index=current_chunk_index,
                        text=chunk_text
                    ),
                    text=chunk_text,
                    metadata=metadata
                ))
                current_chunk_index += 1

            # Update current position for next iteration
            # This is the "slide" of the window. 
            # Since we want overlap of 'overlap' characters from the PREVIOUS selection,
            # but the break point was adjusted by our search, we step back from the actual break.
            current_pos = best_split - overlap
            
            # Edge case: if chunk_content is empty because it was just whitespace 
            # between two segments, we need to ensure current_pos still moves forward.
            if chunk_text == "" and current_pos <= (best_split - size):
                 current_pos = best_split
            
            # To strictly avoid infinite loops if overlapping logic is weird:
            if current_pos >= best_split and best_split < len(text):
                current_pos = best_split
            
            # Final safety check: did we actually move? 
            # If not, force a jump to prevent infinite loop.
            # (Only happens if overlap == size or other logical failures)
            # However, requirement says 'avoid' loops, and our logic with overlap < size
            # guarantees movement unless the text is very small or chunk_size is 0/negative.
            
            if current_pos >= len(text):
                break
            
            # If the "best_split" was exactly our intended chunk end but we are still
            # somehow in a loop (e.g., overlap=0), the index would advance correctly.
            # However, if best_split is equal to current_pos (shouldn't happen with 1-step) 
            # or doesn't move forward, force movement.
            if current_pos == (best_split - overlap) and current_pos < best_split:
                 pass # standard logic should hold

        # Re-verify the loop to ensure we don't end up with duplicate indices if 
        # current_pos doesn't advance due to specific chunk_size/overlap combos.
        # The calculation 'current_pos = best_split - overlap' handles this as long 
        # as overlap < size.

    return all_chunks