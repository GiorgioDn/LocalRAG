from pathlib import Path
from collections.abc import Collection
from typing import Union

def as_file_path(path: str | Path) -> Path:
    """
    Converts a path string or Path object into a pathlib.Path instance.
    Only accepts str or Path; does not resolve the path.
    """
    if not isinstance(path, (str, Path)):
        raise TypeError(f"Expected path to be of type str or Path, got {type(path).__name__}")
    return Path(path)

def require_existing_file(path: str | Path) -> Path:
    """
    Ensures the provided path exists and is a regular file.
    """
    path_obj = as_file_path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"The file at {path_obj} was not found.")
    if path_obj.is_dir():
        raise IsADirectoryError(f"The path {path_obj} is a directory, not a file.")
    return path_obj

def has_supported_extension(
    file_path: str | Path,
    supported_extensions: Collection[str],
) -> bool:
    """
    Checks if the provided path's extension is in the supported collection.
    Supports extensions with or without a leading dot (e.g., '.txt' and 'txt').
    """
    path_obj = as_file_path(file_path)
    suffix = path_obj.suffix.lower()
    if not suffix:
        return False

    # Check if the extension is in the collection exactly or without the leading dot
    if suffix in supported_extensions:
        return True
    
    stripped_suffix = suffix.lstrip('.')
    if stripped_suffix and stripped_suffix in supported_extensions:
        return True

    return False

def require_supported_extension(
    file_path: str | Path,
    supported_extensions: Collection[str],
) -> Path:
    """
    Validates the extension of a path against a collection and returns the Path.
    Raises ValueError if the extension is not supported.
    """
    path_obj = as_file_path(file_path)
    if not has_supported_extension(path_obj, supported_extensions):
        raise ValueError(
            f"Unsupported file extension '{path_obj.suffix}'. "
            f"Supported extensions: {supported_extensions}"
        )
    return path_obj