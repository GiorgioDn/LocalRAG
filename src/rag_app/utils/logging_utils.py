import logging
from typing import Union

def configure_logging(
    level: Union[str, int] = logging.INFO,
    *,
    force: bool = False,
) -> None:
    """
    Configures the 'rag_app' logger for the application.

    Args:
        level: The logging level as an integer (e.g., logging.INFO) or 
                a case-insensitive string (e.g., "DEBUG").
        force: If True, removes and replaces existing handlers.
    
    Raises:
        ValueError: If a non-integer string is provided that does not match a valid level.
    """
    # Resolve the numeric level
    if isinstance(level, str):
        numeric_level = getattr(logging, level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid logging level: {level}")
    else:
        numeric_level = level

    logger = logging.getLogger("rag_app")

    # Handle idempotency and forced reconfiguration
    if force or not logger.handlers:
        logger.handlers = []
    
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        date_format="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.setLevel(numeric_level)
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """
    Retrieves a logger named after the module, prefixed with 'rag_app'.

    Args:
        name: The name of the module/component.

    Raises:
        ValueError: If the provided name is empty or whitespace.
    """
    if not name or not name.strip():
        raise ValueError("Logger name cannot be empty.")
    
    if name.startswith("rag_app."):
        return logging.getLogger(name)
    return logging.getLogger(f"rag_app.{name}")