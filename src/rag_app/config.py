import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    """
    Configuration settings for the local-first RAG application.

    This class defines the core configuration parameters of the system. 
    It is designed to be immutable and validates all inputs upon initialization.
    """
    ollama_base_url: str
    ollama_chat_model: str
    ollama_embed_model: str
    chroma_persist_dir: str
    default_collection: str
    chunk_size: int
    chunk_overlap: int
    top_k: int

    def __post_init__(self) -> None:
        """
        Validate configuration values after initialization.
        Raises ValueError if any value fails the defined constraints.
        """
        # Validate that mandatory string fields are not empty or just whitespace
        string_fields = {
            "ollama_base_url": self.ollama_base_url,
            "ollama_chat_model": self.ollama_chat_model,
            "ollama_embed_model": self.ollama_embed_model,
            "chroma_persist_dir": self.chroma_persist_dir,
            "default_collection": self.default_collection,
        }
        for name, value in string_fields.items():
            if not str(value).strip():
                raise ValueError(f"Configuration '{name}' must not be empty after stripping whitespace.")

        # Validate numeric constraints
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be greater than 0 (received {self.chunk_size}).")
        
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be greater than or equal to 0 (received {self.chunk_overlap}).")
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than chunk_size ({self.chunk_size})."
            )

        if self.top_k <= 0:
            raise ValueError(f"top_k must be greater than 0 (received {self.top_k}).")


def _get_env_int(key: str, default: int) -> int:
    """
    Internal helper to fetch and cast environment variables to integers.
    
    Returns the provided default if the environment variable is missing or empty.
    Raises ValueError if the variable is present but not a valid integer.
    """
    value = os.getenv(key)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"Environment variable {key} must be an integer (received '{value}').")


# Instantiate the settings object for use throughout the application.
def get_default_embed_model():
    model = os.getenv("OLLAMA_EMBED_MODEL")
    return model if model and model.strip() != "" else "nomic-embed-text"

settings = Settings(
    ollama_base_url=os.getenv("OLLAMA_BASE_URL"),
    ollama_chat_model=os.getenv("OLLAMA_CHAT_MODEL"),
    ollama_embed_model=get_default_embed_model(),
    chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR"),
    default_collection=os.getenv("DEFAULT_COLLECTION"),
    chunk_size=_get_env_int("CHUNK_SIZE", 800),
    chunk_overlap=_get_env_int("CHUNK_OVERLAP", 120),
    top_k=_get_env_int("TOP_K", 5),
)

CHROMA_PATH = os.getenv("CHROMA_PATH", "storage/chroma")