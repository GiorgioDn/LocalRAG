import pytest
from typing import List
from unittest.mock import MagicMock
from src.rag_app.models.ollama_embeddings import OllamaEmbeddingClient

class MockOllamaResponse:
    """Simulates the structure of an ollama.Client.embeddings response."""
    def __init__(self, embeddings: List[list[float]]):
        self.embeddings = embeddings

class MockOllamaClient:
    """Mock implementation of ollama.Client for unit testing."""
    def __init__(self):
        self.embeddings_called = 0
        self.last_prompt = None
        self.last_model = None

    def embeddings(self, model: str, prompt: List[str]) -> MockOllamaResponse:
        self.embeddings_called += 1
        self.last_model = model
        self.last_prompt = prompt
        # Usually returns a list of objects with an .embedding attribute or 
        # works as a direct list/dict depending on the library version.
        # The code under test handles both cases.
        return MockOllamaResponse(prompt) # Simple mapping for basic success

class MockEmbeddingObject:
    """Simulates the object structure some versions of ollama SDK use."""
    def __init__(self, embedding_data: list[float]):
        self.embedding = embedding_data

# Overriding the simple mock to handle nested objects if necessary 
# (mimicking what the code expects in its extract logic)
class DetailedMockOllamaResponse:
    def __init__(self, embeddings_list: List[list[float]]):
        # Create a list of objects that have an .embedding attribute
        self.embeddings = [type('Obj', (object,), {'embedding': e}) for e in embeddings_list]

@pytest.fixture
def mock_client():
    return MockOllamaClient()

@pytest.fixture
def client(mock_client):
    """Provides an OllamaEmbeddingClient with a mocked client."""
    return OllamaEmbeddingClient(client=mock_client)

# --- Success Cases ---

def test_embed_texts_success_batch(client, mock_client):
    """Test that embed_texts calls the API once for a batch and returns nested list of floats."""
    texts = ["alpha", "beta"]
    results = client.embed_texts(texts)
    
    assert mock_client.embeddings_called == 1
    assert mock_client.last_model is not None
    assert mock_client.last_prompt == texts
    assert len(results) == 2
    assert isinstance(results[0], list)
    assert all(isinstance(x, float) for x in results[0])

def test_embed_texts_empty_list(client, mock_client):
    """Test that an empty list of texts returns an empty list and makes zero calls."""
    results = client.embed_texts([])
    assert results == []
    assert mock_client.embeddings_called == 0

def test_embed_query_success(client, mock_client):
    """Test that embed_query behaves like a single-item batch."""
    query = "alpha"
    result = client.embed_query(query)
    
    assert mock_client.embeddings_called == 1
    assert mock_client.last_prompt == [query]
    assert len(result) > 0
    assert isinstance(result, list)

# --- Validation Cases ---

@pytest.mark.parametrize("invalid_input", ["", "   ", "\n"])
def test_embed_texts_invalid_inputs(client, mock_client, invalid_input):
    """Test that empty or whitespace strings in a batch raise ValueError before calling client."""
    with pytest.raises(ValueError, match="non-empty string"):
        client.embed_texts([invalid_input])
    assert mock_client.embeddings_called == 0

@pytest.mark.parametrize("invalid_query", ["", "   ", "\n"])
def test_embed_query_invalid_inputs(client, mock_client, invalid_query):
    """Test that an empty or whitespace query raises ValueError before calling client."""
    with pytest.raises(ValueError, match="non-empty string"):
        client.embed_query(invalid_query)
    assert mock_client.embeddings_called == 0

# --- Error Handling & Malformed Responses ---

def test_embed_texts_mismatched_count(client, mock_client):
    """Test that a mismatch between requested and received embeddings raises RuntimeError."""
    # Force the mock to return only one result for two inputs
    mock_client.embeddings = lambda model, prompt: MockOllamaResponse([[1.0, 2.0]])
    
    with pytest.raises(RuntimeError, match="Expected 2 embeddings but received 1"):
        client.embed_texts(["alpha", "beta"])

def test_embed_texts_malformed_structure(client, mock_client):
    """Test that a missing 'embeddings' attribute/field results in an error."""
    # Mock a response object that doesn't have the .embeddings attribute
    mock_client.embeddings = lambda model, prompt: object() 
    
    with pytest.raises(RuntimeError, match="Embedding failed"):
        client.embed_texts(["alpha"])

def test_embed_texts_non_numeric_values(client, mock_client):
    """Test that non-numeric values in the response are caught/handled."""
    # Mock a response where one value is a string instead of a float
    mock_client.embeddings = lambda model, prompt: MockOllamaResponse([[1.0, "not_a_float"]])
    
    with pytest.raises(RuntimeError, match="Embedding failed"):
        client.embed_texts(["alpha"])

def test_client_exception_wrapped_in_runtime_error(client, mock_client):
    """Test that exceptions from the underlying client are wrapped in a RuntimeError."""
    mock_client.embeddings = lambda model, prompt: exec('raise Exception("Low level error")')
    
    with pytest.raises(RuntimeError) as excinfo:
        client.embed_texts(["alpha"])
    assert "Embedding failed" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, Exception)