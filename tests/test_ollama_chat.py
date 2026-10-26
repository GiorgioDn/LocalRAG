import pytest
from typing import Any
from src.rag_app.models.ollama_chat import OllamaChatClient

class MockOllamaResponse:
    """Simulates an ollama.ChatResponse object."""
    def __init__(self, message_content: str):
        # Simulate the nested structure: response.message.content
        self.message = type('Message', (object,), {'content': message_content})

class MockOllamaClient:
    """Mock implementation of ollama.Client for unit testing."""
    def __init__(self):
        self.chat_called = 0
        self.last_model = None
        self.last_messages = None
        self.last_stream = None

    def chat(self, model: str, messages: list[dict], stream: bool = False) -> MockOllamaResponse:
        self.chat_called += 1
        self.last_model = model
        self.last_messages = messages
        self.last_stream = stream
        # Default to a success response for the basic mock
        return MockOllamaResponse("some content")

@pytest.fixture
def mock_client():
    """Provides a mock Ollama client."""
    return MockOllamaClient()

@pytest.fixture
def chat_client(mock_client):
    """Provides an OllamaChatClient with the mocked client."""
    return OllamaChatClient(client=mock_client)

# --- Success Cases ---

def test_answer_success(chat_client, mock_client):
    """Test that answer() correctly interacts with the chat API and returns a stripped result."""
    prompt = "Hello"
    response_text = "  Some\n\ncontent   "
    
    def custom_chat(model, messages, stream=False):
        mock_client.chat_called += 1
        mock_client.last_model = model
        mock_client.last_messages = messages
        mock_client.last_stream = stream
        return MockOllamaResponse(response_text)

    mock_client.chat = custom_chat
    
    result = chat_client.answer(prompt)

    assert mock_client.chat_called == 1
    assert mock_client.last_model is not None  # Ensures a model was selected
    assert mock_client.last_stream is False
    assert mock_client.last_messages == [{"role": "user", "content": prompt}]
    assert result == "Some\n\ncontent"

# --- Validation Cases ---

@pytest.mark.parametrize("invalid_prompt", ["", "   ", "\n"])
def test_answer_validation_fails(chat_client, mock_client, invalid_prompt):
    """Test that empty or whitespace-only prompts raise ValueError before any network calls."""
    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        chat_client.answer(invalid_prompt)
    assert mock_client.chat_called == 0

# --- Malformed Response Cases ---

def test_answer_missing_message_error(chat_client, mock_client):
    """Test that a missing message object in the response results in a RuntimeError."""
    class BrokenResponse:
        pass # No .message attribute
    
    mock_client.chat = lambda model, messages, stream=False: BrokenResponse()
    
    with pytest.raises(RuntimeError, match="Ollama returned an empty response"):
        chat_client.answer("Hello")

def test_answer_empty_content_error(chat_client, mock_client):
    """Test that a message with no content results in a RuntimeError."""
    # Simulate a response where .message exists but .content is None or empty
    class EmptyContentResponse:
        def __init__(self):
            self.message = type('Msg', (object,), {'content': ""})
    
    mock_client.chat = lambda model, messages, stream=False: EmptyContentResponse()
    
    with pytest.raises(RuntimeError, match="Ollama returned an empty response"):
        chat_client.answer("Hello")

def test_answer_non_string_content_error(chat_client, mock_client):
    """Test that non-string content results in a RuntimeError (or standard Python error caught as RuntimeError)."""
    class BadTypeResponse:
        def __init__(self):
            self.message = type('Msg', (object,), {'content': 12345}) # Not a string
    
    mock_client.chat = lambda model, messages, stream=False: BadTypeResponse()
    
    with pytest.raises(RuntimeError):
        # Depending on whether .strip() is called on the result or logic handles types. 
        # In current code, if content isn't a string, .strip() will fail and be caught by 'except Exception'.
        chat_client.answer("Hello")

# --- Client Failure Cases ---

def test_chat_client_exception_wrapped(chat_client, mock_client):
    """Test that exceptions from the underlying client are wrapped in a RuntimeError."""
    mock_client.chat = lambda model, messages, stream=False: exec('raise ConnectionError("Network down")')
    
    with pytest.raises(RuntimeError) as excinfo:
        chat_client.answer("Hello")
    assert "Failed to retrieve response from Ollama" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ConnectionError)