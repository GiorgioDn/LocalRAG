import unittest
import logging
from src.rag_app.utils.logging_utils import configure_logging, get_logger

class TestLoggingUtils(unittest.TestCase):
    def setUp(self):
        # Clear handlers of the 'rag_app' logger before each test to ensure isolation
        logger = logging.getLogger("rag_app")
        logger.handlers = []

    def test_configure_logging_initial_setup(self):
        """Test that configure_logging sets up one handler for the rag_app logger."""
        configure_logging()
        logger = logging.getLogger("rag_app")
        self.assertEqual(len(logger.handlers), 1)
        self.assertFalse(logger.propagate)

    def test_configure_logging_idempotency(self):
        """Test that consecutive calls without force do not add duplicate handlers."""
        configure_logging()
        configure_logging()
        logger = logging.getLogger("rag_app")
        self.assertEqual(len(logger.handlers), 1)

    def test_configure_logging_force(self):
        """Test that force=True resets the handlers."""
        configure_logging()
        # Manually add an extra handler to simulate a dirty state
        logger = logging.getLogger("rag_app")
        logger.addHandler(logging.NullHandler()) 
        self.assertEqual(len(logger.handlers), 2)

        configure_logging(force=True)
        self.assertEqual(len(logger.handlers), 1)

    def test_configure_logging_levels(self):
        """Test that both numeric and string levels are accepted."""
        # Test integer
        configure_logging(level=logging.DEBUG)
        self.assertEqual(logging.getLogger("rag_app").level, logging.DEBUG)
        
        # Test string (case-insensitive)
        configure_logging(level="info")
        self.assertEqual(logging.getLogger("rag_app").level, logging.INFO)
        
        configure_logging(level="CRITICAL")
        self.assertEqual(logging.getLogger("rag_app").level, logging.CRITICAL)

    def test_configure_logging_invalid_level(self):
        """Test that invalid level strings raise ValueError."""
        with self.assertRaises(ValueError):
            configure_logging(level="NOT_A_LEVEL")

    def test_get_logger_standard(self):
        """Test that standard names are prefixed with 'rag_app'."""
        logger = get_logger("ingestion.text_loader")
        self.assertEqual(logger.name, "rag_app.ingestion.text_loader")

    def test_get_logger_with_prefix(self):
        """Test that names already containing 'rag_app' are not double-prefixed."""
        logger = get_logger("rag_app.models.ollama_chat")
        self.assertEqual(logger.name, "rag_app.models.ollama_chat")

    def test_get_logger_blank_name(self):
        """Test that blank or whitespace names raise ValueError."""
        with self.assertRaises(ValueError):
            get_logger("")
        with self.assertRaises(ValueError):
            get_logger("   ")

if __name__ == "__main__":
    unittest.main()