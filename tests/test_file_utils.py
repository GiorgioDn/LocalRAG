import unittest
from pathlib import Path
from rag_app.utils.file_utils import (
    as_file_path,
    require_existing_file,
    has_supported_extension,
    require_supported_extension,
)

class TestFileUtils(unittest.TestCase):
    def test_as_file_path_valid_types(self):
        """Test that as_file_path accepts both string and Path objects."""
        path_str = "some/path/to/file.txt"
        path_obj = Path("some/path/to/file.txt")
        
        self.assertIsInstance(as_file_path(path_str), Path)
        self.assertIsInstance(as_file_path(path_obj), Path)

    def test_as_file_path_invalid_types(self):
        """Test that as_file_path raises TypeError for unsupported types."""
        with self.assertRaises(TypeError):
            as_file_path(123)
        with self.assertRaises(TypeError):
            as_file_path(None)

    def test_require_existing_file_success(self):
        """Test that require_existing_file works for a valid file."""
        # Use a temporary file or an existing one if possible, 
        # but since we aren't writing files here, we rely on the logic check.
        temp_path = Path("test_dummy_file.txt")
        temp_path.touch()
        try:
            result = require_existing_file(temp_path)
            self.assertEqual(result, temp_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_require_existing_file_not_found(self):
        """Test that require_existing_file raises FileNotFoundError for missing files."""
        with self.assertRaises(FileNotFoundError):
            require_existing_file("non_existent_file_999.txt")

    def test_require_existing_file_is_directory(self):
        """Test that require_existing_file raises IsADirectoryError for directories."""
        # Use the current directory as a placeholder for a non-file path
        with self.assertRaises(IsADirectoryError):
            require_existing_file(".")

    def test_has_supported_extension_cases(self):
        """Test that has_supported_extension handles various cases correctly."""
        extensions = [".txt", "pdf", ".jpg"]
        
        # Normal matches
        self.assertTrue(has_supported_extension("file.txt", extensions))
        self.assertTrue(has_supported_extension("image.JPG", extensions)) # Case insensitivity
        self.assertTrue(has_supported_extension("doc.pdf", extensions))
        
        # Matches without leading dots in the list but with . prefix from path
        # (Actually, our implementation checks if ".txt" is in ["..txt"] or "txt" is in ["..txt"])
        self.assertTrue(has_supported_extension("file.TXT", [".TXT"])) # Exact match case-insensitive 
        
        # No suffix
        self.assertFalse(has_supported_extension("no_suffix_file", extensions))

    def test_has_supported_extension_mixed_dots(self):
        """Test that it handles both '.txt' and 'txt' as valid representations in the collection."""
        # Testing "txt" vs ".txt" logic
        exts = [".txt"]
        self.assertTrue(has_supported_extension("file.txt", exts))
        
        exts_no_dot = ["txt"]
        self.assertTrue(has_supported_extension("file.txt", exts_no_dot))

    def test_require_supported_extension_success(self):
        """Test that require_supported_extension returns a Path for valid extensions."""
        extensions = [".pdf", ".txt"]
        path_obj = require_supported_extension("valid.pdf", extensions)
        self.assertIsInstance(path_obj, Path)
        self.assertEqual(path_obj.name, "valid.pdf")

    def test_require_supported_extension_failure(self):
        """Test that require_supported_extension raises ValueError for invalid extensions."""
        extensions = [".pdf", ".txt"]
        with self.assertRaises(ValueError) as cm:
            require_supported_extension("invalid.exe", extensions)
        self.assertIn("Unsupported file extension", str(cm.exception))

if __name__ == "__main__":
    unittest.main()