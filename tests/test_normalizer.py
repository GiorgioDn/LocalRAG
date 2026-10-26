import unittest
from src.rag_app.ingestion.normalizer import normalize_text

class TestNormalizer(unittest.TestCase):
    def test_line_endings(self):
        # Windows line endings
        self.assertEqual(normalize_text("Line1\r\nLine2"), "Line1\nLine2")
        # Old-Mac line endings
        self.assertEqual(normalize_text("Line1\rLine2"), "Line1\nLine2")

    def test_trailing_whitespace(self):
        # Trailing spaces and tabs are removed from individual lines
        self.assertEqual(normalize_text("Line1   \nLine2\t"), "Line1\nLine2")
        self.assertEqual(normalize_text("Line1   \nLine2\t\nLine3"), "Line1\nLine2\nLine3")

    def test_horizontal_whitespace(self):
        # Repeated spaces and tabs within a line become one space
        self.assertEqual(normalize_text("Line1    with   many    spaces"), "Line1 with many spaces")
        self.assertEqual(normalize_text("Line1\t\t\twith  tabs"), "Line1 with tabs")

    def test_excessive_newlines(self):
        # Three or more consecutive newlines become exactly two
        self.assertEqual(normalize_text("Para1\n\n\n\nPara2"), "Para1\n\nPara2")

    def test_outer_whitespace(self):
        # Leading and trailing overall whitespace is removed
        self.assertEqual(normalize_text("   TrimMe   "), "TrimMe")
        self.assertEqual(normalize_text("\n\nNewlines\n\n"), "Newlines")

    def test_whitespace_only_input(self):
        # Whitespace-only input returns ""
        self.assertEqual(normalize_text("   \t\n  "), "")

    def test_type_errors(self):
        # Non-string inputs raise TypeError
        with self.assertRaises(TypeError):
            normalize_text(None)
        with self.assertRaises(TypeError):
            normalize_text(123)
        with self.assertRaises(TypeError):
            normalize_text(b"bytes")

    def test_determinism(self):
        # Input string is not mutated and repeated calls return identical output
        original = "Title\r\n\r\n- Item 1\t\n\n\n\nText"
        expected = "Title\n\n- Item 1\n\nText"
        self.assertEqual(normalize_text(original), expected)
        self.assertEqual(normalize_text(original), expected)

    def test_structure_preservation(self):
        # Ensure headers and bullet points are preserved (no lowercasing, etc.)
        input_text = "## Header\n- List item 1\n* Item 2\n1. Numbered\n\nNext Paragraph"
        expected = "## Header\n- List item 1\n* Item 2\n1. Numbered\n\nNext Paragraph"
        self.assertEqual(normalize_text(input_text), expected)

if __name__ == "__main__":
    unittest.main()