import unittest
from visolex.lexnorm.basic_normalizer import BasicNormalizer

class TestBasicNormalizer(unittest.TestCase):
    def setUp(self):
        """Set up the BasicNormalizer instance for tests."""
        self.normalizer = BasicNormalizer()
    
    def test_basic_normalization_with_lowercase(self):
        """Test normalization with the lowercase option."""
        input_str = "HELLO 🌟 WORLD! This is a TEST."
        expected_output = "hello 🌟 world ! this is a test ."
        result = self.normalizer.basic_normalizer(input_str, lowercase=True)
        self.assertEqual(result, expected_output)

    def test_basic_normalization_without_lowercase(self):
        """Test normalization without the lowercase option."""
        input_str = "HELLO 🌟 WORLD! This is a TEST."
        expected_output = "HELLO 🌟 WORLD ! This is a TEST ."
        result = self.normalizer.basic_normalizer(input_str, lowercase=False)
        self.assertEqual(result, expected_output)

    def test_tone_normalization(self):
        """Test tone normalization functionality."""
        input_str = "Đây là chữ caí tíêng Việt! 🌟"
        expected_output = "đây là chữ cái tiếng việt ! 🌟"
        result = self.normalizer.basic_normalizer(input_str, lowercase=True)
        self.assertEqual(result, expected_output)
    
    def test_emoji_handling(self):
        """Test emoji splitting and handling."""
        input_str = "Hello😊World!"
        expected_output = "Hello 😊 World !"
        result = self.normalizer.basic_normalizer(input_str, lowercase=False)
        self.assertEqual(result, expected_output)

    def test_empty_string(self):
        """Test normalization with an empty string."""
        input_str = ""
        expected_output = ""
        result = self.normalizer.basic_normalizer(input_str, lowercase=True)
        self.assertEqual(result, expected_output)

    def test_mixed_content(self):
        """Test normalization with mixed content."""
        input_str = "Test123 😊...ABC🌟XYZ!"
        expected_output = "Test123 😊 ... ABC 🌟 XYZ !"
        result = self.normalizer.basic_normalizer(input_str, lowercase=False)
        self.assertEqual(result, expected_output)

    def test_edge_case_only_emojis(self):
        """Test normalization with input containing only emojis."""
        input_str = "🌟😊🌈"
        expected_output = "🌟 😊 🌈"
        result = self.normalizer.basic_normalizer(input_str, lowercase=False)
        self.assertEqual(result, expected_output)

if __name__ == "__main__":
    unittest.main()
