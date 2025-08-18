import unittest
from ..test_utils import unsloth_available, skip_if_no_unsloth


class TestTransformerModels(unittest.TestCase):
    """Test cases for transformer models in the forecaster module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Only import modules if unsloth is available
        if unsloth_available():
            from convokit.forecaster import TransformerDecoderModel
            self.TransformerDecoderModel = TransformerDecoderModel
    
    @skip_if_no_unsloth
    def test_transformer_decoder_model_import(self):
        """Test that TransformerDecoderModel can be imported when unsloth is available."""
        from convokit.forecaster import TransformerDecoderModel
        self.assertTrue(TransformerDecoderModel is not None)
    
    def test_transformer_decoder_model_import_without_unsloth(self):
        """Test that TransformerDecoderModel import doesn't fail when unsloth is not available."""
        try:
            from convokit.forecaster import TransformerDecoderModel
            # If we get here, the import succeeded (either with or without unsloth)
            self.assertTrue(True)
        except ImportError as e:
            # If unsloth is not available, we should get a specific error message
            if "not currently installed" in str(e) or "Unsloth GPU requirement not met" in str(e):
                self.assertTrue(True)  # Expected behavior
            else:
                raise  # Unexpected error


if __name__ == '__main__':
    unittest.main()
