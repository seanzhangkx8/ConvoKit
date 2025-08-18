import unittest
import sys
import os
from ..test_utils import unsloth_available, skip_if_no_unsloth


class TestLLMFunctionality(unittest.TestCase):
    """Test LLM functionality with conditional execution."""
    
    def test_unsloth_availability(self):
        """Test that unsloth availability check works."""
        available = unsloth_available()
        self.assertIsInstance(available, bool)
        print(f"Unsloth available: {available}")
    
    def test_skip_decorator(self):
        """Test that skip decorator works correctly."""
        decorator = skip_if_no_unsloth("Test reason")
        self.assertTrue(callable(decorator))
    
    @skip_if_no_unsloth("Unsloth not available")
    def test_llm_modules_when_available(self):
        """Test LLM modules when unsloth is available."""
        # This test will only run if unsloth is available
        try:
            # Test that we can import LLM-related modules
            from convokit.utterance_simulator import UtteranceSimulator
            from convokit.forecaster import TransformerDecoderModel
            from convokit.pivotal_framework import PivotalMomentMeasure
            
            self.assertTrue(True)  # All imports succeeded
        except Exception as e:
            self.fail(f"LLM imports failed when unsloth should be available: {e}")
    
    def test_llm_modules_when_not_available(self):
        """Test LLM modules when unsloth is not available."""
        # This test will always run
        if not unsloth_available():
            # Test that imports fail gracefully when unsloth is not available
            for module_name in ['convokit.utterance_simulator', 'convokit.forecaster', 'convokit.pivotal_framework']:
                try:
                    __import__(module_name)
                    # If we get here, the import succeeded (unsloth is available)
                    self.assertTrue(True)
                except (ImportError, ModuleNotFoundError) as e:
                    # Expected behavior when unsloth is not available
                    error_str = str(e)
                    if ("not currently installed" in error_str or 
                        "Unsloth GPU requirement not met" in error_str or
                        "unsloth is currently not available" in error_str):
                        self.assertTrue(True)  # Expected behavior
                    else:
                        raise  # Unexpected error
        else:
            # If unsloth is available, test that imports succeed
            try:
                from convokit.utterance_simulator import UtteranceSimulator
                from convokit.forecaster import TransformerDecoderModel
                from convokit.pivotal_framework import PivotalMomentMeasure
                self.assertTrue(True)  # All imports succeeded
            except Exception as e:
                self.fail(f"LLM imports failed when unsloth should be available: {e}")
    
    def test_core_functionality_always_works(self):
        """Test that core functionality always works regardless of unsloth."""
        # These imports should always work
        try:
            from convokit import Corpus, Utterance, Speaker
            from convokit.model import Conversation
            from convokit.transformer import Transformer
            
            self.assertTrue(True)  # Core imports succeeded
        except Exception as e:
            self.fail(f"Core functionality import failed: {e}")


if __name__ == '__main__':
    unittest.main()
