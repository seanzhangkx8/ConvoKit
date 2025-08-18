import unittest
import sys
import importlib

class TestImportHandling(unittest.TestCase):
    """Test that import handling works correctly with and without optional dependencies."""
    
    def test_convokit_import_without_unsloth(self):
        """Test that convokit can be imported even when unsloth is not available."""
        # This test should always pass, regardless of whether unsloth is installed
        try:
            import convokit
            self.assertTrue(True)  # Import succeeded
        except ImportError as e:
            # If there's an import error, it should be related to unsloth
            if "not currently installed" in str(e) or "Unsloth GPU requirement not met" in str(e):
                self.assertTrue(True)  # Expected behavior
            else:
                raise  # Unexpected error
    
    def test_utterance_simulator_import_handling(self):
        """Test that utterance simulator import is handled gracefully."""
        try:
            from convokit.utterance_simulator import UtteranceSimulator
            # If we get here, the import succeeded (either with or without unsloth)
            self.assertTrue(True)
        except ImportError as e:
            # If unsloth is not available, we should get a specific error message
            if "not currently installed" in str(e) or "Unsloth GPU requirement not met" in str(e):
                self.assertTrue(True)  # Expected behavior
            else:
                raise  # Unexpected error
    
    def test_pivotal_framework_import_handling(self):
        """Test that pivotal framework import is handled gracefully."""
        try:
            from convokit.pivotal_framework import PivotalMomentMeasure
            # If we get here, the import succeeded (either with or without unsloth)
            self.assertTrue(True)
        except ImportError as e:
            # If unsloth is not available, we should get a specific error message
            if "not currently installed" in str(e) or "Unsloth GPU requirement not met" in str(e):
                self.assertTrue(True)  # Expected behavior
            else:
                raise  # Unexpected error
    
    def test_forecaster_import_handling(self):
        """Test that forecaster import is handled gracefully."""
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
    
    def test_core_modules_always_available(self):
        """Test that core modules are always available regardless of unsloth."""
        # These modules should always be available
        core_modules = [
            'convokit.model',
            'convokit.util',
            'convokit.coordination',
            'convokit.politenessStrategies',
            'convokit.transformer',
            'convokit.convokitPipeline',
            'convokit.hyperconvo',
            'convokit.speakerConvoDiversity',
            'convokit.text_processing',
            'convokit.phrasing_motifs',
            'convokit.prompt_types',
            'convokit.classifier.classifier',
            'convokit.ranker',
            'convokit.fighting_words',
            'convokit.paired_prediction',
            'convokit.bag_of_words',
            'convokit.expected_context_framework',
            'convokit.surprise',
            'convokit.convokitConfig',
            'convokit.redirection',
        ]
        
        for module_name in core_modules:
            try:
                importlib.import_module(module_name)
                self.assertTrue(True)  # Import succeeded
            except ImportError as e:
                self.fail(f"Core module {module_name} should always be available: {e}")


if __name__ == '__main__':
    unittest.main()
