import unittest
import sys
import importlib
from ..test_utils import unsloth_available, skip_if_no_unsloth


class TestConditionalImports(unittest.TestCase):
    """Test conditional import handling without triggering problematic imports during discovery."""

    def test_unsloth_availability_check(self):
        """Test that unsloth availability check works correctly."""
        # This test should always pass
        available = unsloth_available()
        self.assertIsInstance(available, bool)

    def test_skip_decorator_works(self):
        """Test that the skip decorator works correctly."""
        # This test should always pass
        decorator = skip_if_no_unsloth("Test reason")
        self.assertTrue(callable(decorator))

    def test_core_convokit_import(self):
        """Test that core convokit can be imported."""
        try:
            import convokit

            self.assertTrue(True)  # Import succeeded
        except Exception as e:
            self.fail(f"Core convokit import failed: {e}")

    def test_utterance_simulator_module_exists(self):
        """Test that utterance simulator module exists but don't import it."""
        # Just check if the module path exists without importing
        import os

        module_path = os.path.join(os.path.dirname(__file__), "..", "..", "utterance_simulator")
        self.assertTrue(os.path.exists(module_path), "utterance_simulator module should exist")

    def test_forecaster_module_exists(self):
        """Test that forecaster module exists but don't import it."""
        # Just check if the module path exists without importing
        import os

        module_path = os.path.join(os.path.dirname(__file__), "..", "..", "forecaster")
        self.assertTrue(os.path.exists(module_path), "forecaster module should exist")

    @skip_if_no_unsloth("Unsloth not available")
    def test_conditional_import_with_unsloth(self):
        """Test conditional import when unsloth is available."""
        # This test will only run if unsloth is available
        try:
            from convokit.utterance_simulator import UtteranceSimulator

            self.assertTrue(True)  # Import succeeded
        except Exception as e:
            self.fail(f"Import failed when unsloth should be available: {e}")

    def test_conditional_import_without_unsloth(self):
        """Test conditional import when unsloth is not available."""
        # This test will always run
        if not unsloth_available():
            # If unsloth is not available, the import should fail gracefully
            try:
                from convokit.utterance_simulator import UtteranceSimulator

                # If we get here, the import succeeded (unsloth is available)
                self.assertTrue(True)
            except (ImportError, ModuleNotFoundError) as e:
                # Expected behavior when unsloth is not available
                error_str = str(e)
                if (
                    "not currently installed" in error_str
                    or "Unsloth GPU requirement not met" in error_str
                    or "unsloth is currently not available" in error_str
                ):
                    self.assertTrue(True)  # Expected behavior
                else:
                    raise  # Unexpected error
        else:
            # If unsloth is available, the import should succeed
            try:
                from convokit.utterance_simulator import UtteranceSimulator

                self.assertTrue(True)  # Import succeeded
            except Exception as e:
                self.fail(f"Import failed when unsloth should be available: {e}")


if __name__ == "__main__":
    unittest.main()
