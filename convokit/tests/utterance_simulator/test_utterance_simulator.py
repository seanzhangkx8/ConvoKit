import unittest
import sys
from unittest import skipIf
from ..test_utils import unsloth_available, skip_if_no_unsloth


class TestUtteranceSimulator(unittest.TestCase):
    """Test cases for utterance simulator functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Only import modules if unsloth is available
        if unsloth_available():
            from convokit.utterance_simulator import UtteranceSimulator
            from convokit.utterance_simulator.utteranceSimulatorModel import UtteranceSimulatorModel
            from convokit.utterance_simulator.unslothUtteranceSimulatorModel import (
                UnslothUtteranceSimulatorModel,
            )

            self.UtteranceSimulator = UtteranceSimulator
            self.UtteranceSimulatorModel = UtteranceSimulatorModel
            self.UnslothUtteranceSimulatorModel = UnslothUtteranceSimulatorModel

    @skip_if_no_unsloth
    def test_utterance_simulator_import(self):
        """Test that utterance simulator can be imported when unsloth is available."""
        from convokit.utterance_simulator import UtteranceSimulator
        from convokit.utterance_simulator.utteranceSimulatorModel import UtteranceSimulatorModel
        from convokit.utterance_simulator.unslothUtteranceSimulatorModel import (
            UnslothUtteranceSimulatorModel,
        )

        # Basic import test
        self.assertTrue(UtteranceSimulator is not None)
        self.assertTrue(UtteranceSimulatorModel is not None)
        self.assertTrue(UnslothUtteranceSimulatorModel is not None)

    @skip_if_no_unsloth
    def test_unsloth_model_initialization(self):
        """Test that UnslothUtteranceSimulatorModel can be initialized."""
        from convokit.utterance_simulator.unslothUtteranceSimulatorModel import (
            UnslothUtteranceSimulatorModel,
        )

        # Test basic initialization with default parameters
        model = UnslothUtteranceSimulatorModel()
        self.assertIsInstance(model, UnslothUtteranceSimulatorModel)

    def test_utterance_simulator_import_without_unsloth(self):
        """Test that utterance simulator import doesn't fail when unsloth is not available."""
        # This test should pass even when unsloth is not available
        # because the import is handled gracefully in the __init__.py
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


class TestPivotalFramework(unittest.TestCase):
    """Test cases for pivotal framework that depends on utterance simulator."""

    def setUp(self):
        """Set up test fixtures."""
        # Only import modules if unsloth is available
        if unsloth_available():
            from convokit.pivotal_framework import PivotalMomentMeasure

            self.PivotalMomentMeasure = PivotalMomentMeasure

    @skip_if_no_unsloth
    def test_pivotal_framework_import(self):
        """Test that pivotal framework can be imported when unsloth is available."""
        from convokit.pivotal_framework import PivotalMomentMeasure

        self.assertTrue(PivotalMomentMeasure is not None)

    def test_pivotal_framework_import_without_unsloth(self):
        """Test that pivotal framework import doesn't fail when unsloth is not available."""
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


if __name__ == "__main__":
    unittest.main()
