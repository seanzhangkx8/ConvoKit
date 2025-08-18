from unittest import TestLoader, TextTestRunner
import sys
import os


def run_tests():
    """Run all tests with conditional execution based on available dependencies."""

    # Add the parent directory to the path so we can import test_utils
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    loader = TestLoader()
    tests = loader.discover(".")
    testRunner = TextTestRunner(verbosity=2)
    test_results = testRunner.run(tests)

    if test_results.wasSuccessful():
        print("\nAll tests passed!")
        exit(0)
    else:
        print(f"\n{test_results.failures} failures, {test_results.errors} errors")
        exit(1)


if __name__ == "__main__":
    run_tests()
