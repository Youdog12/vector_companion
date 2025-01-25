import unittest

# Discover and run all tests in the tests directory
if __name__ == "__main__":
    loader = unittest.TestLoader()
    tests = loader.discover(start_dir="tests", pattern="test_*.py")
    testRunner = unittest.TextTestRunner()
    testRunner.run(tests)
