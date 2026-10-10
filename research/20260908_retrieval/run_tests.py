"""Run all experiment protocol tests, including GPU-library model checks."""

import sys
import unittest
from pathlib import Path

root = Path(__file__).resolve().parent
suite = unittest.defaultTestLoader.discover(str(root), pattern='test_*.py')
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
