"""Run the active stateless inference contract and model-client regressions."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.validation import INFERENCE_TEST_PATTERNS


def main():
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for pattern in INFERENCE_TEST_PATTERNS:
        suite.addTests(loader.discover(str(ROOT / "tests"), pattern=pattern))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
