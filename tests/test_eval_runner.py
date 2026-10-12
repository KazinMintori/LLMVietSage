"""Contracts for generation-only evaluation and explicit live-quality gating."""
import contextlib
import io
import json
import subprocess
import sys
import unittest
from unittest.mock import patch

from scripts.run_evals import EvalContractError, _assert, _load_cases, _offline_probe, main


class GenerationEvaluationTests(unittest.TestCase):
    def test_case_ids_are_unique_and_modes_belong_to_generation(self):
        cases = _load_cases(None, None)
        self.assertTrue(cases)
        self.assertEqual(len(cases), len({case["id"] for case in cases}))
        self.assertTrue(all(case.get("mode", "offline") in {"offline", "live"} for case in cases))

    def test_all_offline_assertions_execute_without_database(self):
        for case in _load_cases(None, None):
            if case.get("mode", "offline") == "offline":
                with self.subTest(case=case["id"]):
                    result = _offline_probe(case)
                    for assertion in case["assertions"]:
                        _assert(result, assertion)

    def test_unknown_case_filters_and_missing_result_paths_fail(self):
        with self.assertRaises(EvalContractError):
            _load_cases(None, {"missing-case"})
        with self.assertRaises(EvalContractError):
            _assert({"value": 1}, {"op": "equals", "path": "missing", "value": 1})

    def test_live_quality_is_skipped_and_required_live_is_not_a_false_pass(self):
        result = subprocess.run([sys.executable, "-B", "scripts/run_evals.py", "--suite", "normal-chat", "--mode", "live", "--require-live"], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stdout)["summary"]["skipped"], 1)

    def test_live_gate_cannot_be_requested_in_offline_mode(self):
        with patch("sys.argv", ["run_evals.py", "--require-live"]), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main()
        self.assertEqual(error.exception.code, 2)

    def test_offline_cli_reports_real_passes(self):
        output = io.StringIO()
        with patch("sys.argv", ["run_evals.py"]), contextlib.redirect_stdout(output):
            self.assertEqual(main(), 0)
        self.assertGreater(json.loads(output.getvalue())["summary"]["passed"], 0)


if __name__ == "__main__":
    unittest.main()
