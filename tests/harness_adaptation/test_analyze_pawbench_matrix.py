from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[2]
    / "scripts"
    / "harness_adaptation"
    / "analyze_pawbench_matrix.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_pawbench_matrix", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def row(agent: str, task: str, score: float) -> dict:
    return {
        "agent": agent,
        "task_id": task,
        "score": score,
        "passed": score == 1.0,
        "status": "success",
        "execution_time": 1.0,
        "anomaly": {"is_anomalous": False, "has_api_error": False},
    }


class AnalyzeMatrixTest(unittest.TestCase):
    def test_computes_paired_transitions_and_net_difference(self) -> None:
        result = MODULE.analyze(
            [
                row("a", "t1", 1.0),
                row("a", "t2", 1.0),
                row("a", "t3", 0.5),
                row("b", "t1", 0.0),
                row("b", "t2", 1.0),
                row("b", "t3", 1.0),
            ]
        )
        pair = result["pairs"]["a_vs_b"]
        self.assertEqual(pair["left_pass_right_fail"], ["t1"])
        self.assertEqual(pair["right_pass_left_fail"], ["t3"])
        self.assertEqual(pair["paired_net_full_passes"], 0)

    def test_rejects_unmatched_task_sets(self) -> None:
        with self.assertRaisesRegex(ValueError, "not identical"):
            MODULE.analyze([row("a", "t1", 1.0), row("b", "t2", 1.0)])


if __name__ == "__main__":
    unittest.main()
