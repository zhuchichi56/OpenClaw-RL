import importlib.util
import os
import subprocess
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts/harness_adaptation/pawbench_pilot_runner.py"
SCRIPT_DIR = SCRIPT.parent
SPEC = importlib.util.spec_from_file_location("pawbench_pilot_runner", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader


class BuildAgentConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import sys

        sys.path.insert(0, str(SCRIPT_DIR))
        SPEC.loader.exec_module(MODULE)

    def test_all_controls_come_from_shared_endpoint(self) -> None:
        manifest = {
            "model": {"id": "custom/frozen"},
            "shared_endpoint": {
                "base_url": "http://model:30001/v1",
                "docker_network": "pilot",
                "timeout_multiplier": 1.0,
                "context_limit": 32768,
                "max_generated_tokens": 8192,
                "temperature": 0.0,
            },
        }
        config = MODULE.build_agent_config(
            manifest, {"name": "hermes", "image": "hermes:test"}
        )
        self.assertEqual(config["docker_network"], "pilot")
        self.assertEqual(config["max_tokens"], 8192)
        self.assertEqual(config["generate_kwargs"]["temperature"], 0.0)
        self.assertEqual(config["model"], "custom/frozen")
        self.assertTrue(config["automated_only_grading"])


class PawBenchOverlayIntegrationTest(unittest.TestCase):
    def test_hybrid_automated_only_bypasses_llm_judge(self) -> None:
        pawbench_root = os.environ.get("PAWBENCH_ROOT")
        if not pawbench_root:
            self.skipTest("PAWBENCH_ROOT is required for the overlay integration test")
        code = r'''
from types import SimpleNamespace
from pawbench import grader

sentinel = object()
original_auto = grader._grade_automated
original_judge = grader._grade_llm_judge
try:
    grader._grade_automated = lambda *args, **kwargs: sentinel
    grader._grade_llm_judge = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("LLM judge must not run")
    )
    result = grader.grade_task(
        task=SimpleNamespace(grading_type="hybrid"),
        execution_result={},
        automated_only=True,
    )
    assert result is sentinel
finally:
    grader._grade_automated = original_auto
    grader._grade_llm_judge = original_judge
'''
        subprocess.run(
            ["python", "-c", code],
            cwd=pawbench_root,
            env={**os.environ, "PYTHONPATH": pawbench_root},
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
