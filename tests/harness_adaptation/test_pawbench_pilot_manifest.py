from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts" / "harness_adaptation" / "pawbench_pilot_manifest.py"
SPEC = importlib.util.spec_from_file_location("pawbench_pilot_manifest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def task_text(task_id: str, scenario: str, environment: str = "closed") -> str:
    return f"""---
id: {task_id}
name: {task_id}
category: office
grading_type: hybrid
timeout_seconds: 300
labels:
  environment: {environment}
  scenario: {scenario}/Example
  complexity: L2
  modality:
    type: text
    channels: []
---
## Prompt
Do work.
## Automated Checks
```python
def grade(transcript, workspace_path):
    return {{"ok": 1.0}}
```
"""


class ManifestSelectionTest(unittest.TestCase):
    def test_selection_is_stratified_and_excludes_open_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "T001.md").write_text(task_text("one", "Office_Productivity"))
            (root / "T002.md").write_text(task_text("two", "Data_Analytics"))
            (root / "T003.md").write_text(task_text("three", "Office_Productivity"))
            (root / "T004.md").write_text(task_text("four", "Content_Creation"))
            (root / "T005.md").write_text(task_text("five", "Knowledge", "open"))
            tasks = MODULE.select_tasks(root, 4)
            self.assertEqual([t["task_id"] for t in tasks], ["four", "two", "one", "three"])
            self.assertTrue(all(t["sha256"] for t in tasks))

    def test_rejects_per_harness_endpoint_override(self) -> None:
        manifest = {
            "schema": MODULE.SCHEMA,
            "execution_overlay": {
                "patch": MODULE.PAWBENCH_PATCH_RELPATH,
                "sha256": MODULE.sha256_file(MODULE.PAWBENCH_PATCH),
                "request_proxy": MODULE.REQUEST_PROXY_RELPATH,
                "request_proxy_sha256": MODULE.sha256_file(MODULE.REQUEST_PROXY),
            },
            "shared_endpoint": {"docker_network": "pilot-network"},
            "harnesses": [
                {"name": "qwenpaw", "base_url": "http://wrong"},
                {"name": "openclaw"},
                {"name": "hermes"},
            ],
        }
        with self.assertRaisesRegex(ValueError, "override is forbidden"):
            MODULE.verify_manifest(manifest, Path("."), require_images=False)


if __name__ == "__main__":
    unittest.main()
