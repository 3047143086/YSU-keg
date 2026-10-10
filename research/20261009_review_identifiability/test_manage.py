import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("ident_manage", Path(__file__).with_name("manage.py"))
manager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manager)


class ImmutableManagerTests(unittest.TestCase):
    def test_payload_refuses_dirty_or_uncommitted_bundle(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "input_audit.json").write_text("{}")
            with patch.object(manager, "LOCAL", root), patch.object(
                    manager.subprocess, "check_output", side_effect=["a"*40, " M core.py"]):
                with self.assertRaisesRegex(RuntimeError, "clean bundle"):
                    manager.deployment_payload()

    def test_missing_input_audit_refuses_deployment(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(manager, "LOCAL", Path(folder)):
            with self.assertRaisesRegex(RuntimeError, "input audit"):
                manager.deployment_payload()

    def test_existing_launch_lock_blocks_start_without_touching_pid(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "launch.lock").write_text("existing launch")
            with patch.object(manager, "REMOTE", str(root)):
                source = manager.remote_source("start")
            result = subprocess.run([__import__("sys").executable, "-c", source], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("already launched", result.stderr)
            self.assertFalse((root / "launch.json").exists())

    def test_changed_payload_is_refused_before_overwriting_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "core.py").write_bytes(b"original")
            payload = {"core.py": manager.base64.b64encode(b"different").decode()}
            with patch.object(manager, "REMOTE", str(root)):
                source = manager.remote_source("deploy", payload)
            result = subprocess.run([__import__("sys").executable, "-c", source], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Existing payload differs", result.stderr)
            self.assertEqual((root / "core.py").read_bytes(), b"original")


if __name__ == "__main__":
    unittest.main()
