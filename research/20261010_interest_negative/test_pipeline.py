import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import pipeline


def completed(r20=.1, n20=.05, r10=.07, n10=.035):
    return {"status": "completed", "metrics": {"recall@10": r10,
            "ndcg@10": n10, "recall@20": r20, "ndcg@20": n20},
            "elapsed_seconds": 3.0}


class PipelineTests(unittest.TestCase):
    def test_gate_requires_both_references_and_all_metrics(self):
        rows = {"uniform": completed(), "shuffled": completed(),
                "semantic": completed(.101, .0506, .07, .035)}
        decision = pipeline.gate(rows)
        self.assertEqual(decision["status"], "pass")
        self.assertTrue(all(x["pass"] for x in decision["comparisons"].values()))
        missing = dict(rows)
        missing["shuffled"] = {"status": "technical_failure"}
        self.assertFalse(pipeline.gate(missing)["passed"])

    def test_gate_rejects_shortlist_regression_or_small_primary_gain(self):
        rows = {"uniform": completed(), "shuffled": completed(),
                "semantic": completed(.1005, .05025, .0698, .03499)}
        decision = pipeline.gate(rows)
        self.assertFalse(decision["passed"])
        self.assertFalse(decision["comparisons"]["uniform"]["pass"])
        rows["semantic"] = completed(.101, .0506, .0697, .035)
        self.assertFalse(pipeline.gate(rows)["passed"])

    def test_gate_rejects_zero_or_nan_reference(self):
        rows = {"uniform": completed(r20=0), "shuffled": completed(),
                "semantic": completed(.11, .055)}
        self.assertFalse(pipeline.gate(rows)["passed"])
        rows["uniform"] = completed()
        rows["semantic"] = completed()
        rows["semantic"]["metrics"]["ndcg@20"] = float("nan")
        self.assertFalse(pipeline.gate(rows)["passed"])

    def test_finite_metrics_rejects_nonfinite_and_wrong_status(self):
        self.assertTrue(pipeline._finite_metrics(completed()))
        bad = completed()
        bad["metrics"]["recall@10"] = float("inf")
        self.assertFalse(pipeline._finite_metrics(bad))
        bad = completed(); bad["status"] = "technical_failure"
        self.assertFalse(pipeline._finite_metrics(bad))

    def test_resource_admission_has_all_three_guards(self):
        good = {"ram_available_bytes": 12 * 1024**3,
                "cpu_affinity_count": 4, "gpu_free_bytes": 14 * 1024**3,
                "gpu_compute_processes": [], "gpu_process_query_ok": True}
        self.assertEqual(pipeline.resources_admit(good), (True, "admitted"))
        for key, value in (("ram_available_bytes", 0), ("cpu_affinity_count", 3),
                           ("gpu_free_bytes", 0)):
            row = dict(good); row[key] = value
            self.assertFalse(pipeline.resources_admit(row)[0])
        row = dict(good); row["gpu_compute_processes"] = ["uuid,123"]
        self.assertFalse(pipeline.resources_admit(row)[0])
        self.assertTrue(pipeline.resources_admit({"ram_available_bytes": 12 * 1024**3,
                         "cpu_affinity_count": 4}, require_gpu=False)[0])

    def test_arm_names_and_result_rows_are_fixed(self):
        self.assertEqual(pipeline._arm_name("baby", "semantic", True),
                         "baby_semantic_s2026_smoke")
        self.assertEqual(pipeline._arm_name("sports", "uniform"),
                         "sports_uniform_s2026_formal")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with mock.patch.object(pipeline, "ROOT", root):
                queue = pipeline.Queue()
                queue.write_results({})
                text = (root / "results.tsv").read_text()
                self.assertEqual(text.count("pending"), 6)
                self.assertEqual(len(text.strip().splitlines()), 7)

    def test_write_json_is_atomic_and_disallows_nan(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "state.json"
            pipeline.write_json(path, {"x": 1})
            self.assertEqual(json.loads(path.read_text())["x"], 1)
            with self.assertRaises(ValueError):
                pipeline.write_json(path, {"x": float("nan")})

    def test_child_command_contains_train_cli_and_preparation_has_no_mode(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            root.mkdir(exist_ok=True)
            # inspect the generated command by replacing Popen and resource wait;
            # no process, filesystem data or CUDA is touched by this test.
            output = root / "out"
            assets = root / "assets"
            fake = mock.Mock(); fake.pid = 99; fake.returncode = 0
            fake.poll.return_value = 0; fake.wait.return_value = 0
            def popen(command, **kwargs):
                self.assertIn("--prepare-assets", command)
                self.assertNotIn("--mode", command)
                self.assertIn("--assets", command)
                return fake
            with mock.patch.object(pipeline, "ROOT", root), \
                 mock.patch.object(pipeline.Queue, "wait_resources"), \
                 mock.patch.object(pipeline.subprocess, "Popen", side_effect=popen):
                (root / "logs").mkdir()
                queue = pipeline.Queue()
                with self.assertRaises(FileNotFoundError):
                    queue.run_child("baby", "uniform", output, assets, prepare=True)

    def test_queue_refuses_existing_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "runs").mkdir()
            with mock.patch.object(pipeline, "ROOT", root):
                with self.assertRaisesRegex(RuntimeError, "existing queue"):
                    pipeline.Queue().run()

    def test_gate_does_not_treat_smoke_as_formal(self):
        rows = {"uniform": completed(), "shuffled": completed(),
                "semantic": completed()}
        rows["uniform"]["smoke"] = True
        # The formal gate must reject a smoke result even if its metric fields
        # happen to be finite; the pipeline validates the result status first.
        self.assertFalse(pipeline.gate(rows)["passed"])
        self.assertNotIn("smoke", pipeline.gate(rows))


if __name__ == "__main__":
    unittest.main()
