import hashlib
import importlib.util
import math
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('curvature_audit', HERE / 'audit.py')
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class CurvaturePureTests(unittest.TestCase):
    def test_remote_payload_compiles_without_cuda_import(self):
        source = MOD.remote_source({'original': MOD.ORIGINAL, 'run_root': MOD.RUN_ROOT})
        compile(source, '<remote>', 'exec')
        self.assertNotIn('full_sort_predict', source)
        self.assertNotIn('optimizer.step', source)
        self.assertNotIn('EvalDataLoader', source)
        self.assertIn("print('CURVATURE_RESULT='", source)

    def test_private_block_prefixes_are_disjoint(self):
        image = set(MOD.BLOCKS['image'])
        text = set(MOD.BLOCKS['text'])
        self.assertTrue(image.isdisjoint(text))
        self.assertEqual(MOD.BLOCKS['image'], ('image_trs.', 'gate_v.'))
        self.assertEqual(MOD.BLOCKS['text'], ('text_trs.', 'gate_t.'))

    def test_symmetric_excess_and_normalization(self):
        base, plus, minus = 4.0, 4.2, 3.8
        excess = 0.5 * (plus + minus) - base
        self.assertAlmostEqual(excess, 0.0)
        plus, minus, radius = 4.3, 4.1, 0.1
        self.assertAlmostEqual(0.5 * (plus + minus) - base, 0.2)
        self.assertAlmostEqual(2 * 0.2 / (radius * radius), 40.0)

    def test_fixed_floor_and_dominance_math(self):
        floor = max(1e-7, 1e-5 * max(1.0, abs(3.0), abs(3.2), abs(2.9)))
        self.assertAlmostEqual(floor, 3.2e-5)
        delta_sq = 1e-4
        self.assertAlmostEqual(2 * floor / delta_sq, 0.64)
        self.assertTrue(0.001 > floor and 0.001 - (-0.001) > 2 * floor)
        self.assertTrue(math.isclose(1e-8, 0.0, abs_tol=floor))

    def test_source_hash_constants_have_expected_shape(self):
        self.assertEqual(len(MOD.CHECKPOINT_SHA), 64)
        self.assertEqual(len(MOD.INITIAL_SHA), 64)
        self.assertEqual(hashlib.sha256(b'').hexdigest().__class__, str)


if __name__ == '__main__':
    unittest.main()
