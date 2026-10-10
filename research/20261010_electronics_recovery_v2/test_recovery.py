"""Meaningful fail-closed gate, lineage, budget and argument mutation tests."""
import copy
import contextlib
import io
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from prerequisite_gate import (PROJECTOR_SHA256, PrerequisiteError, check_cpu, check_gpu,
                               inspect_prerequisites, require_prerequisites)
from recovery_protocol import (JOB_SECONDS, QUEUE_SECONDS, SEEDS, assert_algorithm_config,
                               numerical_receipt, parse_frozen_arguments, run_command)

ROOT = Path(__file__).resolve().parent


def fixture(name):
    return json.loads((ROOT / 'receipts' / (name + '.json')).read_text())


def passing_gpu_fixture():
    # Synthetic positive fixture based on the frozen negative receipt's schema.
    # It is used only by tests; no runtime file or scientific receipt is changed.
    value = copy.deepcopy(fixture('spectral_gpu'))
    value['status'] = 'completed_pass'
    value['phase_model']['candidate_integration_passed'] = True
    for row in value['phase_model']['rows']:
        row['candidate_vs_original_input']['passed'] = True
        for band in row['candidate_vs_original_bands']:
            band['passed'] = True
    return value


class GateTests(unittest.TestCase):
    def test_actual_cpu_pass_and_gpu_failure_remain_separate(self):
        check_cpu(fixture('spectral_cpu'))
        with self.assertRaisesRegex(PrerequisiteError, 'GPU projector audit did not pass'):
            check_gpu(fixture('spectral_gpu'))
        self.assertTrue(fixture('spectral_gpu')['phase_archived']['candidate_passed'])

    def test_changing_only_overall_status_cannot_rescue_failed_model(self):
        value = fixture('spectral_gpu')
        value['status'] = 'completed_pass'
        with self.assertRaises(PrerequisiteError):
            check_gpu(value)

    def test_all_twelve_model_bands_and_parameter_guards_are_required(self):
        check_gpu(passing_gpu_fixture())
        mutations = [lambda v: v['phase_model']['rows'].pop(),
                     lambda v: v['phase_model']['rows'][7]['candidate_vs_original_bands'][2].update(passed=False),
                     lambda v: v['phase_model'].update(state_unchanged=False),
                     lambda v: v['phase_model'].update(loss_matches=False),
                     lambda v: v['phase_model'].update(participating_parameters_match=False),
                     lambda v: v.update(optimizer_updates=1),
                     lambda v: v.update(test_loader_constructed=True),
                     lambda v: v['phase_model'].update(generic_gradient_equivalence_passed=False),
                     lambda v: v['phase_model'].update(candidate_gradient_finite={}),
                     lambda v: v['protected_after'].update(all_match=False)]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                value = passing_gpu_fixture()
                mutate(value)
                with self.assertRaises(PrerequisiteError):
                    check_gpu(value)

    def test_all_cpu_captures_and_available_gradient_controls_are_required(self):
        for index in (0, 5, 11):
            value = fixture('spectral_cpu')
            value['captures'][index]['exact_forward_bands'][1] = False
            with self.assertRaises(PrerequisiteError):
                check_cpu(value)
        value = fixture('spectral_cpu')
        value['captures'][0]['generic_input_gradient_finite'] = True
        value['captures'][0]['generic_gradient_comparison']['passed'] = False
        with self.assertRaises(PrerequisiteError):
            check_cpu(value)

    def test_missing_and_sha_drift_fail_closed(self):
        manifest = json.loads((ROOT / 'prerequisites.json').read_text())
        result = inspect_prerequisites(manifest, loader=lambda _: {}, hasher=lambda _: 'wrong')
        self.assertFalse(result['eligible'])
        self.assertEqual(len(result['errors']), 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'prerequisites.json').write_text(json.dumps(manifest))
            with self.assertRaises(PrerequisiteError):
                require_prerequisites(path)

    def test_frozen_failed_manifest_has_no_launch_permission(self):
        manifest = json.loads((ROOT / 'prerequisites.json').read_text())
        self.assertIs(manifest['ready_for_training'], False)
        self.assertEqual(manifest['reason'], 'spectral_gpu_completed_fail')
        self.assertEqual(manifest['entries']['spectral_gpu']['sha256'],
                         'a840a3a1d952a8300d44951af77d8068b28770057a0ba3ca1dc833e0c21b7eb0')
        self.assertIs(manifest['excluded_eval_cache'], True)


class ProtocolTests(unittest.TestCase):
    def test_original_algorithm_config_is_identical_except_seed_and_plugin(self):
        reference = json.loads((ROOT / 'v1_reference_config.json').read_text())['effective_config']
        actual = dict(reference, seed=2030, spectral_projector_file='frozen',
                      spectral_projector_sha256=PROJECTOR_SHA256)
        assert_algorithm_config(actual, reference)
        for key, value in [('epochs', 30), ('stopping_step', 2), ('learning_rate', .0002),
                           ('train_batch_size', 64), ('behavior_weight', .0),
                           ('freq_decomp_method', 'eigh'), ('feature_ablation', 'image'),
                           ('eval_batch_size', 128), ('seed', 999), ('learner', 'sgd')]:
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                assert_algorithm_config(dict(actual, **{key: value}), reference)

    def test_arguments_disallow_resume_smoke_other_device_or_budget_override(self):
        command = run_command(ROOT, 'python', 2026)[3:]
        parse_frozen_arguments(command, ROOT)
        for extra in ('--smoke', '--max-epochs', '--learning-rate', '--resume'):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_frozen_arguments(command + [extra, '1'], ROOT)
        for key, value in (('--device', 'cpu'), ('--gpu-id', '0'), ('--seed', '999'),
                           ('--cuda-memory-fraction', '1.0'), ('--plugin-class', 'BRIDGE')):
            wrong = list(command)
            wrong[wrong.index(key) + 1] = value
            with self.subTest(key=key), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_frozen_arguments(wrong, ROOT)

    def test_fixed_budget_and_correct_numerical_change_labels(self):
        self.assertEqual(SEEDS, (2026, 2027, 2028, 2029, 2030))
        self.assertEqual(JOB_SECONDS, 24 * 3600)
        self.assertEqual(QUEUE_SECONDS, 72 * 3600)
        receipt = numerical_receipt(613, {'calls': 12})
        self.assertIs(receipt['backward_implementation_changed'], True)
        self.assertIs(receipt['solver_changed'], False)
        self.assertIs(receipt['optimizer_changed'], False)
        self.assertIs(receipt['eval_cache_integrated'], False)
        self.assertNotIn('learning_algorithm_changed', receipt)

    def test_original_train_numerical_observer_and_reviewed_projector_are_exact(self):
        expected = {'train.py': '769f11a19de1be8036a9c8e48bd03ab0dc29aee4f6f64e924fcb1011470649be',
                    'numerics.py': '54a073c5a9adf0676e8369b33d9eb2fa01e487cf76e2ecc60343af77183dec19',
                    'projector.py': PROJECTOR_SHA256}
        for name, sha256 in expected.items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), sha256)
        self.assertFalse((ROOT / 'eval_cache.py').exists())


if __name__ == '__main__':
    unittest.main()
