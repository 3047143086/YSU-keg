"""Red-team the independently versioned prerequisite, lifecycle and lineage gates."""
import copy
import contextlib
import io
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from prerequisite_gate import (PROJECTOR_SHA256, PrerequisiteError, check_cpu,
    check_old_gpu_archived, check_repeat, check_math_tests, check_same_input,
    check_same_input_cpu_tests, SAME_INPUT_CONFIG_SHA256,
    inspect_prerequisites, require_prerequisites, require_root_review)
from recovery_protocol import (JOB_SECONDS, QUEUE_SECONDS, SEEDS, assert_algorithm_config,
    numerical_receipt, parse_frozen_arguments, run_command)
from same_input_gate import EXPECTED_PARAMETERS, EXPECTED_SHAPES
ROOT=Path(__file__).resolve().parent

def fixture(name):
    return json.loads((ROOT/'receipts'/(name+'.json')).read_bytes())


def same_input_fixture():
    # Complete synthetic schema for rejection tests only. Never a runtime receipt.
    digest='1'*64
    flags=('input_unchanged','same_storage_shadow_input','shadow_rng_unchanged',
        'candidate_rng_unchanged','shadow_outputs_detached','outputs_finite','input_version_storage_unchanged')
    rows=[]
    for index,shape in enumerate(EXPECTED_SHAPES):
        event={'finite':True,'element_count':shape[0]*shape[1]}
        row={'capture_index':index,'module':'FeqDecomposeOperator','input_shape':list(shape),
            'input_dtype':'torch.float32','bands_requested':3,'successful_shadow_svd_calls':1,
            'successful_candidate_svd_calls':1,'svd_failures':0,'bands_exact':[True]*3,
            'candidate_outputs_custom_grad_fn':[True]*3,
            'candidate_output_gradient_events':[[dict(event)],[],[]],
            'actual_input_gradient_events':[dict(event)],'original_band_sha256':[digest]*3,
            'candidate_band_sha256':[digest]*3,'svd_events':[{'phase':phase,'capture_index':index,
            'full_matrices':False,'driver_specified':False,'successful':True} for phase in ('shadow','candidate')]}
        row.update({flag:True for flag in flags})
        for phase in ('before','after_shadow','after_candidate'):
            row['input_sha256_'+phase]=digest
            row['input_metadata_'+phase]={'version':0,'shape':list(shape),'device':'cuda:0'}
        rows.append(row)
    result={'status':'completed_pass','source_revision':'e99e57b42f52f9e6494dd527e0c2eab6ce6d4fc6',
        'full_forward_count':1,'full_forward_calls_started':1,'backward_calls':1,'rows':rows,
        'actual_loss_finite':True,'loss_component_count':1,'loss_values':[.091],'total_loss_value':.091,
        'optimizer_updates':0,'warnings':[],
        'integration_config_sha256_before':digest,'integration_config_sha256_after':digest,
        'gradients':{name:{'finite':True,'element_count':1} for name in EXPECTED_PARAMETERS}}
    for flag in ('state_unchanged','parameter_identities_unchanged','preference_aliases_unchanged',
        'observer_restored','svd_identity_restored','interaction_unchanged','restored_rng_matches',
        'forward_rng_unchanged','backward_rng_unchanged','backend_settings_unchanged',
        'protected_hashes_unchanged','old_receipts_unchanged','deployment_config_unchanged',
        'registered_parameter_coverage_complete','all_alias_gradients_finite','no_reported_svd_fallback',
        'warning_observer_restored','participating_parameter_names_match_original'):
        result[flag]=True
    for flag in ('optimizer_constructed','training_started','test_evaluated','cache_used',
        'solver_override','historical_crash_replayed','generic_vjp_equivalence_claimed','publication_gain_claimed'):
        result[flag]=False
    for phase in ('protected_before','protected_after','old_evidence_before','old_evidence_after'):
        result[phase]={'all_match':True}
    return result


class GateTests(unittest.TestCase):
    def assert_rejected(self, value):
        with self.assertRaises(PrerequisiteError):check_same_input(value,'1'*64)

    def test_complete_synthetic_evidence_checks_individual_rows(self):
        check_same_input(same_input_fixture(),'1'*64)

    def test_partial_or_extra_12_36_27_evidence_rejects(self):
        mutations=[lambda v:v['rows'].pop(),lambda v:v['rows'].append(copy.deepcopy(v['rows'][0])),
            lambda v:v['rows'][0]['bands_exact'].pop(),lambda v:v['rows'][0]['bands_exact'].append(True),
            lambda v:v['gradients'].pop(EXPECTED_PARAMETERS[0]),
            lambda v:v['gradients'].update(extra={'finite':True,'element_count':1}),
            lambda v:v['rows'][0].update(capture_index=1),lambda v:v.update(full_forward_count=2)]
        for index,mutate in enumerate(mutations):
            value=same_input_fixture();mutate(value)
            with self.subTest(mutation=index):self.assert_rejected(value)

    def test_gradient_reach_nonfinite_and_detach_rejects(self):
        mutations=[lambda v:v['rows'][0].update(candidate_output_gradient_events=[[],[],[]]),
            lambda v:v['rows'][0].update(actual_input_gradient_events=[]),
            lambda v:v['rows'][0]['candidate_output_gradient_events'][0][0].update(finite=False),
            lambda v:v['rows'][0]['actual_input_gradient_events'][0].update(finite=False),
            lambda v:v['rows'][0]['candidate_outputs_custom_grad_fn'].__setitem__(0,False),
            lambda v:v['gradients'][EXPECTED_PARAMETERS[0]].update(finite=False),
            lambda v:v.update(backward_calls=0)]
        for index,mutate in enumerate(mutations):
            value=same_input_fixture();mutate(value)
            with self.subTest(mutation=index):self.assert_rejected(value)

    def test_individual_identity_rng_source_guard_and_false_pass_status_rejects(self):
        base=same_input_fixture()
        for flag,value in base.items():
            if value is True:
                wrong=copy.deepcopy(base);wrong[flag]=False
                with self.subTest(flag=flag):self.assert_rejected(wrong)
        for phase in ('protected_before','protected_after','old_evidence_before','old_evidence_after'):
            wrong=copy.deepcopy(base);wrong[phase]['all_match']=False;self.assert_rejected(wrong)
        wrong=copy.deepcopy(base);wrong['status']='completed_fail';self.assert_rejected(wrong)
        wrong=copy.deepcopy(base);wrong['integration_config_sha256_after']='2'*64;self.assert_rejected(wrong)

    def test_svd_fallback_warning_approximate_band_and_mutation_cannot_pass(self):
        mutations=[lambda v:v['rows'][0]['bands_exact'].__setitem__(0,False),
            lambda v:v['rows'][0].update(input_sha256_after_shadow='2'*64),
            lambda v:v['rows'][0]['input_metadata_after_candidate'].update(version=1),
            lambda v:v['rows'][0].update(svd_failures=1),
            lambda v:v['rows'][0]['svd_events'][1].update(driver_specified=True),
            lambda v:v.update(warnings=[{'message':'SVD failed to converge; fallback used'}]),
            lambda v:v.update(loss_values=[float('nan')])]
        for index,mutate in enumerate(mutations):
            wrong=same_input_fixture();mutate(wrong)
            with self.subTest(mutation=index):self.assert_rejected(wrong)

    def test_old_failed_receipt_stays_failed_and_archived_phase_is_not_new_pass(self):
        old=fixture('spectral_gpu');check_old_gpu_archived(old)
        self.assert_rejected(old)
        wrong=copy.deepcopy(old);wrong['status']='completed_pass'
        with self.assertRaises(PrerequisiteError):check_old_gpu_archived(wrong)
        wrong=copy.deepcopy(old);wrong['phase_model']['candidate_integration_passed']=True
        with self.assertRaises(PrerequisiteError):check_old_gpu_archived(wrong)
        wrong=copy.deepcopy(old);wrong['phase_archived']['rows'].pop()
        with self.assertRaises(PrerequisiteError):check_old_gpu_archived(wrong)

    def test_existing_cpu_math_repeat_and_replay_evidence_validates(self):
        check_cpu(fixture('spectral_cpu'));check_math_tests(fixture('projector_cpu_tests'))
        check_repeat(fixture('original_repeat_repeat_receipt'))

    def test_missing_source_revision_or_sha_drift_fail_closed(self):
        manifest=json.loads((ROOT/'prerequisites.json').read_bytes())
        result=inspect_prerequisites(manifest,loader=lambda _: {},hasher=lambda _: 'wrong')
        self.assertFalse(result['eligible']);self.assertEqual(len(result['errors']),7)
        wrong=copy.deepcopy(manifest);wrong['entries'].pop('original_repeat')
        self.assertFalse(inspect_prerequisites(wrong)['eligible'])
        wrong=copy.deepcopy(manifest);wrong['entries']['old_gpu_archived']['sha256']='2'*64
        self.assertFalse(inspect_prerequisites(wrong,loader=lambda _: {},hasher=lambda _: '2'*64)['eligible'])

    def test_ready_flag_cannot_rescue_pending_anchors_and_review(self):
        manifest=json.loads((ROOT/'prerequisites.json').read_bytes())
        manifest['ready_for_training']=True
        manifest['entries']['same_input_gpu']['anchors_pending']=True
        result=inspect_prerequisites(manifest,loader=lambda _: {},hasher=lambda _: 'wrong')
        self.assertFalse(result['eligible']);self.assertIn('same_input_gpu',result['checks'])
        self.assertIn('anchors missing',result['checks']['same_input_gpu']['error'])
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/'root_review.json').write_text(json.dumps({'status':'pending'}))
            with self.assertRaises(PrerequisiteError):require_root_review(directory,'deploy')

    def test_actual_exact_archives_validate_all_seven_roles(self):
        manifest=json.loads((ROOT/'prerequisites.json').read_bytes())
        manifest['ready_for_training']=True  # Fixture only; never writes the runtime file.
        result=inspect_prerequisites(manifest,base_root=ROOT,use_local=True)
        self.assertTrue(result['eligible'],result['errors'])
        self.assertEqual(len(result['checks']),7)
        for role in ('same_input_gpu','same_input_cpu_tests','old_gpu_archived','original_repeat'):
            wrong=copy.deepcopy(manifest);wrong['entries'][role]['sha256']='2'*64
            self.assertFalse(inspect_prerequisites(wrong,base_root=ROOT,use_local=True)['eligible'])
        wrong=copy.deepcopy(manifest)
        wrong['entries']['same_input_gpu']['source_revision']='0'*40
        self.assertFalse(inspect_prerequisites(wrong,base_root=ROOT,use_local=True)['eligible'])
        wrong=copy.deepcopy(manifest)
        wrong['entries']['same_input_gpu']['payload_sha256']['integration_gate.py']='2'*64
        self.assertFalse(inspect_prerequisites(wrong,base_root=ROOT,use_local=True)['eligible'])

    def test_actual_same_input_cpu_raw_cannot_be_replaced_by_source_test_pass(self):
        entry=json.loads((ROOT/'prerequisites.json').read_bytes())['entries']['same_input_cpu_tests']
        raw=json.loads((ROOT/entry['local_path']).read_bytes())
        check_same_input_cpu_tests(raw,SAME_INPUT_CONFIG_SHA256)
        mutations=[lambda v:v.update(exit_code=1),lambda v:v.update(stderr='Ran 31 tests\nOK'),
            lambda v:v.update(stderr=v['stderr'].replace('... ok','... skipped',1)),
            lambda v:v.update(integration_config_sha256='2'*64),
            lambda v:v.update(predecessor_anchors_match=False)]
        for mutate in mutations:
            wrong=copy.deepcopy(raw);mutate(wrong)
            with self.assertRaises(PrerequisiteError):check_same_input_cpu_tests(wrong,SAME_INPUT_CONFIG_SHA256)

    def test_partial_root_review_payload_cannot_authorize_remote_action(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'train.py').write_text('source')
            review={'status':'approved','reviewed_by':'/root','remote_actions_authorized':True,
                    'reviewed_payload_sha256':{'train.py':hashlib.sha256(b'source').hexdigest()}}
            (root/'root_review.json').write_text(json.dumps(review))
            with self.assertRaisesRegex(PrerequisiteError,'complete preparation payload'):
                require_root_review(root,'deploy')

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
