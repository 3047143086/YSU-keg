"""Exact same-input forward and finite actual-gradient conjunctions."""
EXPECTED_PARAMETERS = (
    'FeqDecomposeOperator.fusion_layer_item.freq_gate.0.bias',
    'FeqDecomposeOperator.fusion_layer_item.freq_gate.0.weight',
    'FeqDecomposeOperator.fusion_layer_item.freq_weights',
    'FeqDecomposeOperator.fusion_layer_item.gate_scale',
    'FeqDecomposeOperator.fusion_layer_user.freq_gate.0.bias',
    'FeqDecomposeOperator.fusion_layer_user.freq_gate.0.weight',
    'FeqDecomposeOperator.fusion_layer_user.freq_weights',
    'FeqDecomposeOperator.fusion_layer_user.gate_scale',
    'behavior_item_bias.weight', 'behavior_user_bias.weight', 'gate_base_scale', 'gate_behavior_scale',
    'id_preference', 'image_embedding.weight', 'item_id_embedding.weight',
    't_gcn.MLP.bias', 't_gcn.MLP.weight', 't_gcn.MLP_1.bias', 't_gcn.MLP_1.weight',
    't_preference', 'text_embedding.weight',
    'v_gcn.MLP.bias', 'v_gcn.MLP.weight', 'v_gcn.MLP_1.bias', 'v_gcn.MLP_1.weight',
    'v_preference', 'weight_u',
)
EXPECTED_SHAPES = tuple((192403 if index in (0, 1, 2, 6, 7, 8) else 63001, 64) for index in range(12))


def same_input_forward_passed(result):
    if result.get('full_forward_count') != 1 or not result.get('actual_loss_finite') or result.get('loss_component_count', 0) < 1:
        return False
    rows = result.get('rows', [])
    if len(rows) != 12 or [row.get('capture_index') for row in rows] != list(range(12)):
        return False
    for index, row in enumerate(rows):
        if (row.get('module') != 'FeqDecomposeOperator' or tuple(row.get('input_shape', [])) != EXPECTED_SHAPES[index]
                or row.get('input_dtype') != 'torch.float32' or row.get('bands_requested') != 3):
            return False
        if any(row.get(flag) is not True for flag in (
            'input_unchanged', 'same_storage_shadow_input', 'shadow_rng_unchanged',
            'candidate_rng_unchanged', 'shadow_outputs_detached', 'outputs_finite',
            'input_version_storage_unchanged')):
            return False
        if row.get('successful_shadow_svd_calls') != 1 or row.get('successful_candidate_svd_calls') != 1 or row.get('svd_failures') != 0:
            return False
        exact = row.get('bands_exact', [])
        if len(exact) != 3 or any(value is not True for value in exact):
            return False
        custom = row.get('candidate_outputs_custom_grad_fn', [])
        if len(custom) != 3 or any(value is not True for value in custom):
            return False
    return True


def integration_passed(result):
    if not same_input_forward_passed(result) or result.get('backward_calls') != 1:
        return False
    gradients = result.get('gradients', {})
    if set(gradients) != set(EXPECTED_PARAMETERS) or len(gradients) != 27:
        return False
    if any(row.get('finite') is not True or row.get('element_count', 0) < 1 for row in gradients.values()):
        return False
    for row in result['rows']:
        outputs = row.get('candidate_output_gradient_events', [])
        inputs = row.get('actual_input_gradient_events', [])
        if len(outputs) != 3 or not any(outputs) or not inputs:
            return False
        observed = inputs + [event for events in outputs for event in events]
        if any(event.get('finite') is not True or event.get('element_count', 0) < 1 for event in observed):
            return False
    return all(result.get(flag) is True for flag in (
        'state_unchanged', 'parameter_identities_unchanged', 'preference_aliases_unchanged',
        'observer_restored', 'svd_identity_restored', 'interaction_unchanged', 'restored_rng_matches',
        'forward_rng_unchanged', 'backward_rng_unchanged', 'backend_settings_unchanged',
        'protected_hashes_unchanged', 'old_receipts_unchanged', 'deployment_config_unchanged',
        'registered_parameter_coverage_complete', 'all_alias_gradients_finite', 'no_reported_svd_fallback'))
