"""Strict CPU reconstruction of the frozen GPU checkpoint registration shape."""
import ast


def align_original_gpu_structure(model, state, source_text):
    import torch
    # The frozen unused attribute must have exactly one Store and no Loads.
    attrs = [node for node in ast.walk(ast.parse(source_text))
             if isinstance(node, ast.Attribute) and node.attr == 'result_embed']
    if len(attrs) != 1 or not isinstance(attrs[0].ctx, ast.Store):
        raise RuntimeError('result_embed is no longer proven to be construction-only')
    aliases = ('id', 'v', 't')
    expected_extra = {name + '_preference' for name in aliases}
    model_keys, checkpoint_keys = set(model.state_dict()), set(state)
    if model_keys - checkpoint_keys not in ({'result_embed'}, set()) or checkpoint_keys - model_keys != expected_extra:
        raise RuntimeError('Checkpoint registration mismatch differs from the frozen GPU peculiarity')
    for name in aliases:
        alias, canonical = name + '_preference', name + '_gcn.preference'
        if not torch.equal(state[alias], state[canonical]):
            raise RuntimeError('Checkpoint preference alias disagrees with canonical parameter: ' + alias)
    # GPU construction converts this CPU-created Parameter to a plain Tensor.
    # On CPU .to(cpu) is a no-op, so explicitly reproduce the GPU registration
    # shape only for this construction-only attribute. No algorithm weight
    # is discarded and the frozen source definition is unchanged.
    unused = getattr(model, 'result_embed').detach()
    if 'result_embed' in model._parameters:
        delattr(model, 'result_embed')
    elif not hasattr(model, 'result_embed'):
        raise RuntimeError('Construction-only result_embed attribute is absent')
    model.result_embed = unused
    for name in aliases:
        parameter = getattr(model, name + '_gcn').preference
        setattr(model, name + '_preference', parameter)
        if getattr(model, name + '_preference') is not parameter:
            raise RuntimeError('CPU preference binding did not retain object identity')
    if set(model.state_dict()) != checkpoint_keys:
        raise RuntimeError('CPU model lacks exact checkpoint state-key coverage')
    model.load_state_dict(state, strict=True)
    after = model.state_dict()
    if set(after) != checkpoint_keys or any(not torch.equal(after[key].detach().cpu(), state[key].detach().cpu()) for key in after):
        raise RuntimeError('Strict CPU checkpoint values differ')
    return {'strict_load': True, 'exact_state_key_coverage': True,
            'exact_checkpoint_tensor_values': True,
            'preference_alias_tensor_equal': True, 'preference_alias_object_identity': True,
            'original_gpu_registration_shape_preserved': True,
            'construction_only_plain_tensor_attribute': 'result_embed',
            'discarded_algorithm_weights': []}
