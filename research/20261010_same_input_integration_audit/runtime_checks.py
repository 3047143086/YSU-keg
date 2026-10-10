"""Read-only state/RNG/backend checks; gradients are inspected separately."""
import gc
import hashlib
import os
import pickle
import random
from parameter_snapshot import all_named_parameters, parameter_coverage


def rng_hashes(helper, packet=None):
    import numpy as np
    import torch
    values = packet if packet is not None else {'python': random.getstate(), 'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}
    return {'python': hashlib.sha256(pickle.dumps(values['python'], protocol=4)).hexdigest(),
            'numpy': hashlib.sha256(pickle.dumps(values['numpy'], protocol=4)).hexdigest(),
            'torch': helper.tensor_hash(values['torch']),
            'cuda': [helper.tensor_hash(value) for value in values['cuda']]}


def backend_settings():
    import torch
    return {'deterministic_algorithms_enabled': torch.are_deterministic_algorithms_enabled(),
        'deterministic_warn_only_enabled': (torch.is_deterministic_algorithms_warn_only_enabled()
                                            if hasattr(torch, 'is_deterministic_algorithms_warn_only_enabled') else None),
        'cudnn_deterministic': torch.backends.cudnn.deterministic, 'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_enabled': torch.backends.cudnn.enabled, 'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'float32_matmul_precision': torch.get_float32_matmul_precision() if hasattr(torch, 'get_float32_matmul_precision') else None,
        'cuda_version': torch.version.cuda, 'cudnn_version': torch.backends.cudnn.version(),
        'environment': {name: os.environ.get(name) for name in ('CUBLAS_WORKSPACE_CONFIG', 'NVIDIA_TF32_OVERRIDE',
                            'CUDA_VISIBLE_DEVICES', 'CUDA_LAUNCH_BLOCKING', 'PYTHONHASHSEED')}}


def state_snapshot(helper, model):
    import torch
    state = model.state_dict()
    if any(not torch.isfinite(value).all().item() for value in state.values()):
        raise FloatingPointError('Learned state must be finite')
    aliases = {prefix: getattr(model, prefix + '_preference', None) is getattr(model, prefix + '_gcn').preference
               for prefix in ('id', 'v', 't')}
    if not all(aliases.values()):
        raise RuntimeError('Preference alias Parameter identity changed')
    return {'values_sha256': helper.state_hashes(model),
        'schemas': {name: {'shape': list(value.shape), 'dtype': str(value.dtype)} for name, value in state.items()},
        'parameter_identities': {name: id(value) for name, value in all_named_parameters(model)},
        'parameter_requires_grad': {name: value.requires_grad for name, value in all_named_parameters(model)},
        'parameter_state_coverage': parameter_coverage(model), 'preference_alias_identity': aliases,
        'module_training_modes': {name: module.training for name, module in model.named_modules()}}


def interaction_snapshot(helper, interaction):
    import torch
    values = [interaction] if torch.is_tensor(interaction) else list(interaction)
    if not values or any(not torch.is_tensor(value) for value in values):
        raise RuntimeError('Frozen interaction must contain tensors')
    return [{'sha256': helper.tensor_hash(value), 'shape': list(value.shape), 'dtype': str(value.dtype)} for value in values]


def unique_parameter_rows(model):
    seen = set()
    rows = []
    for name, value in all_named_parameters(model):
        if id(value) not in seen:
            rows.append((name, value))
            seen.add(id(value))
    return rows


def cleanup(model):
    import torch
    cleared = []
    for name in ('id_rep', 'v_rep', 't_rep', 'result_embed'):
        value = getattr(model, name, None)
        if torch.is_tensor(value) and not isinstance(value, torch.nn.Parameter):
            cleared.append(name)
            setattr(model, name, None)
    value = None
    gc.collect()
    torch.cuda.empty_cache()
    return cleared
