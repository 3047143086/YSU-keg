"""Read registered Parameter paths without version-sensitive deduplication flags."""


def all_named_parameters(model):
    rows = []

    def visit(module, prefix, ancestors):
        if id(module) in ancestors:
            raise ValueError('Cyclic Module registration prevents a complete parameter snapshot')
        ancestors = ancestors | {id(module)}
        for name, parameter in module._parameters.items():
            if parameter is not None:
                rows.append((prefix + name, parameter))
        for name, child in module._modules.items():
            if child is not None:
                visit(child, prefix + name + '.', ancestors)

    visit(model, '', set())
    if len({name for name, _ in rows}) != len(rows):
        raise ValueError('Registered Parameter paths must be unique')
    return rows


def parameter_coverage(model):
    import torch
    rows = dict(all_named_parameters(model))
    # keep_vars exposes registered parameters without wrapping them as detached tensors.
    expected = {name: value for name, value in model.state_dict(keep_vars=True).items()
                if isinstance(value, torch.nn.Parameter)}
    if set(rows) != set(expected) or any(rows[name] is not expected[name] for name in rows):
        raise ValueError('All registered Parameter state paths/objects must be covered; buffers are excluded')
    return {'all_parameter_state_paths_covered': True, 'registered_path_count': len(rows),
            'unique_parameter_object_count': len({id(value) for value in rows.values()}),
            'buffer_paths_excluded': sorted(set(model.state_dict()) - set(expected))}


def synthetic_check():
    import torch
    model = torch.nn.Module()
    shared = torch.nn.Parameter(torch.ones(2))
    model.direct = shared
    model.same = shared
    child = torch.nn.Module()
    child.weight = shared
    child.register_buffer('buffer', torch.zeros(2))
    model.child = child
    model.child_alias = child
    before = {name: value.clone() for name, value in model.state_dict().items()}
    coverage = parameter_coverage(model)
    rows = dict(all_named_parameters(model))
    if set(rows) != {'direct', 'same', 'child.weight', 'child_alias.weight'} or any(value is not shared for value in rows.values()):
        raise RuntimeError('Synthetic shared-parameter/child-alias coverage failed')
    if any(not torch.equal(before[name], value) for name, value in model.state_dict().items()):
        raise RuntimeError('Read-only parameter enumeration mutated synthetic state')
    return {'torch_version': torch.__version__, 'parameter_paths': sorted(rows),
            'state_unchanged': True, 'coverage': coverage}
