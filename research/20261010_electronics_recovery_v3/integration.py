"""Instance-local integration draft; no training or global source mutation."""
from types import MethodType


def install_projector(model, projector_function):
    """Replace one instance's SVD-band derivative, keeping its parameters intact.

    The caller must verify the frozen carrier and projector source hashes and
    reviewed receipts. This helper performs no solver selection or recovery.
    """
    operator = model.FeqDecomposeOperator
    if type(operator).__name__ != 'FrequencyDecompositionModule':
        raise TypeError('Expected the frozen BRIDGE frequency decomposition module')
    if operator.decomp_method != 'svd' or operator.M != 3:
        raise ValueError('Expected the original three-band SVD configuration')
    if '_projector_recovery_original' in vars(operator):
        raise RuntimeError('Numerical recovery is installed once per model instance')
    parameters_before = tuple((name, id(value)) for name, value in model.named_parameters())
    state_keys_before = tuple(model.state_dict())
    original = operator.frequency_decompose_svd
    counts = {'calls': 0}

    def recovered(instance, representation):
        if instance is not operator:
            raise RuntimeError('Frequency recovery belongs to exactly one module')
        guard = getattr(model, '_spectral_recovery_guard', None)
        outputs = (projector_function(representation, instance.M) if guard is None else
                   guard.projector_call(representation, instance.M, projector_function))
        if len(outputs) != instance.M or any(output.shape != representation.shape for output in outputs):
            raise RuntimeError('Frequency component contract changed')
        counts['calls'] += 1
        return list(outputs)

    operator._projector_recovery_original = original
    operator.frequency_decompose_svd = MethodType(recovered, operator)
    if parameters_before != tuple((name, id(value)) for name, value in model.named_parameters()):
        raise RuntimeError('Numerical integration changed parameter identity')
    if state_keys_before != tuple(model.state_dict()):
        raise RuntimeError('Numerical integration changed state keys')
    return counts
