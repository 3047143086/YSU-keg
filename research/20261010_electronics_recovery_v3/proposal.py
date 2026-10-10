"""Numerically repaired original BRIDGE; import only inside its audited runner."""
import hashlib
import importlib.util
import json
from pathlib import Path

from models.bridge import BRIDGE
from integration import install_projector
from prerequisite_gate import PROJECTOR_SHA256
from recovery_protocol import assert_algorithm_config


def load_frozen_projector(path, expected_sha256):
    path = Path(path).resolve()
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError('A frozen projector SHA256 is required')
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise RuntimeError('Reviewed numerical implementation changed')
    spec = importlib.util.spec_from_file_location('frozen_recovery_projector', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.projector_bands


class ProjectorRecoveryBRIDGE(BRIDGE):
    """Same parameters/scoring, original SVD forward and band-level first VJP."""
    def __init__(self, config, dataset):
        bundle = Path(__file__).resolve().parent
        reference = json.loads((bundle / 'v1_reference_config.json').read_text())['effective_config']
        assert_algorithm_config(config.final_config_dict, reference)
        if Path(config['spectral_projector_file']).resolve() != bundle / 'projector.py':
            raise RuntimeError('The numerical plugin belongs to this exact frozen bundle')
        if config['spectral_projector_sha256'] != PROJECTOR_SHA256:
            raise RuntimeError('Projector numerical version changed')
        super().__init__(config, dataset)
        for name in ('use_activity_simplex', 'use_behavior_proto',
                     'use_private_residual', 'use_residual_gate', 'use_user_graph'):
            if getattr(self, name):
                raise RuntimeError('Unexpected original-baseline branch: ' + name)
        if self.msr_weight != 0 or self.freq_decomp_method != 'svd':
            raise RuntimeError('Expected unchanged original BRIDGE configuration')
        projector = load_frozen_projector(config['spectral_projector_file'],
                                          config['spectral_projector_sha256'])
        self._spectral_recovery_counts = install_projector(self, projector)
