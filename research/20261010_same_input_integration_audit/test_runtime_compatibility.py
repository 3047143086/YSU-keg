"""Actual CPU Torch/frozen method checks before any full-model CUDA probe."""
import importlib.util
from pathlib import Path
import sys
import unittest

import torch
from repeat_guards import sha
from runtime_checks import unique_parameter_rows

PROJECTOR = Path('/data/sunzixuan/spectral_projector_audit_20261010_v1/projector.py')
SOURCE = Path('/data/sunzixuan/electronics_baseline_20261009_v1/carrier/src')


class RuntimeCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sha(PROJECTOR) != 'e44eb739482f3fe0448e1c9fdb066e36c4d6ea22d05a0c25a32734f29a856ab6':
            raise RuntimeError('CPU compatibility test requires exact frozen projector')
        spec=importlib.util.spec_from_file_location('_actual_cpu_frozen_projector',str(PROJECTOR))
        cls.projector=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.projector)
        sys.path.insert(0,str(SOURCE))
        from models.dual_frequency_encoder import FrequencyDecompositionModule
        cls.original_class=FrequencyDecompositionModule

    def value(self):
        return torch.cat((torch.diag(torch.tensor([7.,4.,1.],dtype=torch.float64)),torch.zeros(3,3,dtype=torch.float64))).requires_grad_()

    def test_real_custom_grad_fn_type_and_shadow_detachment(self):
        value=self.value();module=self.original_class.__new__(self.original_class);torch.nn.Module.__init__(module);module.M=3
        with torch.no_grad(): old=self.original_class.frequency_decompose_svd(module,value.detach())
        new=self.projector.projector_bands(value,3)
        self.assertTrue(all(not output.requires_grad and output.grad_fn is None for output in old))
        self.assertTrue(all(output.requires_grad and isinstance(output.grad_fn,self.projector._ProjectorBands._backward_cls) for output in new))
        self.assertTrue(all(torch.equal(left,right) for left,right in zip(old,new)))
        self.assertFalse(torch.cuda.is_initialized())

    def test_output_input_hooks_reach_and_preserve_actual_gradient(self):
        first=self.value();second=first.detach().clone().requires_grad_()
        outputs=self.projector.projector_bands(first,3);observed=[]
        first.register_hook(lambda gradient: observed.append(('input',gradient.detach().clone())))
        for index,output in enumerate(outputs):
            output.register_hook(lambda gradient,index=index: observed.append((index,gradient.detach().clone())))
        sum(output.square().sum() for output in outputs).backward()
        sum(output.square().sum() for output in self.projector.projector_bands(second,3)).backward()
        self.assertEqual(len(observed),4);self.assertTrue(all(torch.isfinite(gradient).all() for _,gradient in observed))
        self.assertTrue(torch.equal(first.grad,second.grad));self.assertFalse(torch.cuda.is_initialized())

    def test_first_registered_alias_name_matches_original_deduplication(self):
        model=torch.nn.Module();model.preference=torch.nn.Parameter(torch.ones(2))
        model.child=torch.nn.Module();model.child.preference=model.preference
        self.assertEqual([name for name,_ in unique_parameter_rows(model)],['preference'])
        self.assertEqual([name for name,_ in unique_parameter_rows(model)], [name for name,_ in model.named_parameters()])


if __name__ == '__main__': unittest.main()
