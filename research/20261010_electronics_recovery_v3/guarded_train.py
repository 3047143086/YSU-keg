"""Fresh frozen Electronics recovery; original uncached train/eval orchestration."""
import json
import os
import sys
from pathlib import Path

from prerequisite_gate import require_prerequisites, require_root_review, verify_frozen
from recovery_protocol import numerical_receipt, parse_frozen_arguments


def main():
    root = Path(__file__).resolve().parent
    args = parse_frozen_arguments(sys.argv[1:], root)
    # These checks happen before any Torch/CUDA import, data loader or output directory.
    require_root_review(root, 'start')
    require_prerequisites(root)
    verify_frozen(root)
    from run_queue import resources
    resource = resources()
    if not (resource['query_succeeded'] and not resource['gpu1_compute_processes'] and
            resource['gpu_free_bytes'] >= 14 * 1024 ** 3 and
            resource['ram_available_bytes'] >= 12 * 1024 ** 3 and resource['available_cpu_count'] >= 4):
        raise RuntimeError('Fresh child resource check failed before Torch/CUDA initialization')
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    sys.path[:0] = [str(root), str(args.bridge_root / 'src')]
    from common.trainer import Trainer
    from recovery_guard import RecoveryTrainingGuard
    import numpy as np
    import torch
    import train
    reference = json.loads((root / 'v1_reference_config.json').read_text())
    if torch.__version__ != reference['torch_version'] or np.__version__ != reference['numpy_version']:
        raise RuntimeError('Original numerical dependency versions changed')

    guard = None
    original_init, original_epoch = Trainer.__init__, Trainer._train_epoch
    original_loop = train.training_loop
    # Trainer.evaluate and train.final_validation are deliberately never replaced.
    original_evaluate, original_final_validation = Trainer.evaluate, train.final_validation

    def guarded_init(self, *arguments, **keywords):
        nonlocal guard
        original_init(self, *arguments, **keywords)
        if guard is not None or self.mg or self.clip_grad_norm:
            raise RuntimeError('Expected one original Adam trainer without clipping or multistep changes')
        guard = RecoveryTrainingGuard(self.model, self.optimizer, args.output)

    def guarded_epoch(self, train_data, epoch_idx, loss_func=None):
        guard.enter_epoch(epoch_idx)
        try:
            return original_epoch(self, train_data, epoch_idx, loss_func=loss_func)
        except (FloatingPointError, RuntimeError) as error:
            if not (args.output / 'failure.json').exists():
                guard.fail('backward' if isinstance(error, FloatingPointError) else 'epoch_runtime', str(error))
            raise

    def guarded_loop(trainer, train_data, valid, max_epochs, patience, loss_func, save, emit):
        if max_epochs != 300 or patience != 20:
            raise RuntimeError('Original epoch/patience rules changed')
        def captured_save(epoch, score, metrics):
            save(epoch, score, metrics)
            import torch
            from numerics import rng_state
            packet = {'epoch': epoch, 'rng': rng_state(), 'all_items': list(train_data.all_items),
                      'dataframe': train_data.dataset.df, 'loader_pr': train_data.pr,
                      'loader_inter_pr': train_data.inter_pr, 'test_evaluated': False}
            torch.save(packet, args.output / 'best_rng.pending.pt')
            os.replace(args.output / 'best_rng.pending.pt', args.output / 'best_rng.pt')

        def captured_emit(row):
            row['finite_gradient_steps_completed'] = guard.steps
            row['nonfinite_steps_skipped'] = 0
            row['backward_implementation_changed'] = True
            row['eval_cache_integrated'] = False
            emit(row)
        return original_loop(trainer, train_data, valid, max_epochs, patience,
                             loss_func, captured_save, captured_emit)

    Trainer.__init__, Trainer._train_epoch = guarded_init, guarded_epoch
    train.training_loop = guarded_loop
    try:
        train.main()
        if Trainer.evaluate is not original_evaluate or train.final_validation is not original_final_validation:
            raise RuntimeError('Original uncached evaluation path changed')
        receipt = numerical_receipt(guard.steps, guard.model._spectral_recovery_counts)
        with (args.output / 'numerics_receipt.json').open('x') as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
    finally:
        if guard is not None:
            guard.close()
        Trainer.__init__, Trainer._train_epoch = original_init, original_epoch
        train.training_loop = original_loop
        if args.output.exists():
            try:
                post = verify_frozen(root)
                post['prerequisites'] = require_prerequisites(root)
            except BaseException as error:
                post = {'all_match': False, 'error': repr(error), 'test_evaluated': False}
            with (args.output / 'post_run_guard.json').open('x') as stream:
                json.dump(post, stream, indent=2, allow_nan=False)
            if not post['all_match']:
                raise RuntimeError('Protected recovery state changed during job')


if __name__ == '__main__':
    main()
