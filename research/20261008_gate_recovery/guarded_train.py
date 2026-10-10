"""Use the unchanged training runner with finite-gradient and replay capture."""
import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--bridge-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpu-id', default='1')
    args, _ = parser.parse_known_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id if args.device == 'cuda' else ''
    sys.path[:0] = [str(Path(__file__).resolve().parent), str(args.bridge_root / 'src')]
    from common.trainer import Trainer
    from numerics import FiniteTrainingGuard
    import train

    guard = None
    original_init, original_epoch = Trainer.__init__, Trainer._train_epoch
    original_loop = train.training_loop

    def guarded_init(self, *arguments, **keywords):
        nonlocal guard
        original_init(self, *arguments, **keywords)
        if guard is not None or self.mg:
            raise RuntimeError('Expected one trainer with the frozen single-step configuration')
        guard = FiniteTrainingGuard(self.model, self.optimizer, args.output)

    def guarded_epoch(self, train_data, epoch_idx, loss_func=None):
        guard.enter_epoch(epoch_idx)
        return original_epoch(self, train_data, epoch_idx, loss_func=loss_func)

    def guarded_loop(trainer, train_data, valid, max_epochs, patience, loss_func, save, emit):
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
            emit(row)
        return original_loop(trainer, train_data, valid, max_epochs, patience,
                             loss_func, captured_save, captured_emit)

    Trainer.__init__, Trainer._train_epoch = guarded_init, guarded_epoch
    train.training_loop = guarded_loop
    try:
        train.main()
        with (args.output / 'numerics_receipt.json').open('x') as stream:
            json.dump({'finite_gradient_steps_completed': guard.steps, 'nonfinite_steps_skipped': 0,
                       'solver_changed': False, 'learning_algorithm_changed': False,
                       'test_evaluated': False}, stream, indent=2, allow_nan=False)
    finally:
        if guard is not None:
            guard.close()
        Trainer.__init__, Trainer._train_epoch = original_init, original_epoch
        train.training_loop = original_loop


if __name__ == '__main__':
    main()
