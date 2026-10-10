"""Observe the first bad backward before Adam can write invalid parameters.

This training/validation probe resumes weights and optimizer state, but cannot
replay the old run's unsaved RNG. It does not produce scientific results.
"""
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

SOURCE = Path('/data/sunzixuan/decoupled_gate_20261008_v1')
OUTPUT = Path(__file__).resolve().parent


def main():
    os.chdir(SOURCE / 'carrier/src')
    sys.path[:0] = [str(SOURCE), str(SOURCE / 'carrier/src')]
    import numpy as np
    import torch
    from common.trainer import Trainer
    from proposal import DecoupledGateBRIDGE
    from train import digest, scalarize, split_train_valid, training_loop
    from utils.configurator import Config
    from utils.dataloader import EvalDataLoader, TrainDataLoader
    from utils.dataset import RecDataset
    from utils.utils import init_seed

    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.55, device=0)
    config = Config('BRIDGE', 'baby', {'use_gpu': True, 'gpu_id': os.environ['CUDA_VISIBLE_DEVICES'],
                    'seed': [2026], 'epochs': 300, 'stopping_step': 20,
                    'eval_batch_size': 64, 'save_recommended_topk': False,
                    'req_training': True, 'valid_metric': 'Recall@20'})
    scalarize(config)
    config.final_config_dict.update(json.loads((SOURCE / 'configs/baby_edge.json').read_text()))
    init_seed(2026)
    dataset = RecDataset(config)
    train_dataset, valid_dataset = split_train_valid(dataset, config)
    train = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    valid = EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=64)
    train.pretrain_setup()
    model = DecoupledGateBRIDGE(config, train).to(config['device'])
    trainer = Trainer(config, model)
    if trainer.mg:
        raise RuntimeError('Probe requires the frozen single-step optimizer configuration')
    checkpoint_path = SOURCE / 'runs/baby_edge_s2026/best.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    for alias, nested in (('id_preference', 'id_gcn.preference'),
                          ('v_preference', 'v_gcn.preference'), ('t_preference', 't_gcn.preference')):
        if not torch.equal(checkpoint['model'][alias], checkpoint['model'][nested]):
            raise RuntimeError('Checkpoint preference alias differs: ' + alias)
        module_name, field = nested.split('.')
        setattr(model, alias, getattr(getattr(model, module_name), field))
    model.load_state_dict(checkpoint['model'], strict=True)
    trainer.optimizer.load_state_dict(checkpoint['optimizer'])
    trainer.lr_scheduler.load_state_dict(checkpoint['scheduler'])
    report = {'checkpoint': str(checkpoint_path), 'checkpoint_sha256': digest(checkpoint_path),
              'checkpoint_epoch': checkpoint['epoch'], 'test_evaluated': False,
              'validation_evaluated': True, 'exact_failed_rng_replay': False,
              'max_epochs': 40, 'max_seconds': 900, 'status': 'running',
              'batches_checked': 0, 'epochs': [], 'bad_svd_backward': []}
    del checkpoint
    started = time.monotonic()
    state = {'epoch': 39, 'batch': -1, 'captures': [], 'handles': [], 'rng': None, 'interaction': None}
    original_svd, original_step = torch.linalg.svd, trainer.optimizer.step

    def finite(value):
        return bool(torch.isfinite(value).all().item())

    def clear_hooks():
        for handle in state['handles']:
            handle.remove()
        state['handles'].clear()

    def traced_svd(rep, *args, **kwargs):
        if not finite(rep):
            raise FloatingPointError('Nonfinite SVD input before decomposition')
        result = original_svd(rep, *args, **kwargs)
        if not torch.is_grad_enabled():
            return result
        singular = result[1].detach()
        gaps = singular[:-1] - singular[1:]
        record = {'call': len(state['captures']), 'shape': list(rep.shape),
                  'input_finite': True, 'outputs_finite': all(finite(value) for value in result),
                  'singular_max': singular.max().item(), 'singular_min': singular.min().item(),
                  'min_adjacent_gap': gaps.min().item(), 'equal_adjacent_singular_values': int((gaps == 0).sum().item()),
                  'band_boundary_gaps': [gaps[index].item() for index in (21, 42)]}
        capture = {'record': record, 'input': rep.detach(), 'singular': singular, 'output_gradients': {}}
        state['captures'].append(capture)
        for name, value in zip(('U', 'S', 'Vh'), result):
            if value.requires_grad:
                def observe_output(gradient, name=name, capture=capture):
                    capture['output_gradients'][name] = gradient.detach()
                    return gradient
                state['handles'].append(value.register_hook(observe_output))
        if rep.requires_grad:
            def observe_input(gradient):
                if not finite(gradient):
                    bad = dict(record, epoch=state['epoch'], batch=state['batch'],
                               input_gradient_finite=False,
                               output_gradients_finite=all(finite(value) for value in capture['output_gradients'].values()),
                               nonfinite_elements=int((~torch.isfinite(gradient)).sum().item()))
                    report['bad_svd_backward'].append(bad)
                    path = OUTPUT / ('bad_svd_e{}_b{}_call{}.pt'.format(state['epoch'], state['batch'], record['call']))
                    torch.save({'input': capture['input'].cpu(), 'singular': singular.cpu(), 'record': bad,
                                'output_gradients': {name: value.cpu() for name, value in capture['output_gradients'].items()}}, path)
                return gradient
            state['handles'].append(rep.register_hook(observe_input))
        return result

    def checked_loss(interaction):
        clear_hooks()
        state['captures'].clear()
        state['batch'] += 1
        state['interaction'] = interaction.detach().clone()
        state['rng'] = {'python': random.getstate(), 'numpy': np.random.get_state(),
                        'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}
        if time.monotonic() - started > 900:
            raise TimeoutError('Bounded numerical probe exhausted 900 seconds')
        loss = model.calculate_loss(interaction)
        if not finite(loss):
            raise FloatingPointError('Nonfinite forward loss')
        return loss

    def checked_step(*args, **kwargs):
        invalid = [name for name, value in model.named_parameters()
                   if value.grad is not None and not finite(value.grad)]
        if invalid:
            report.update(status='nonfinite_backward_identified', first_bad_epoch=state['epoch'],
                          first_bad_batch=state['batch'], nonfinite_parameter_gradients=invalid,
                          svd_calls=[capture['record'] for capture in state['captures']])
            torch.save({'model': model.state_dict(), 'optimizer': trainer.optimizer.state_dict(),
                        'interaction': state['interaction'].cpu(), 'rng': state['rng'],
                        'epoch': state['epoch'], 'batch': state['batch']}, OUTPUT / 'before_bad_step.pt')
            raise FloatingPointError('Invalid gradients blocked BEFORE optimizer.step')
        result = original_step(*args, **kwargs)
        report['batches_checked'] += 1
        invalid = [name for name, value in model.named_parameters() if not finite(value)]
        if invalid:
            report.update(status='nonfinite_optimizer_identified', nonfinite_parameters=invalid)
            raise FloatingPointError('Optimizer created nonfinite parameters from finite gradients')
        clear_hooks()
        state['captures'].clear()
        return result

    def emit(row):
        row['epoch'] += 39
        report['epochs'].append(row)
        state['epoch'], state['batch'] = row['epoch'] + 1, -1
        with (OUTPUT / 'progress.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False, default=str) + '\n')
        print(json.dumps({'epoch': row['epoch'], 'loss': row['loss_components'],
                          'batches_checked': report['batches_checked']}, allow_nan=False), flush=True)

    torch.linalg.svd, trainer.optimizer.step = traced_svd, checked_step
    try:
        training_loop(trainer, train, valid, 40, 20, checked_loss, lambda *args: None, emit)
        report['status'] = 'bounded_probe_completed_no_failure'
    except BaseException as error:
        if report['status'] == 'running':
            report['status'] = 'probe_stopped'
        report.update(error=str(error), traceback=traceback.format_exc())
    finally:
        clear_hooks()
        torch.linalg.svd, trainer.optimizer.step = original_svd, original_step
        if report['status'] == 'nonfinite_backward_identified':
            # Replay only the captured batch under anomaly detection, before
            # any bad gradient was permitted to alter parameters.
            random.setstate(state['rng']['python'])
            np.random.set_state(state['rng']['numpy'])
            torch.set_rng_state(state['rng']['torch'])
            torch.cuda.set_rng_state_all(state['rng']['cuda'])
            trainer.optimizer.zero_grad()
            try:
                with torch.autograd.detect_anomaly():
                    model.calculate_loss(state['interaction']).backward()
                report['anomaly_replay'] = 'did_not_reproduce'
            except BaseException:
                report['anomaly_replay'] = traceback.format_exc()
        report['elapsed_seconds'] = time.monotonic() - started
        report['source_checkpoint_unchanged'] = digest(checkpoint_path) == report['checkpoint_sha256']
        with (OUTPUT / 'diagnostic.json').open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False, default=str)
        print(json.dumps({key: value for key, value in report.items() if key != 'epochs'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
