"""Bounded training-only numerical probe from the last valid checkpoint.

This is not a formal experiment or an exact replay: the failed training did
not save its RNG state. All optimizer changes are confined to this process.
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

SOURCE = Path('/data/sunzixuan/decoupled_gate_20261008_v1')
OUTPUT = Path(__file__).resolve().parent


def main():
    gpu = os.environ['CUDA_VISIBLE_DEVICES']
    os.chdir(SOURCE / 'carrier/src')
    sys.path[:0] = [str(SOURCE), str(SOURCE / 'carrier/src')]
    import torch
    from common.trainer import Trainer
    from proposal import DecoupledGateBRIDGE
    from train import scalarize, split_train_valid, digest
    from utils.configurator import Config
    from utils.dataloader import TrainDataLoader
    from utils.dataset import RecDataset
    from utils.utils import init_seed

    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.55, device=0)
    config = Config('BRIDGE', 'baby', {'use_gpu': True, 'gpu_id': gpu,
                    'seed': [2026], 'epochs': 300, 'stopping_step': 20,
                    'eval_batch_size': 64, 'save_recommended_topk': False,
                    'req_training': True, 'valid_metric': 'Recall@20'})
    scalarize(config)
    config.final_config_dict.update(json.loads((SOURCE / 'configs/baby_edge.json').read_text()))
    init_seed(2026)
    dataset = RecDataset(config)
    train_dataset, _ = split_train_valid(dataset, config)
    train = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    train.pretrain_setup()
    model = DecoupledGateBRIDGE(config, train).to(config['device'])
    trainer = Trainer(config, model)
    checkpoint_path = SOURCE / 'runs/baby_edge_s2026/best.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    # The carrier registers these aliases on its first encoder forward. Bind
    # only verified aliases, after optimizer construction preserves its order.
    for alias, nested in (('id_preference', 'id_gcn.preference'),
                          ('v_preference', 'v_gcn.preference'),
                          ('t_preference', 't_gcn.preference')):
        if not torch.equal(checkpoint['model'][alias], checkpoint['model'][nested]):
            raise RuntimeError('Checkpoint preference alias differs: ' + alias)
        module_name, field = nested.split('.')
        setattr(model, alias, getattr(getattr(model, module_name), field))
    model.load_state_dict(checkpoint['model'], strict=True)
    trainer.optimizer.load_state_dict(checkpoint['optimizer'])
    report = {'checkpoint': str(checkpoint_path), 'checkpoint_sha256': digest(checkpoint_path),
              'checkpoint_epoch': checkpoint['epoch'], 'test_evaluated': False,
              'validation_evaluated': False, 'exact_failed_rng_replay': False,
              'max_batches': 58, 'max_seconds': 180,
              'status': 'running', 'steps': [], 'nonfinite_svd_gradients': []}
    del checkpoint
    invalid_parameters = [name for name, value in model.named_parameters()
                          if not torch.isfinite(value).all().item()]
    if invalid_parameters:
        raise FloatingPointError('Checkpoint parameters are already nonfinite: ' + str(invalid_parameters))

    original_svd = torch.linalg.svd
    captures = []
    handles = []
    batch_index = -1

    def traced_svd(rep, *args, **kwargs):
        if not torch.isfinite(rep).all().item():
            raise FloatingPointError('Nonfinite input reached SVD')
        result = original_svd(rep, *args, **kwargs)
        singular = result[1].detach()
        gaps = singular[:-1] - singular[1:]
        record = {'call': len(captures), 'shape': list(rep.shape),
                  'singular_max': singular.max().item(), 'singular_min': singular.min().item(),
                  'min_adjacent_gap': gaps.min().item(),
                  'equal_adjacent_singular_values': int((gaps == 0).sum().item()),
                  'outputs_finite': all(torch.isfinite(value).all().item() for value in result)}
        captures.append(record)
        if rep.requires_grad:
            captured_input = rep.detach().clone()

            def observe_gradient(gradient):
                if not torch.isfinite(gradient).all().item():
                    failure = dict(record, batch=batch_index,
                                   nonfinite_gradient_elements=int((~torch.isfinite(gradient)).sum().item()))
                    report['nonfinite_svd_gradients'].append(failure)
                    torch.save({'input': captured_input.cpu(), 'record': failure},
                               OUTPUT / ('nonfinite_svd_batch{}_call{}.pt'.format(batch_index, record['call'])))
                return gradient

            handles.append(rep.register_hook(observe_gradient))
        return result

    started = time.monotonic()
    model.train()
    torch.linalg.svd = traced_svd
    try:
        for batch_index, interaction in enumerate(train):
            if batch_index >= 58 or time.monotonic() - started >= 180:
                break
            captures.clear()
            trainer.optimizer.zero_grad()
            loss = model.calculate_loss(interaction)
            if not torch.isfinite(loss).item():
                raise FloatingPointError('Nonfinite forward loss in diagnostic batch ' + str(batch_index))
            loss.backward()
            for handle in handles:
                handle.remove()
            handles.clear()
            nonfinite = [name for name, value in model.named_parameters()
                         if value.grad is not None and not torch.isfinite(value.grad).all().item()]
            row = {'batch': batch_index, 'loss': loss.item(), 'svd_calls': list(captures),
                   'nonfinite_parameter_gradients': nonfinite}
            report['steps'].append(row)
            if nonfinite:
                report['status'] = 'nonfinite_backward_identified'
                break
            if trainer.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), **trainer.clip_grad_norm)
            trainer.optimizer.step()
            print(json.dumps({'batch': batch_index, 'loss': loss.item()}), flush=True)
        else:
            report['status'] = 'bounded_probe_completed_no_failure'
        if report['status'] == 'running':
            report['status'] = 'bounded_probe_completed_no_failure'
    except BaseException as error:
        report.update(status='diagnostic_failure', error=str(error), traceback=traceback.format_exc())
    finally:
        torch.linalg.svd = original_svd
        for handle in handles:
            handle.remove()
        report['elapsed_seconds'] = time.monotonic() - started
        report['source_checkpoint_unchanged'] = digest(checkpoint_path) == report['checkpoint_sha256']
        with (OUTPUT / 'diagnostic.json').open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        print(json.dumps({key: value for key, value in report.items() if key != 'steps'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
