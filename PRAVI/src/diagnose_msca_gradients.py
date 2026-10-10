"""Fixed-budget gradient-conflict diagnostic for official MSCA."""

from __future__ import annotations

import argparse
import json
import logging
from collections import OrderedDict
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.optim as optim

from models.msca import MSCA
from utils.configurator import Config
from utils.dataloader import TrainDataLoader
from utils.dataset import RecDataset
from utils.gradient_conflict import (
    msca_loss_components,
    probe_gradient_geometry,
    sum_loss_components,
    summarize_probe_records,
    weighted_msca_loss_components,
)
from utils.utils import init_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["baby", "sports", "elec"], default="baby")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--probes-per-epoch", type=int, default=3)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def build_config(args):
    return Config(
        "MSCA",
        args.dataset,
        {
            "gpu_id": args.gpu_id,
            "seed": args.seed,
            "epochs": args.epochs,
            "enable_visualization": False,
            "n_layers": 2,
            "fusion_coeff": 0.4,
            "cl_weight": 0.005,
            "reg_weight": 3e-7,
        },
    )


def evenly_spaced_probe_indices(batch_count, probe_count):
    if batch_count <= 0:
        raise ValueError("training data must contain at least one batch")
    probe_count = max(1, min(probe_count, batch_count))
    return set(np.linspace(0, batch_count - 1, probe_count, dtype=int).tolist())


def run_diagnostic(args):
    if args.epochs <= 0 or args.probes_per_epoch <= 0:
        raise ValueError("epochs and probes-per-epoch must be positive")

    config = build_config(args)
    init_seed(args.seed)
    dataset = RecDataset(config)
    # RecDataset populates ``inter_num`` in __str__; quick_start performs the
    # same call before constructing data loaders.
    str(dataset)
    train_dataset, _, _ = dataset.split()
    str(train_dataset)
    train_data = TrainDataLoader(
        config,
        train_dataset,
        batch_size=config["train_batch_size"],
        shuffle=True,
    )
    train_data.pretrain_setup()

    model = MSCA(config, train_data).to(config["device"])
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    schedule = config.get("learning_rate_scheduler", [1.0, 50])
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: schedule[0] ** (epoch / schedule[1]),
    )

    parameter_groups = OrderedDict(
        (
            ("user_id", [model.user_embedding.weight]),
            ("item_id", [model.item_id_embedding.weight]),
        )
    )
    probe_indices = evenly_spaced_probe_indices(
        len(train_data), args.probes_per_epoch
    )
    records = []
    equivalence_checked = False

    print(
        "DIAGNOSTIC_CONFIG "
        + json.dumps(
            {
                "dataset": args.dataset,
                "seed": args.seed,
                "epochs": args.epochs,
                "probes_per_epoch": args.probes_per_epoch,
                "batch_count": len(train_data),
                "probe_indices": sorted(probe_indices),
                "device": str(config["device"]),
                "n_layers": 2,
                "fusion_coeff": 0.4,
                "cl_weight": 0.005,
                "reg_weight": 3e-7,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for epoch in range(args.epochs):
        model.train()
        epoch_start = time()
        epoch_loss = 0.0
        for batch_index, interaction in enumerate(train_data):
            optimizer.zero_grad(set_to_none=True)
            if batch_index in probe_indices:
                raw_components = msca_loss_components(model, interaction)
                weighted_components = weighted_msca_loss_components(
                    model, raw_components
                )
                loss = sum_loss_components(weighted_components)

                if not equivalence_checked:
                    reference_loss = model.calculate_loss(interaction)
                    absolute_delta = abs(
                        float(reference_loss.detach().item())
                        - float(loss.detach().item())
                    )
                    tolerance = 1e-6 * max(
                        1.0, abs(float(reference_loss.detach().item()))
                    )
                    print(
                        "LOSS_EQUIVALENCE "
                        + json.dumps(
                            {
                                "reference": float(reference_loss.detach().item()),
                                "decomposed": float(loss.detach().item()),
                                "absolute_delta": absolute_delta,
                                "tolerance": tolerance,
                                "passed": absolute_delta <= tolerance,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    if absolute_delta > tolerance:
                        raise RuntimeError(
                            "decomposed loss does not match official MSCA"
                        )
                    equivalence_checked = True
                    del reference_loss

                term_stats = probe_gradient_geometry(
                    weighted_components, parameter_groups
                )
                for term_values in term_stats.values():
                    user_stats = term_values["user_id"]
                    item_stats = term_values["item_id"]
                    # Reconstruct the two-block statistic from sufficient
                    # statistics rather than asking autograd for duplicates.
                    dot = user_stats["dot"] + item_stats["dot"]
                    ref_sq = (
                        user_stats["reference_norm"] ** 2
                        + item_stats["reference_norm"] ** 2
                    )
                    aux_sq = (
                        user_stats["auxiliary_norm"] ** 2
                        + item_stats["auxiliary_norm"] ** 2
                    )
                    ref_norm = ref_sq**0.5
                    aux_norm = aux_sq**0.5
                    term_values["id_backbone"] = {
                        "cosine": dot / max(ref_norm * aux_norm, 1e-30),
                        "dot": dot,
                        "reference_norm": ref_norm,
                        "auxiliary_norm": aux_norm,
                        "norm_ratio": aux_norm / max(ref_norm, 1e-30),
                        "used_parameters": (
                            user_stats["used_parameters"]
                            + item_stats["used_parameters"]
                        ),
                    }
                record = {
                    "epoch": epoch + 1,
                    "batch": batch_index,
                    "terms": term_stats,
                }
                records.append(record)
                print(
                    "GRADIENT_PROBE "
                    + json.dumps(record, sort_keys=True),
                    flush=True,
                )
            else:
                loss = model.calculate_loss(interaction)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss at epoch {epoch + 1}, batch {batch_index}"
                )
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().item())

        scheduler.step()
        print(
            "DIAGNOSTIC_EPOCH "
            + json.dumps(
                {
                    "epoch": epoch + 1,
                    "loss_sum": epoch_loss,
                    "seconds": time() - epoch_start,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    summary = summarize_probe_records(records)
    summary["dataset"] = args.dataset
    summary["seed"] = args.seed
    summary["epochs"] = args.epochs
    print("GRADIENT_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return summary


def main():
    logging.basicConfig(level=logging.INFO)
    run_diagnostic(parse_args())


if __name__ == "__main__":
    main()
