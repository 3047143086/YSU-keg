"""Screen CF-LCB graph configurations against validation interactions only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from utils.pravi import build_pravi_interactions, torch_graph_to_scipy


def parse_config(value: str):
    fields = value.split(":")
    if len(fields) not in {4, 5}:
        raise ValueError(
            "config must be topk:shrinkage:kappa:min_modal_support"
            "[:min_confidence]"
        )
    topk, shrinkage, kappa, support = fields[:4]
    min_confidence = fields[4] if len(fields) == 5 else 0.0
    return (
        int(topk),
        float(shrinkage),
        float(kappa),
        int(support),
        float(min_confidence),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="baby")
    parser.add_argument("--data-path", default="./data")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["1:10:1:1", "2:10:1:1", "2:3:1:1", "2:1:1:1", "2:10:2:1", "2:10:1:2"],
        help="topk:shrinkage:kappa:min_modal_support[:min_confidence]",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.data_path) / args.dataset
    interactions = np.loadtxt(
        dataset_dir / f"{args.dataset}.inter",
        skiprows=1,
        usecols=(0, 1, 4),
        dtype=np.int64,
    )
    n_users = int(interactions[:, 0].max()) + 1
    n_items = int(interactions[:, 1].max()) + 1
    shape = (n_users, n_items)

    train_rows = interactions[interactions[:, 2] == 0]
    valid_rows = interactions[interactions[:, 2] == 1]
    train = sp.coo_matrix(
        (np.ones(len(train_rows)), (train_rows[:, 0], train_rows[:, 1])),
        shape=shape,
    ).tocsr()
    valid_keys = set((valid_rows[:, 0] * n_items + valid_rows[:, 1]).tolist())

    cache_dir = dataset_dir / "cache" / "MSCA"
    graph_paths = [
        cache_dir / "image_adj_10_True.pt",
        cache_dir / "text_adj_10_True.pt",
    ]
    graphs = [
        torch_graph_to_scipy(torch.load(path, map_location="cpu"))
        for path in graph_paths
    ]

    rows = []
    for raw_config in args.configs:
        topk, shrinkage, kappa, support, min_confidence = parse_config(raw_config)
        virtual, stats = build_pravi_interactions(
            train,
            graphs,
            topk=topk,
            shrinkage=shrinkage,
            uncertainty_kappa=kappa,
            mass_budget_ratio=0.1,
            min_confidence=min_confidence,
            min_modal_support=support,
        )
        coo = virtual.tocoo()
        keys = coo.row.astype(np.int64) * n_items + coo.col
        hit_mask = np.fromiter(
            (int(key) in valid_keys for key in keys), dtype=bool, count=len(keys)
        )
        hits = int(hit_mask.sum())
        rows.append(
            {
                "config": raw_config,
                "topk": topk,
                "shrinkage": shrinkage,
                "kappa": kappa,
                "min_modal_support": support,
                "min_confidence": min_confidence,
                "edges": int(virtual.nnz),
                "mass_ratio": stats["virtual_to_train_mass"],
                "users": stats["users_with_virtual_edges"],
                "lcb_rejections": stats["lcb_rejections"],
                "valid_hits": hits,
                "edge_precision": float(hits / max(virtual.nnz, 1)),
                "valid_coverage": float(hits / max(len(valid_rows), 1)),
                "mean_hit_weight": (
                    float(coo.data[hit_mask].mean()) if hits else 0.0
                ),
                "mean_weight": float(coo.data.mean()) if virtual.nnz else 0.0,
                "global_reliabilities": stats["global_reliabilities"],
                "mean_posterior_uncertainty": stats[
                    "mean_posterior_uncertainty"
                ],
            }
        )
        print(json.dumps(rows[-1], sort_keys=True))

    rows.sort(key=lambda row: (row["valid_hits"], row["edge_precision"]), reverse=True)
    print("VALIDATION_RANKING")
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
