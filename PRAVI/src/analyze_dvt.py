"""Measure DVT transport coverage and validation-edge recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from utils.dvt import (
    build_degree_preserving_transport,
    build_evidence_cycle_transport,
)
from utils.pravi import (
    build_pravi_interactions,
    build_pravi_transport_evidence,
    torch_graph_to_scipy,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="baby")
    parser.add_argument("--data-path", default="./data")
    parser.add_argument(
        "--topk",
        "--topks",
        dest="topks",
        nargs="+",
        type=int,
        default=[2, 5, 10, 20],
    )
    parser.add_argument("--shrinkage", type=float, default=10.0)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--min-modal-support", type=int, default=1)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument(
        "--method",
        choices=(
            "two_switch",
            "cycle_cover",
            "net_utility_cycle",
            "profile_drift_cycle",
        ),
        default="two_switch",
    )
    parser.add_argument("--max-swaps-per-user", type=int, default=1)
    parser.add_argument("--cycle-min-confidence", type=float, default=0.0)
    parser.add_argument("--min-net-margin", type=float, default=0.0)
    parser.add_argument("--min-donor-history", type=int, default=2)
    args = parser.parse_args()

    dataset_dir = Path(args.data_path) / args.dataset
    interactions = np.atleast_2d(
        np.loadtxt(
            dataset_dir / f"{args.dataset}.inter",
            skiprows=1,
            usecols=(0, 1, 4),
            dtype=np.int64,
        )
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

    results = []
    for topk in args.topks:
        donor_upper_bounds = None
        if args.method in {"net_utility_cycle", "profile_drift_cycle"}:
            candidates, donor_upper_bounds, pravi_stats = (
                build_pravi_transport_evidence(
                    train,
                    graphs,
                    topk=topk,
                    shrinkage=args.shrinkage,
                    uncertainty_kappa=args.kappa,
                    min_confidence=args.min_confidence,
                    min_modal_support=args.min_modal_support,
                    min_donor_history=args.min_donor_history,
                    donor_cost_mode=(
                        "profile_drift"
                        if args.method == "profile_drift_cycle"
                        else "loo_retrieval"
                    ),
                )
            )
        else:
            candidates, pravi_stats = build_pravi_interactions(
                train,
                graphs,
                topk=topk,
                shrinkage=args.shrinkage,
                uncertainty_kappa=args.kappa,
                mass_budget_ratio=0.1,
                min_confidence=args.min_confidence,
                min_modal_support=args.min_modal_support,
            )
        if args.method in {
            "cycle_cover",
            "net_utility_cycle",
            "profile_drift_cycle",
        }:
            delta, dvt_stats = build_evidence_cycle_transport(
                train,
                candidates,
                min_arc_confidence=args.cycle_min_confidence,
                donor_upper_bounds=donor_upper_bounds,
                min_net_margin=args.min_net_margin,
            )
        else:
            delta, dvt_stats = build_degree_preserving_transport(
                train,
                candidates,
                max_swaps_per_user=args.max_swaps_per_user,
            )
        positive = delta.tocoo()
        positive_mask = positive.data > 0.0
        positive_keys = (
            positive.row[positive_mask].astype(np.int64) * n_items
            + positive.col[positive_mask]
        )
        hit_mask = np.fromiter(
            (int(key) in valid_keys for key in positive_keys),
            dtype=bool,
            count=len(positive_keys),
        )
        hits = int(hit_mask.sum())
        positive_edges = int(positive_mask.sum())
        result = {
            "topk": int(topk),
            "method": args.method,
            "candidate_edges": int(candidates.nnz),
            "swaps": dvt_stats.get("swaps", 0),
            "cycles": dvt_stats.get("cycles", 0),
            "transport_arcs": dvt_stats.get("transport_arcs", 0),
            "cycle_length_mean": dvt_stats.get("cycle_length_mean", 0.0),
            "cycle_length_max": dvt_stats.get("cycle_length_max", 0),
            "cycle_length_histogram": dvt_stats.get(
                "cycle_length_histogram", {}
            ),
            "candidate_graph_arcs": dvt_stats.get("candidate_graph_arcs", 0),
            "considered_owner_arcs": dvt_stats.get(
                "considered_owner_arcs", 0
            ),
            "donor_rejected_owner_arcs": dvt_stats.get(
                "donor_rejected_owner_arcs", 0
            ),
            "users": dvt_stats["users"],
            "user_coverage": dvt_stats["user_coverage"],
            "dvt_candidate_coverage": dvt_stats["candidate_coverage"],
            "row_residual_max": dvt_stats["row_residual_max"],
            "column_residual_max": dvt_stats["column_residual_max"],
            "positive_edges": positive_edges,
            "positive_valid_hits": hits,
            "positive_valid_precision": float(hits / max(positive_edges, 1)),
            "positive_valid_coverage": float(hits / max(len(valid_rows), 1)),
            "confidence": dvt_stats["confidence"],
            "selected_donor_upper_bound": dvt_stats.get(
                "donor_upper_bound", {}
            ),
            "selected_net_margin": dvt_stats.get("net_margin", {}),
            "training_net_utility": dvt_stats.get("net_margin", {}).get(
                "sum", 0.0
            ),
            "donor_evidence": pravi_stats.get("donor_upper_bound", {}),
            "pravi_mean_confidence": pravi_stats["mean_confidence"],
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True))

    print("DVT_RESULTS")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
