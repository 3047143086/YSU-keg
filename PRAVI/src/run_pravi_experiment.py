"""Run fixed-budget MSCA/PRAVI comparisons without a large grid search."""

from __future__ import annotations

import argparse

from utils.quick_start import quick_start


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=[
            "MSCA",
            "MSCA_PRAVI",
            "MSCA_VI",
            "MSCA_SDA",
            "MSCA_HCR",
            "MSCA_UECA",
            "MSCA_SG",
            "MSCA_DCRL",
        ],
        required=True,
    )
    parser.add_argument("--dataset", choices=["baby", "sports", "elec"], default="baby")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--stopping-step", type=int, default=20)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--fusion-coeff", type=float, default=0.4)
    parser.add_argument("--cl-weight", type=float, default=0.005)
    parser.add_argument("--reg-weight", type=float, default=3e-7)
    parser.add_argument("--pravi-topk", type=int, default=5)
    parser.add_argument("--pravi-rho", type=float, default=0.1)
    parser.add_argument("--pravi-kappa", type=float, default=1.0)
    parser.add_argument("--pravi-min-confidence", type=float, default=0.0)
    parser.add_argument("--pravi-probe-topl", type=int, default=10)
    parser.add_argument("--pravi-shrinkage", type=float, default=10.0)
    parser.add_argument("--pravi-min-modal-support", type=int, choices=[1, 2], default=1)
    parser.add_argument(
        "--pravi-scope",
        choices=["collab", "collab_semantic", "first_order", "dvt"],
        default="first_order",
    )
    parser.add_argument("--pravi-response-scale", type=float, default=1.0)
    parser.add_argument("--dvt-max-swaps-per-user", type=int, default=1)
    parser.add_argument(
        "--dvt-transport",
        choices=[
            "two_switch",
            "cycle_cover",
            "net_utility_cycle",
            "profile_drift_cycle",
        ],
        default="two_switch",
    )
    parser.add_argument(
        "--dvt-weight-mode", choices=["unit", "confidence"], default="unit"
    )
    parser.add_argument("--dvt-min-arc-confidence", type=float, default=0.0)
    parser.add_argument("--dvt-min-net-margin", type=float, default=0.0)
    parser.add_argument("--dvt-min-donor-history", type=int, default=2)
    parser.add_argument(
        "--keep-virtual-negatives",
        action="store_true",
        help="Do not remove PRAVI candidate edges from sampled BPR negatives.",
    )
    parser.add_argument(
        "--reliability-mode",
        choices=["personalized", "global", "uniform"],
        default="personalized",
    )
    parser.add_argument(
        "--reliability-conditioning",
        choices=[
            "none",
            "cl_weight",
            "fusion_route",
            "attention_prior",
            "both",
            "consensus_residual",
        ],
        default="none",
        help="Train-only user reliability mechanism for semantic alignment.",
    )
    parser.add_argument(
        "--reliability-strength",
        type=float,
        default=0.5,
        help="Bounded strength of reliability-conditioned alignment.",
    )
    parser.add_argument(
        "--consensus-private-strength",
        type=float,
        default=0.25,
        help="Weight of the disagreement-gated private residual alignment.",
    )
    parser.add_argument(
        "--sda-adapter-rank",
        type=int,
        default=8,
        help="Rank of each modality-specific content adapter.",
    )
    parser.add_argument(
        "--sda-adapter-scale",
        type=float,
        default=0.10,
        help="Bounded scale of the modality-specific residual adapters.",
    )
    parser.add_argument(
        "--sda-bridge-mix",
        type=float,
        default=0.05,
        help="Agreement-gated shared content residual strength.",
    )
    parser.add_argument(
        "--sda-gate-temperature",
        type=float,
        default=4.0,
        help="Temperature of the detached cross-modal agreement gate.",
    )
    parser.add_argument(
        "--sda-gate-center",
        type=float,
        default=0.0,
        help="Center of the agreement gate.",
    )
    parser.add_argument(
        "--sda-cmsa-weight",
        type=float,
        default=0.01,
        help="Weight of the structure-aware cross-modal feature loss.",
    )
    parser.add_argument(
        "--sda-cmsa-batch",
        type=int,
        default=256,
        help="Maximum number of unique positive items used by CMSA per batch.",
    )
    parser.add_argument(
        "--sda-tau",
        type=float,
        default=0.20,
        help="Temperature for the structural teacher and cross-modal logits.",
    )
    parser.add_argument(
        "--hcr-max-scale",
        type=float,
        default=0.10,
        help="Maximum absolute scale of each LCHR modality residual.",
    )
    parser.add_argument(
        "--hcr-activity-prior",
        type=float,
        default=10.0,
        help="History-length prior controlling sparse-user residual capacity.",
    )
    parser.add_argument(
        "--hcr-coherence-temperature",
        type=float,
        default=0.25,
        help="Temperature for per-user modality reliability weights.",
    )
    parser.add_argument(
        "--hcr-min-context",
        type=int,
        default=2,
        help="Minimum remaining history after leave-one-out correction.",
    )
    parser.add_argument(
        "--disable-ueca",
        action="store_true",
        help="Run MSCA_UECA through its exact MSCA identity path.",
    )
    parser.add_argument("--sg-num-clusters", type=int, default=4)
    parser.add_argument("--sg-cluster-weight", type=float, default=0.1)
    parser.add_argument("--sg-loss-weight", type=float, default=0.001)
    parser.add_argument("--sg-random-state", type=int, default=0)
    parser.add_argument("--dcrl-weight", type=float, default=0.02)
    parser.add_argument("--dcrl-margin", type=float, default=0.10)
    parser.add_argument("--dcrl-temperature", type=float, default=0.20)
    parser.add_argument("--dcrl-min-weight", type=float, default=0.10)
    parser.add_argument("--binary-weights", action="store_true")
    parser.add_argument(
        "--vi-strategy", choices=["overlay", "synergistic"], default="overlay"
    )
    parser.add_argument("--vi-lambda", type=float, default=0.005)
    parser.add_argument("--vi-topk", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Tune on validation data without evaluating the test split.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.epochs = 2
        args.stopping_step = 1

    config = {
        "gpu_id": args.gpu_id,
        "seed": [args.seed],
        "epochs": args.epochs,
        "stopping_step": args.stopping_step,
        "enable_visualization": False,
        "evaluate_test_each_epoch": False,
        "final_test_after_training": not args.validation_only,
        "n_layers": [args.n_layers],
        "fusion_coeff": [args.fusion_coeff],
        "cl_weight": [args.cl_weight],
        "reg_weight": [args.reg_weight],
        "pravi_topk": args.pravi_topk,
        "pravi_mass_budget_ratio": args.pravi_rho,
        "pravi_uncertainty_kappa": args.pravi_kappa,
        "pravi_min_confidence": args.pravi_min_confidence,
        "pravi_probe_topl": args.pravi_probe_topl,
        "pravi_shrinkage": args.pravi_shrinkage,
        "pravi_min_modal_support": args.pravi_min_modal_support,
        "pravi_propagation_scope": args.pravi_scope,
        "pravi_response_scale": args.pravi_response_scale,
        "pravi_dvt_max_swaps_per_user": args.dvt_max_swaps_per_user,
        "pravi_dvt_transport": args.dvt_transport,
        "pravi_dvt_weight_mode": args.dvt_weight_mode,
        "pravi_dvt_min_arc_confidence": args.dvt_min_arc_confidence,
        "pravi_dvt_min_net_margin": args.dvt_min_net_margin,
        "pravi_dvt_min_donor_history": args.dvt_min_donor_history,
        "pravi_filter_virtual_negatives": not args.keep_virtual_negatives,
        "pravi_reliability_mode": args.reliability_mode,
        "pravi_reliability_conditioning": args.reliability_conditioning,
        "pravi_reliability_strength": args.reliability_strength,
        "pravi_consensus_private_strength": args.consensus_private_strength,
        "sda_enabled": args.model == "MSCA_SDA",
        "sda_adapter_rank": args.sda_adapter_rank,
        "sda_adapter_scale": args.sda_adapter_scale,
        "sda_bridge_mix": args.sda_bridge_mix,
        "sda_gate_temperature": args.sda_gate_temperature,
        "sda_gate_center": args.sda_gate_center,
        "sda_cmsa_weight": args.sda_cmsa_weight,
        "sda_cmsa_batch": args.sda_cmsa_batch,
        "sda_tau": args.sda_tau,
        "hcr_enabled": args.model == "MSCA_HCR",
        "hcr_max_scale": args.hcr_max_scale,
        "hcr_activity_prior": args.hcr_activity_prior,
        "hcr_coherence_temperature": args.hcr_coherence_temperature,
        "hcr_min_context": args.hcr_min_context,
        "ueca_enabled": args.model == "MSCA_UECA" and not args.disable_ueca,
        "sg_num_clusters": args.sg_num_clusters,
        "sg_cluster_weight": args.sg_cluster_weight,
        "sg_loss_weight": args.sg_loss_weight,
        "sg_random_state": args.sg_random_state,
        "dcrl_enabled": args.model == "MSCA_DCRL",
        "dcrl_weight": args.dcrl_weight,
        "dcrl_margin": args.dcrl_margin,
        "dcrl_temperature": args.dcrl_temperature,
        "dcrl_min_weight": args.dcrl_min_weight,
        "pravi_binary_weights": args.binary_weights,
        "vi_strategy": args.vi_strategy,
        "vi_lambda": args.vi_lambda,
        "vi_topk": args.vi_topk,
    }
    quick_start(
        model=args.model,
        dataset=args.dataset,
        config_dict=config,
        save_model=False,
    )


if __name__ == "__main__":
    main()
