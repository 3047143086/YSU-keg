"""Loss decomposition and gradient diagnostics for the official MSCA model."""

from __future__ import annotations

from collections import OrderedDict
from statistics import fmean, median

import torch
import torch.nn.functional as F


ALIGNMENT_COMPONENTS = (
    "align_user_image",
    "align_user_text",
    "align_item_image",
    "align_item_text",
    "align_user_structure",
    "align_item_structure",
)


def msca_loss_components(model, interaction):
    """Return the nine raw terms whose weighted sum is official MSCA loss.

    This interface is intentionally external to ``MSCA.calculate_loss`` so the
    frozen baseline path is untouched. It is valid for official MSCA and for a
    future subclass that preserves all of these loss definitions.
    """
    users, pos_items, neg_items = interaction[:3]
    (
        final_user_embeds,
        final_item_embeds,
        collab_embeds,
        struct_embeds,
        image_embeds,
        text_embeds,
    ) = model.forward(test=False)

    batch_users = final_user_embeds[users]
    pos_scores = torch.sum(batch_users * final_item_embeds[pos_items], dim=1)
    neg_scores = torch.sum(batch_users * final_item_embeds[neg_items], dim=1)
    pos_scores, neg_scores = model._refine_pairwise_scores(
        pos_scores, neg_scores, users, pos_items, neg_items
    )
    rank_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

    collab_user, collab_item = torch.split(
        collab_embeds, [model.n_users, model.n_items], dim=0
    )
    struct_user, struct_item = torch.split(
        struct_embeds, [model.n_users, model.n_items], dim=0
    )
    image_user, image_item = torch.split(
        image_embeds, [model.n_users, model.n_items], dim=0
    )
    text_user, text_item = torch.split(
        text_embeds, [model.n_users, model.n_items], dim=0
    )

    return OrderedDict(
        (
            ("rank", rank_loss),
            (
                "align_user_image",
                model.cal_cl_loss(
                    collab_user[users],
                    image_user[users],
                    model.tau,
                    sample_weight=model._get_cl_sample_weight(users, 0, True),
                ),
            ),
            (
                "align_user_text",
                model.cal_cl_loss(
                    collab_user[users],
                    text_user[users],
                    model.tau,
                    sample_weight=model._get_cl_sample_weight(users, 1, True),
                ),
            ),
            (
                "align_item_image",
                model.cal_cl_loss(
                    collab_item[pos_items],
                    image_item[pos_items],
                    model.tau,
                    sample_weight=model._get_cl_sample_weight(
                        pos_items, 0, False
                    ),
                ),
            ),
            (
                "align_item_text",
                model.cal_cl_loss(
                    collab_item[pos_items],
                    text_item[pos_items],
                    model.tau,
                    sample_weight=model._get_cl_sample_weight(
                        pos_items, 1, False
                    ),
                ),
            ),
            (
                "align_user_structure",
                model.cal_cl_loss(
                    collab_user[users], struct_user[users], model.tau
                ),
            ),
            (
                "align_item_structure",
                model.cal_cl_loss(
                    collab_item[pos_items], struct_item[pos_items], model.tau
                ),
            ),
            ("regularization", model.cal_reg_loss()),
            ("feature_learning", model._get_feature_learning_loss(pos_items)),
        )
    )


def weighted_msca_loss_components(model, components):
    """Apply the exact scalar coefficients used by official MSCA."""
    weighted = OrderedDict()
    weighted["rank"] = components["rank"]
    for name in ALIGNMENT_COMPONENTS:
        weighted[name] = model.cl_weight * components[name]
    weighted["regularization"] = (
        model.reg_weight * components["regularization"]
    )
    weighted["feature_learning"] = components["feature_learning"]
    return weighted


def sum_loss_components(components):
    values = iter(components.values())
    total = next(values)
    for value in values:
        total = total + value
    return total


def loss_gradients(loss, parameters, retain_graph=True):
    return torch.autograd.grad(
        loss,
        tuple(parameters),
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )


def gradient_statistics(reference_gradients, auxiliary_gradients, indices=None):
    """Measure direction and relative size without concatenating gradients."""
    if indices is None:
        indices = range(len(reference_gradients))

    dot = 0.0
    reference_sq_norm = 0.0
    auxiliary_sq_norm = 0.0
    used_parameters = 0
    for index in indices:
        reference = reference_gradients[index]
        auxiliary = auxiliary_gradients[index]
        if reference is None or auxiliary is None:
            continue
        reference = reference.detach()
        auxiliary = auxiliary.detach()
        dot += float(torch.sum(reference * auxiliary).item())
        reference_sq_norm += float(torch.sum(reference * reference).item())
        auxiliary_sq_norm += float(torch.sum(auxiliary * auxiliary).item())
        used_parameters += 1

    eps = 1e-30
    reference_norm = reference_sq_norm**0.5
    auxiliary_norm = auxiliary_sq_norm**0.5
    cosine = dot / max(reference_norm * auxiliary_norm, eps)
    norm_ratio = auxiliary_norm / max(reference_norm, eps)
    return {
        "cosine": cosine,
        "dot": dot,
        "reference_norm": reference_norm,
        "auxiliary_norm": auxiliary_norm,
        "norm_ratio": norm_ratio,
        "used_parameters": used_parameters,
    }


def probe_gradient_geometry(weighted_components, parameter_groups):
    """Compare every weighted alignment gradient with the BPR gradient."""
    parameters = []
    group_indices = OrderedDict()
    for group_name, group_parameters in parameter_groups.items():
        start = len(parameters)
        parameters.extend(group_parameters)
        group_indices[group_name] = tuple(range(start, len(parameters)))

    rank_gradients = loss_gradients(
        weighted_components["rank"], parameters, retain_graph=True
    )
    terms = OrderedDict()
    for term_name in ALIGNMENT_COMPONENTS:
        auxiliary_gradients = loss_gradients(
            weighted_components[term_name], parameters, retain_graph=True
        )
        terms[term_name] = {
            group_name: gradient_statistics(
                rank_gradients, auxiliary_gradients, indices
            )
            for group_name, indices in group_indices.items()
        }

    alignment_total = sum_loss_components(
        OrderedDict(
            (name, weighted_components[name]) for name in ALIGNMENT_COMPONENTS
        )
    )
    total_gradients = loss_gradients(
        alignment_total, parameters, retain_graph=True
    )
    terms["alignment_total"] = {
        group_name: gradient_statistics(rank_gradients, total_gradients, indices)
        for group_name, indices in group_indices.items()
    }
    return terms


def summarize_probe_records(
    records,
    group_name="id_backbone",
    cosine_threshold=-0.05,
    min_conflict_fraction=0.25,
    min_norm_ratio=0.01,
    min_negative_work_fraction=0.10,
    min_material_terms=2,
):
    """Aggregate probes and apply the preregistered RAPA diagnostic gate."""
    if not records:
        raise ValueError("at least one gradient probe is required")

    term_names = tuple(records[0]["terms"].keys())
    term_summary = OrderedDict()
    for term_name in term_names:
        observations = [record["terms"][term_name][group_name] for record in records]
        cosines = [item["cosine"] for item in observations]
        norm_ratios = [item["norm_ratio"] for item in observations]
        dots = [item["dot"] for item in observations]
        negative_work = sum(max(0.0, -value) for value in dots)
        absolute_work = sum(abs(value) for value in dots)
        term_summary[term_name] = {
            "mean_cosine": fmean(cosines),
            "median_cosine": median(cosines),
            "min_cosine": min(cosines),
            "conflict_fraction": sum(
                value < cosine_threshold for value in cosines
            )
            / len(cosines),
            "median_norm_ratio": median(norm_ratios),
            "negative_work_fraction": negative_work
            / max(absolute_work, 1e-30),
        }

    material_terms = []
    for term_name in ALIGNMENT_COMPONENTS:
        values = term_summary[term_name]
        if (
            values["conflict_fraction"] >= min_conflict_fraction
            and values["median_norm_ratio"] >= min_norm_ratio
            and values["negative_work_fraction"]
            >= min_negative_work_fraction
        ):
            material_terms.append(term_name)

    aggregate = term_summary["alignment_total"]
    aggregate_material = (
        aggregate["conflict_fraction"] >= 0.20
        and aggregate["median_norm_ratio"] >= 0.03
        and aggregate["negative_work_fraction"] >= min_negative_work_fraction
    )
    passed = len(material_terms) >= min_material_terms or aggregate_material
    return {
        "probe_count": len(records),
        "group": group_name,
        "terms": term_summary,
        "gate": {
            "passed": passed,
            "material_terms": material_terms,
            "aggregate_material": aggregate_material,
            "cosine_threshold": cosine_threshold,
            "min_conflict_fraction": min_conflict_fraction,
            "min_norm_ratio": min_norm_ratio,
            "min_negative_work_fraction": min_negative_work_fraction,
            "min_material_terms": min_material_terms,
        },
    }
