"""One-shot remote execution wrapper for the frozen CPU-tested precheck."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIGINAL = "/data/sunzixuan/interest_negative_20261010_v1"
RUN_ROOT = "/data/sunzixuan/feature_perturbation_precheck_20261010_v4"
PYTHON = "/data/sunzixuan/miniconda3/envs/MMRec/bin/python"

REMOTE = r'''
import hashlib, json, math, os, random, sys, time, traceback
from pathlib import Path

# Set visibility before importing torch: logical cuda:0 is physical GPU 1.
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "4"
import numpy as np
import torch
torch.set_num_threads(4)
ROOT = Path("__ORIGINAL__")
RUN = Path("__RUN_ROOT__")
START = time.monotonic()
EPSILON = 0.01

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def write_json(name, value):
    path = RUN / name
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(str(pending), str(path))

def state_sha(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def graph_sha(model):
    result = {}
    for name in ("norm_R", "norm_adj", "struct_original_adj", "image_original_adj", "text_original_adj"):
        value = getattr(model, name).detach()
        h = hashlib.sha256(); h.update(str(value.shape).encode()); h.update(str(value.layout).encode())
        if value.is_sparse:
            value = value.coalesce()
            h.update(value.indices().cpu().contiguous().numpy().tobytes())
            h.update(value.values().cpu().contiguous().numpy().tobytes())
        else:
            h.update(value.cpu().contiguous().numpy().tobytes())
        result[name] = h.hexdigest()
    return result

def rng_sha():
    h = hashlib.sha256(); h.update(repr(random.getstate()).encode())
    state = np.random.get_state(); h.update(state[0].encode()); h.update(state[1].tobytes())
    h.update(repr(state[2:]).encode()); h.update(torch.get_rng_state().numpy().tobytes())
    for value in torch.cuda.get_rng_state_all(): h.update(value.cpu().numpy().tobytes())
    return h.hexdigest()

def official_bpr(model, batch):
    original = model.cal_bpr_loss
    captured = []
    def capture(*args, **kwargs):
        value = original(*args, **kwargs); captured.append(value); return value
    present = "cal_bpr_loss" in model.__dict__
    prior = model.__dict__.get("cal_bpr_loss")
    model.cal_bpr_loss = capture
    try:
        model.calculate_loss(batch)
    finally:
        if present: model.cal_bpr_loss = prior
        else: del model.cal_bpr_loss
    if len(captured) != 1: raise RuntimeError("official BPR hook call count")
    value = captured[0]
    if not bool(torch.isfinite(value).item()): raise FloatingPointError("nonfinite official BPR")
    return value

def functional_bpr(model, image_weight, batch):
    """Exact MSCA forward with only image_embedding.weight replaced."""
    image_item = model.item_id_embedding.weight * model.gate_v(model.image_trs(image_weight))
    text_item = model.item_id_embedding.weight * model.gate_t(model.text_trs(model.text_embedding.weight))
    ego = torch.cat([model.user_embedding.weight, model.item_id_embedding.weight], dim=0)
    all_embeddings = []
    for _ in range(model.n_layers):
        ego = torch.sparse.mm(model.norm_adj, ego); all_embeddings.append(ego)
    collab = torch.mean(torch.stack(all_embeddings, dim=1), dim=1, keepdim=False)
    struct = model.semantic_encode(model.struct_original_adj, model.item_id_embedding.weight)
    image = model.semantic_encode(model.image_original_adj, image_item)
    text = model.semantic_encode(model.text_original_adj, text_item)
    views = [image, text, struct]
    weights = model.softmax(torch.cat([model.query_common(view) for view in views], dim=-1))
    redundant = sum(weight.unsqueeze(1) * view for weight, view in zip(weights.unbind(dim=1), views))
    final = collab + model.fusion_coeff * (sum(views) - redundant)
    users, items = torch.split(final, [model.n_users, model.n_items], dim=0)
    return model.cal_bpr_loss(users[batch[0]], items[batch[1]], items[batch[2]])

def row_replace(base, rows, delta):
    return base.index_copy(0, rows, base.index_select(0, rows) + delta)

def attack_delta(rows, grads):
    feature_norm = rows.norm(dim=1, keepdim=True)
    grad_norm = grads.norm(dim=1, keepdim=True)
    zero_features = int((feature_norm <= 0).sum().item())
    zero_gradients = int((grad_norm <= 0).sum().item())
    counts = {"zero_feature_rows": zero_features,
              "zero_gradient_rows": zero_gradients,
              "nonfinite_feature_norms": int((~torch.isfinite(feature_norm)).sum().item()),
              "nonfinite_gradient_norms": int((~torch.isfinite(grad_norm)).sum().item())}
    if any(counts.values()):
        return None, counts
    return EPSILON * feature_norm * grads / grad_norm, counts

def random_delta(rows, batch_id):
    generator = torch.Generator(device="cpu"); generator.manual_seed(20261030 + batch_id)
    noise = torch.randn(rows.shape, generator=generator, device="cpu", dtype=rows.dtype).to(rows.device)
    noise_norm = noise.norm(dim=1, keepdim=True)
    if bool((noise_norm <= 0).any().item()): raise RuntimeError("zero random direction row")
    return EPSILON * rows.norm(dim=1, keepdim=True) * noise / noise_norm

def run():
    if RUN.exists(): raise RuntimeError("run root exists; refusing relaunch")
    RUN.mkdir(parents=True)
    os.chdir(ROOT / "carrier" / "src")
    sys.path[:0] = [str(ROOT), str(ROOT / "carrier" / "src")]
    from models.msca import MSCA
    from utils.configurator import Config
    from utils.dataloader import TrainDataLoader
    from utils.dataset import RecDataset
    from utils.utils import init_seed
    config_saved = json.loads((ROOT / "runs/baby_uniform_s2026_formal/config.json").read_text())
    config = Config("MSCA", "baby", config_saved.copy(), mg=False)
    if str(config["device"]) != "cuda" or config["inter_splitting_label"] != "x_label":
        raise RuntimeError("unexpected frozen config")
    dataset = RecDataset(config)
    train_df = dataset.df.loc[dataset.df["x_label"] == 0].drop(columns=["x_label"]).copy()
    train_data = dataset.copy(train_df); str(train_data)
    loader = TrainDataLoader(config, train_data, batch_size=2048, shuffle=True)
    archive = Path("/data/sunzixuan/curvature_diagnostic_20261010_v1/data/batches.npz")
    if not archive.is_file(): raise RuntimeError("private fixed batch archive missing")
    loaded = np.load(archive, allow_pickle=False)
    arrays = loaded["batches"]
    # Existing curvature archive stores each batch as (users, positives,
    # negatives), hence the public array shape is (3, 3, 2048).
    if arrays.shape != (3, 3, 2048): raise RuntimeError("fixed batch shape mismatch")
    receipt = {"path": str(archive), "sha256": digest(archive), "shape": list(arrays.shape),
               "batch_hashes": [hashlib.sha256(row.tobytes()).hexdigest() for row in arrays],
               "private_ids": True, "TRAIN_only": True}
    write_json("batch_receipt.json", receipt)
    init_seed(2026); loader.pretrain_setup()
    model = MSCA(config, loader).to(config["device"]); model.eval()
    if model.text_trs.in_features != model.text_embedding.weight.shape[1]:
        raise RuntimeError("official feature transform shape mismatch: text_trs_in=" +
                           str(model.text_trs.in_features) + ", text_weight=" +
                           str(tuple(model.text_embedding.weight.shape)))
    model_sha_before = state_sha(model); graph_before = graph_sha(model); rng_before = rng_sha()
    records = []
    batches = [torch.from_numpy(row).long().to(config["device"]) for row in arrays]
    for batch_id, batch in enumerate(batches):
        before = state_sha(model); graph = graph_sha(model)
        clean_official = official_bpr(model, batch)
        features = model.image_embedding.weight.detach().clone().requires_grad_(True)
        clean_functional = functional_bpr(model, features, batch)
        parity = abs(float(clean_official.item()) - float(clean_functional.item()))
        parity_floor = 1e-5 * max(1.0, abs(float(clean_official.item())))
        if parity > parity_floor: raise RuntimeError("clean BPR wrapper parity failure")
        rows = torch.unique(torch.cat((batch[1], batch[2]))).long()
        gradient = torch.autograd.grad(clean_functional, features, retain_graph=False, create_graph=False)[0]
        selected = features.detach().index_select(0, rows)
        targeted, row_checks = attack_delta(selected, gradient.detach().index_select(0, rows))
        if targeted is None:
            result = {"schema_version": 1, "status": "blocked", "dataset": "baby",
                      "source_sha256": digest(ROOT / "carrier/src/models/msca.py"),
                      "model_initial_state_sha256": model_sha_before, "graph_sha256": graph_before,
                      "records": [{"batch_id": batch_id, "batch_hash": receipt["batch_hashes"][batch_id],
                                   "selected_rows": int(rows.numel()),
                                   "clean_official": clean_official.item(),
                                   "clean_functional": clean_functional.item(),
                                   "parity_abs": parity, "parity_floor": parity_floor,
                                   "row_checks": row_checks, "attack_evaluated": False}],
                      "targeted_gt_random_count": 0, "optimizer_instantiated": False,
                      "optimizer_steps": 0, "validation_or_test": False,
                      "state_restored": state_sha(model) == model_sha_before,
                      "global_rng_unchanged": rng_sha() == rng_before,
                      "reason": "invalid feature or image-BPR gradient row",
                      "elapsed_seconds": time.monotonic() - START}
            write_json("precheck.json", result)
            print("PRECHECK_RESULT=" + json.dumps(result, allow_nan=False), flush=True)
            return
        random_null = random_delta(selected, batch_id)
        with torch.no_grad():
            target_features = row_replace(features.detach(), rows, targeted)
            random_features = row_replace(features.detach(), rows, random_null)
            target_bpr = functional_bpr(model, target_features, batch)
            random_bpr = functional_bpr(model, random_features, batch)
        clean = float(clean_functional.item()); target = float(target_bpr.item()); random_value = float(random_bpr.item())
        target_gap = target - clean; random_gap = random_value - clean
        if not all(math.isfinite(x) for x in (clean, target, random_value, target_gap, random_gap)):
            raise FloatingPointError("nonfinite precheck value")
        after = state_sha(model); after_graph = graph_sha(model); rng_after = rng_sha()
        if before != after or graph != after_graph or rng_before != rng_after:
            raise RuntimeError("state, graph, or global RNG changed")
        records.append({"batch_id": batch_id, "batch_hash": receipt["batch_hashes"][batch_id],
                       "selected_rows": int(rows.numel()), "clean_official": clean_official.item(),
                       "clean_functional": clean, "parity_abs": parity, "parity_floor": parity_floor,
                       "target_bpr": target, "random_bpr": random_value, "target_gap": target_gap,
                       "random_gap": random_gap, "targeted_gt_random": bool(target_gap > random_gap),
                       "row_checks": row_checks,
                       "epsilon_rel": EPSILON})
    if model_sha_before != state_sha(model) or graph_before != graph_sha(model) or rng_before != rng_sha():
        raise RuntimeError("final restoration mismatch")
    result = {"schema_version": 1, "status": "eligible_precheck" if sum(r["targeted_gt_random"] for r in records) >= 2 else "diagnostic_fail",
              "dataset": "baby", "source_sha256": digest(ROOT / "carrier/src/models/msca.py"),
              "model_initial_state_sha256": model_sha_before, "graph_sha256": graph_before,
              "records": records, "targeted_gt_random_count": sum(r["targeted_gt_random"] for r in records),
              "optimizer_instantiated": False, "optimizer_steps": 0, "validation_or_test": False,
              "state_restored": True, "global_rng_unchanged": True,
              "elapsed_seconds": time.monotonic() - START}
    write_json("precheck.json", result)
    print("PRECHECK_RESULT=" + json.dumps(result, allow_nan=False), flush=True)

try:
    run()
except Exception as error:
    failure = {"status": "technical_failure", "error_type": type(error).__name__,
               "error": str(error), "traceback": traceback.format_exc(),
               "elapsed_seconds": time.monotonic() - START}
    if RUN.is_dir(): write_json("failure.json", failure)
    print("PRECHECK_RESULT=" + json.dumps(failure), flush=True)
    raise
'''


def remote_source() -> str:
    return REMOTE.replace("__ORIGINAL__", ORIGINAL).replace("__RUN_ROOT__", RUN_ROOT)


def invoke(timeout: int):
    sys.path.insert(0, str(HERE.parent / "20260905"))
    from remote_session import Session
    connection = Session("213")
    try:
        return connection.run_python(remote_source(), PYTHON, timeout)
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("compile", "run"))
    args = parser.parse_args()
    source = remote_source()
    compile(source, "<remote-precheck>", "exec")
    if args.action == "compile":
        print("remote precheck syntax PASS")
        return
    code, output = invoke(3600)
    marker = "PRECHECK_RESULT="
    start = output.find(marker)
    if start < 0:
        raise RuntimeError("remote result marker missing: " + output[-2000:])
    value, _ = json.JSONDecoder().raw_decode(output[start + len(marker):])
    destination = HERE / ("precheck.json" if code == 0 else "failure.json")
    destination.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value.get(key) for key in ("status", "error", "elapsed_seconds", "targeted_gt_random_count")}, indent=2))
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
