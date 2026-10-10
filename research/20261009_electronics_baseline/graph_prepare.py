"""Build the original BRIDGE Electronics item graph by its source formula."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

BLOCK_SIZE = 4096
K = 10
IMAGE_WEIGHT = 0.1


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_graph(encoder, features, block_size=BLOCK_SIZE, k=K):
    harness = SimpleNamespace(device=features.device, knn_k=k,
                              config={"knn_chunk_size": block_size})
    harness.compute_normalized_laplacian = lambda indices, size: encoder.compute_normalized_laplacian(
        harness, indices, size)
    return encoder.get_knn_adj_mat(harness, features)


def build_modality(path, encoder, torch):
    import numpy as np
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.ndim != 2 or array.shape[0] != 63001 or not np.isfinite(array).all():
        raise ValueError("Unexpected Electronics feature array: " + str(path))
    features = torch.from_numpy(np.asarray(array, dtype=np.float32))
    indices, graph = source_graph(encoder, features)
    metadata = {
        "path": str(path), "shape": list(array.shape), "source_dtype": str(array.dtype),
        "working_dtype": str(features.dtype), "k": K, "block_size": BLOCK_SIZE,
        "normalization": "l2 clamp_min 1e-12", "topk": "torch.topk(similarity, k, dim=-1)",
        "edge_count": int(indices.shape[1]), "device": "cpu", "self_neighbors_preserved": True,
        "indices_sha256": hashlib.sha256(indices.cpu().numpy().tobytes()).hexdigest(),
        "values_sha256": hashlib.sha256(graph._values().cpu().numpy().tobytes()).hexdigest(),
    }
    del features, indices
    return graph, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--carrier-root", type=Path, required=True)
    parser.add_argument("--source-graph-hash", required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        raise RuntimeError("Graph preparation is CPU-only and must not reserve a GPU")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(str(args.output))
    import torch
    torch.set_num_threads(4)
    source = args.carrier_root / "src/models/dual_frequency_encoder.py"
    if sha256(source) != args.source_graph_hash:
        raise RuntimeError("Frozen encoder source hash mismatch")
    sys.path.insert(0, str(args.carrier_root / "src"))
    from models.dual_frequency_encoder import DualFrequencyEncoder
    started = time.time()
    image, image_meta = build_modality(args.data_root / "image_feat.npy", DualFrequencyEncoder, torch)
    text, text_meta = build_modality(args.data_root / "text_feat.npy", DualFrequencyEncoder, torch)
    graph = IMAGE_WEIGHT * image + (1.0 - IMAGE_WEIGHT) * text
    if tuple(graph.shape) != (63001, 63001):
        raise RuntimeError("Unexpected combined graph shape")
    pending = args.output.with_name(args.output.name + ".pending." + str(os.getpid()))
    torch.save(graph, pending)
    pending.replace(args.output)
    record = {"status": "completed", "graph_path": str(args.output),
              "graph_sha256": sha256(args.output), "shape": list(graph.shape),
              "nnz_uncoalesced": int(graph._nnz()), "k": K, "block_size": BLOCK_SIZE,
              "image_weight": IMAGE_WEIGHT, "text_weight": 1.0 - IMAGE_WEIGHT,
              "source_graph_source_hash": args.source_graph_hash,
              "image": image_meta, "text": text_meta,
              "torch_version": torch.__version__, "cpu_threads": 4,
              "elapsed_seconds": time.time() - started, "gpu_used": False}
    with (args.output.parent / "graph_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    print(json.dumps(record, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
