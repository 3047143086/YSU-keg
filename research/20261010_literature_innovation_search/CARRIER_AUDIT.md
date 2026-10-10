# Carrier audit, 2026-10-10

Use the official **MSCA** source at `/data/sunzixuan/MSCA-main/src` for the new negative-sampling hypothesis. It has no SVD initialization and uses the existing Baby/Sports features and sparse graph caches. A fresh original-MSCA baseline and the intervention must share this exact carrier, configuration, validation-only runner, seed and resources. Historical BRIDGE scores are evidence that its runner completed, not a substitute MSCA baseline.

This audit used bounded read-only server inventory plus preparation of a new isolated carrier. It did not initialize CUDA, launch training, modify an old experiment, or touch GPU 0/PID 600045. User instructions already authorize continuing experiments and pushing their implementation; no additional approval is needed for the new isolated run.

## Official MSCA source and configuration

The complete inspected source/config text and hashes are in `msca_inventory.json`. The actual official source contains 25 files, 112,964 bytes. Important SHA256 values:

| File relative to official `src` | SHA256 |
| --- | --- |
| `models/msca.py` | `066aafcf2e5373b20b2fdf9b3660302b8dfcaf580710d3ab864fac4f19e2360b` |
| `common/trainer.py` | `9520a7a712bb29ec0cb7c6319e362a0f3a89d33cd9ebfa7acd0afe9b48051e39` |
| `configs/model/MSCA.yaml` | `a95df9ba3adb0f0657dc16e7c9231938c35d6c830b4e24da081040f97435f927` |

`Config(model='MSCA', dataset=..., config_dict=..., mg=False)` reads `configs/` under the current working directory. The runner therefore needs to change into the copied `carrier/src`. Its `data_path` is concatenated directly with `dataset`; use an absolute path ending in `/`. The model YAML stores tuning values as one-element lists, so supply the following scalar overrides explicitly rather than starting the grid-search entry point:

| Frozen setting | Baby | Sports |
| --- | --- | --- |
| `n_layers` | 2 | 3 |
| `fusion_coeff` | 0.4 | 0.3 |
| `cl_weight` | 0.005 | 0.005 |
| `reg_weight` | 3e-7 | 5e-8 |

Other original defaults are embedding size 64, `knn_k=10`, `is_sparse=True`, `tau=0.2`, Adam learning rate 0.001, batch size 2048, one sampled negative, `use_full_sampling=False`, seed supplied explicitly, and patience 20. Evaluation uses Recall/NDCG/Precision/MAP at 5, 10, 20 and 50; Recall@20 is the checkpoint metric. A bounded epoch budget and evaluation batch size can be fixed identically for both arms and recorded as protocol overrides.

Interaction fields are `userID`, `itemID`, `timestamp`, and the train/validation/test split is `x_label`. The original `TrainDataLoader._sample_neg_ids` draws with `random.sample(self.all_items, 1)` and rejects membership in `history_items_per_u`. `_get_neg_sample` emits the ordinary stacked user/positive/negative batch. This provides a narrow intervention point without editing the model, its initialization, graph construction, Trainer, or evaluator.

The original MSCA constructor loads `image_adj_10_True.pt` and `text_adj_10_True.pt`. Both already exist for Baby and Sports. If absent it writes a graph cache to the configured dataset directory. The new carrier consequently has private copies of these graph caches; immutable interaction/features are used through links to the original data. No cache generation or source directory write is required. MSCA uses `.cuda()` in its constructor; a real model smoke needs GPU 1, while protocol/sampling unit tests can run with CUDA hidden.

## Existing MSCA execution evidence and entry-point risks

`/data/sunzixuan/PRAVI_remote/src/models/msca.py` is an extended model with identity-by-default refinement/weighting hooks. Its SHA256 is `3751fb09a7fc0579a0ebd102e46376a1897e595f74929e4f2447705aaceb537d`; it is a different source from the official carrier above. The directory has all Baby/Sports assets and an actual `experiments213` directory. Inventory confirmed `msca_baby_official_seed2026.stdout` (104,449 bytes), `msca_dcrl_baby_s2026_gpu1.stdout` (109,692 bytes), and completed-but-unsuccessful MSCA_SG logs (99,787 bytes). Filenames alone are not new result validation.

Local `PRAVI/RAPA_EXPERIMENT.md` records a completed 12-epoch/36-probe MSCA gradient audit, with exact source-objective parity and no material gradient-direction conflicts. `PRAVI/MSCA_SG_EXPERIMENT.md` records prior original MSCA Baby rounded validation R@20 0.1043 and N@20 0.0462; the SG intervention failed its validation screen. These records establish a usable non-SVD path, but the new official-source study must rerun its paired baseline.

Both official `main.py`/`utils/quick_start.py` and the PRAVI generic entry point construct test data and run a parameter loop. They are unsuitable for choosing the present hypothesis. The legacy PRAVI wrapper supports `--validation-only`, but the new study should reuse the established validation-only orchestration with the official MSCA `Trainer._train_epoch` and `Trainer.evaluate(valid_loader, is_test=False)` rather than launching generic quick-start scripts.

Other available models include MGCN, FREEDOM, BM3, LATTICE and more in `baselines/MRLib/Multimodal-Recommendation-Library-main/src/models`. These are library ports; this audit did not authenticate separate official repositories. Some graph-cache paths write to data directories. Its generic main/quick-start also evaluates test data and selects across parameter combinations, so it is not an isolated validation-only baseline launcher. Its entry point sets a 48-thread NumExpr limit and would require an explicit four-thread override.

## Stable BRIDGE carrier and completed baselines

The previous runner can be inspected at `research/20261009_candidate_gate/train.py`; its bytes are identical to the October 8 stable recovery and September 10 innovation runner. It preserves the original Trainer training/evaluation calls, constructs only train and validation datasets, uses full-sort validation with training-history exclusion, and selects a checkpoint by rounded four-decimal Recall@20, keeping the earlier epoch on ties. Its checkpoint reload produces exact unrounded per-user validation metrics. Model/source/input/plugin hashes are checked before and after training, and checkpoint/per-user artifacts are hashed.

The BRIDGE carrier in `research/20261010_frozen_carrier/src` contains the verified 35-file snapshot of the remote source. All 35 local hashes passed in this audit. The six checked core hashes also match across `/data/sunzixuan/bridge_official`, October 8 target-isolated/recovery carriers, October 9 candidate-gate, and the frozen Electronics baseline source. Editable `baselines/BRIDGE-official` should not be substituted on the assumption that its current bytes match.

The October 8 original BRIDGE baseline results and their checkpoint/per-user hashes were checked again on the server:

| Dataset, seed 2026 | Completed epochs | Best epoch, zero based | Exact validation R@20 | Exact validation N@20 |
| --- | ---: | ---: | ---: | ---: |
| Baby | 111 | 89 | 0.10546504793740588 | 0.04910749907453286 |
| Sports | 130 | 108 | 0.1195296505011454 | 0.05647686319160367 |

Remote runs are `/data/sunzixuan/target_isolated_20261008_autoresearch_v1/runs/baby_baseline_s2026` and `sports_baseline_s2026`. Baby checkpoint SHA256 is `6deb59326f61137d658565d4c19fc892d7a5cb50578a120abc12497bcbc469a9`; Sports is `75da60539924743bb9c7a5aeb89689859cce5c8e914def4b9fd8040ee1dd402e`. They took approximately 2,074 and 5,956 seconds respectively. The Electronics numerical failures remain frozen and provide no successful baseline result for this new mechanism.

The old numerical guard checks finite tensors/gradients/optimizer state, records the first failure, and does not sanitize, skip, or silently retry failed updates. It is specific to the BRIDGE SVD path. The new non-SVD MSCA study should keep ordinary finite checks but should not enable old SVD repair/recovery code.

## Resources, budgets and new deployment helper

The bounded server inventory found Python 3.8, PyTorch 1.12.1+cu113, NumPy **1.24.4**, SciPy 1.7.3, PyG 2.3.1 and pandas 1.3.5 in the MMRec environment. Old notes reporting NumPy 1.22 are not the current runtime. Transformers, Sentence Transformers and OpenCLIP were not installed. Existing CLIP ViT-B-16 weights are present, but this study needs neither raw-image encoding nor a new semantic model.

At preparation, GPU 1 was an RTX 3090 with 24,232 MiB free and no compute process. No owned research training/queue process matched the bounded process inspection. GPU 0 is occupied and must remain untouched. The new helper queries only GPU 1 metadata and launches with `CUDA_VISIBLE_DEVICES=1`; physical GPU 1 becomes logical CUDA 0. `Config` itself overwrites `CUDA_VISIBLE_DEVICES` from `gpu_id`, so the runner must pass physical `gpu_id=1` and apply any memory-fraction call to logical device 0. Set Torch CPU threads and OMP/MKL/OpenBLAS/NumExpr thread limits to four.

The old queue permits one child at a time, smoke timeout 1,800 seconds, formal timeout eight hours and total budget 48 hours including resource waits. Its GPU selector scans both devices, which must not be copied unchanged: pin physical GPU 1. Never restart an old frozen manager or overwrite an old run directory. A fresh baseline is safer than legacy cross-study result reuse because source, model and configuration changed.

The new helper is `research/20261010_interest_negative/manage.py`, with four actions:

```powershell
python research/20261010_interest_negative/manage.py prepare
python research/20261010_interest_negative/manage.py deploy
python research/20261010_interest_negative/manage.py status --brief
python research/20261010_interest_negative/manage.py launch
```

Only `prepare` has been executed by this audit. It created `/data/sunzixuan/interest_negative_20261010_v1/carrier/src` with all 25 original files unchanged, copied the official `LICENSE` when present, and populated `carrier/data/` with the complete Baby/Sports input inventory. The two model graph caches per dataset are private copies; other immutable assets are read-only links. After adding the license and complete data inventory, the source/data protection manifest hash is `82973a12cd72a83393623e430555d68871b8c606e2030bf1296697a1ecc7fa43`; the carrier manifest hash is `61117fee5451883a9e585bc4c5142c4a4cfcecf8a7c185c86b013941b7012434`.

`deploy` requires a clean committed isolated bundle, transfers a compressed base64 archive with explicit SHA256 inventory, rejects unsafe paths, and verifies original/carrier hashes. `launch` is a separate explicit operation: it verifies deployed revision, source/data/bundle hashes, runs CPU unittest discovery with CUDA hidden, requires nonzero passing tests, rechecks ownership/resources and at least 14 GiB free on GPU 1, then starts only the new `pipeline.py` detached. A launch receipt makes subsequent deployment/launch attempts refuse. Root will invoke launch after completing the protocol and runner; this audit has not invoked it.

`status` writes timestamped receipts and a latest snapshot under the local experiment's `status/` directory. It retrieves results/progress/report/log tails, artifact SHA256s, checkpoint/per-user hash matches when available, the exact recorded manager PID/command/owner identity, owned research processes, and GPU 1 metadata. It does not overwrite local protocol/model source. Local helper and embedded remote Python syntax checks passed.
