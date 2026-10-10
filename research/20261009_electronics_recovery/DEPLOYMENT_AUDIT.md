# Electronics Recovery Audit

Date: 2026-10-09. This record covers data recovery only; no model training was
started.

The source was read from host214:

`/data/gongshanghui/sun_exp/LOBSTER/data/elec`

The destination was created on host213 at the isolated path:

`/data/sunzixuan/bridge_extra_assets_20261009/elec`

It is outside `/data/sunzixuan/bridge_official`; the preflight directory was
empty and no official input was overwritten.

## Transfer Checks

- Target free space before transfer: `86686052352` bytes, above the required
  5 GiB.
- Remote SCP return code: `0`.
- Source files were hashed before transfer and destination files were hashed
  after transfer.
- Source and destination sizes and SHA256 values match for all three files.
- The source was read-only throughout; `source_modified` is `false`.
- The transfer used the existing `remote_session` connection helper and kept
  credentials in memory. No credential is stored in this directory.

## Structural Checks

`target_structure.json` is a read-only audit from host213. It confirms the
interaction schema, contiguous zero-based IDs, split counts, feature shapes,
and finiteness of all feature blocks:

- `elec.inter`: 1,689,188 rows; 192,403 users; 63,001 items; split counts
  `1,254,441 / 211,296 / 223,451` for labels `0 / 1 / 2`.
- `image_feat.npy`: `(63001, 4096)`, `float64`, all blocks finite.
- `text_feat.npy`: `(63001, 384)`, `float32`, all blocks finite.
- `mm_adj_10.pt` is not present. Its creation or recovery is a separate,
  hash-audited preprocessing step.

The recovered files are not connected to the current BRIDGE experiment and no
Electronics training, validation, test evaluation, or graph generation was
performed by this recovery task.

## Local Artifacts

- `recover.py`: bounded server-to-server SCP and size/hash verification.
- `audit.py`: read-only structural validation.
- `source_manifest.json`: audited source sizes and hashes.
- `target_preflight.json`: empty-destination and disk-space preflight.
- `target_manifest.json`: destination sizes and hashes.
- `transfer_receipt.json`: completed transfer receipt.
- `target_structure.json`: feature and interaction structure audit.

`recover.py` refuses a non-empty destination rather than overwriting recovered
assets. Future read-only structural checks can use:

```powershell
python research/20261009_electronics_recovery/audit.py
```
