# Final audit — Baby interest-negative pilot

This audit is based on the immutable deployment receipt, the final
`research_report.json`, `baby_gate.json`, `results.tsv`, `latest_status.json`,
and the three formal-arm result receipts. The protected source revision is
`e1775dad7b57f2b1ffae08609d720e8122eccfa9` and the deployment reports a clean
working tree.

## Formal arms

All three preregistered Baby arms exited with code 0 and have
`status=completed`, `test_loader_constructed=false`, `test_evaluated=false`,
and `protected_paths_match=true`. They share the same initialization hash,
TRAIN group-asset hashes, validation population, and positive training order
through the 56-epoch overlap:

* `initial_state_sha256`: `4aacd659cec4a99f4e2e1898c681bb44fd302479b71f031af1794af7a5900c40`
* `group_assets_sha256`: `e8f512cd8072f2afa5ddd81f7a49607384ae7273613034beeb7234a3510f97e3`
* `group_diagnostics_sha256`: `553d362befbf61d1ed27c760afd914145cc599019828a2ade33308dd4ca43761`
* validation population: 19,445 users and 20,559 positives
* comparability receipt: `passed=true`, `arms=3`, `overlap_epochs=56`

The fixed-seed results are:

| arm | epochs | finite updates | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 |
|---|---:|---:|---:|---:|---:|---:|
| uniform control | 66 | 3,828 | 0.067161713747 | 0.036664605294 | 0.102898406984 | 0.045759199907 |
| semantic negatives | 56 | 3,248 | 0.066733154563 | 0.036486660125 | 0.103789932533 | 0.045869779291 |
| shuffled-group control | 56 | 3,248 | 0.067071716318 | 0.036554489356 | 0.103472308954 | 0.045755449922 |

The per-arm checkpoint and per-user hashes were verified by the independent
status audit. The different best-epoch lengths are the fixed early-stopping
outcome, not a missing or crashed job.

## Frozen gate and conclusion

Semantic negatives failed both preregistered reference comparisons. Against
uniform, relative Recall@20 was `0.008664133628` (passes the `0.005` threshold),
but relative NDCG@20 was `0.002416549771` (fails), and Recall@10 delta was
`-0.000428559184` (fails the `-0.0002` floor). Against shuffled groups,
relative Recall@20 was `0.003069648115` (fails), relative NDCG@20 was
`0.002498704930` (fails), and Recall@10 delta was `-0.000338561755` (fails).
Both comparisons are therefore `pass=false`; `baby_gate.json` reports
`passed=false`, `status=fail`.

The queue correctly stopped after the Baby gate. No Sports preparation or
formal arm was launched; all three Sports rows remain `pending/missing` in
`results.tsv`. There was no technical crash in the three completed formal
arms. `research_report.json` records `status=completed_fail`,
`publication_evidence_established=false`, `novelty_established=false`, and
`test_evaluated=false`.

`pipeline_status.json` has a stale `completed_formal_arms: 0` field, while its
`completed` list contains all three completed Baby arms and
`research_report.json` records `completed_formal_arms: 3`; this is a reporting
counter inconsistency only and does not change the gate or arm receipts.

The status report records normalized-text SHA-256
`9da891466d2ee2d3572883e06ec8b89fec4bda891c97b8f0fbac18ccc13d4100` for
`results.tsv`; the on-disk Windows-CRLF byte hash is
`798e5c80b577eb09dd6cf6484e42848a0f7e56d260d5f71d3c974ba7ad12d573`, whose
LF-normalized bytes produce the recorded hash.

## Runtime provenance

The training receipts imported NumPy `1.22.4` and PyTorch `1.12.1+cu113`.
The earlier manager preflight package metadata reported NumPy `1.24.4` from
the environment's `dist-info`; this metadata/import mismatch is recorded here
for reproducibility. The imported runtime was identical across all three
formal arms and did not alter the frozen source, TRAIN assets, validation
population, or gate comparison.

This pilot is a completed negative result. It does not support a publication
claim or justify treating semantic negative sampling as a positive innovation
on Baby under the preregistered gate.
