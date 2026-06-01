# seqnoc — NoC classifier on the tokenized `noc_filtered_grouped` dataset

External tokenized dataset (NOT PROVEDIt / GlobalFiler). Different kit
(28 loci, PowerPlex-Fusion-6C-like: Penta D/E, DYS570/576), different
representation: each profile is a sequence of up to 220 peaks ×
3 features `(locus_idx 0-27, allele, log_height)` + a validity mask.
Split is donor-grouped (leakage-safe). Labels 0-indexed (0=NoC1 … 4=NoC5).

## Best result (kept here)

Base set-Transformer (`scripts/train_seqnoc.py`, d_model=96, 3 layers,
349 K params), balanced sampler, early-stop on val macro-F1.

| Pipeline | accuracy | macro-F1 |
|---|---|---|
| **Base (no finetune)** | **0.807** | **0.437** |

Per-class F1 (NoC1..5): `[0.978, 0.421, 0.524, 0.225, 0.034]`
(`seqnoc_metrics.json`, checkpoint `best_seqnoc.pt`).

## What did NOT help (and why)

- **Bigger model (2.27 M, d192/5L)** — overfit, val macro-F1 0.505 < small 0.576.
- **TTA (shuffle + jitter + MC-dropout)** — shuffle is a no-op (the
  set-Transformer is permutation invariant); jitter barely moved argmax.
- **Per-class bias tuning** — small lift on val, did not generalize.
- **CORN ordinal head + train-time aug (v2)** — val macro-F1 plateaued at
  0.447, no better than softmax v1.
- **Finetune on a donor-disjoint test half, eval the other half**
  (`scripts/finetune_seqnoc_test.py`, `seqnoc_ft_test_metrics.json`) —
  made it WORSE: pooled macro-F1 0.437 → 0.288. Each half (~1900 profiles)
  is too small; the model overfit and forgot the train-learned features,
  collapsing NoC3/4 → NoC5.

## Why ~0.44 is a data-bound ceiling, not a model limit

Sanity check on the test features (max distinct alleles per locus = MAC,
the classic forensic NoC heuristic):

| NoC | n | mean peaks | mean MAC |
|---|---|---|---|
| 1 | 2886 | 78.7 | 8.66 |
| 2 | 95 | 109.1 | 10.55 |
| 3 | 295 | 109.0 | 10.89 |
| 4 | 297 | 117.6 | 11.23 |
| 5 | 270 | 86.0 | 9.62 |

- A true single-source (NoC=1) profile can have at most 2 alleles per locus,
  so MAC should be 2. Here NoC=1 has MAC = 8.66 → the peaks are **unfiltered**
  (stutter / noise / pull-up all present); "filtered" in the dataset name is
  not artefact removal.
- MAC does not separate NoC: NoC1 (8.66) ≈ NoC5 (9.62). The allele-count
  signal — the core NoC cue — is destroyed.
- Total peak count is non-monotonic (NoC5 has FEWER peaks than NoC3),
  because high-NoC samples here are low-template / degraded; template amount
  and mixture ratio confound the peak count.
- A pure MAC rule scores acc 0.05 / macro-F1 0.034 on this test set.

Conclusion: with only `(locus, allele, log_height)` and no peak-label
probability to filter artefacts, classes 2-5 are not cleanly separable.
macro-F1 ≈ 0.44 reflects "NoC1 vs rest" being the only reliable split.
More capacity / ordinal loss / TTA cannot recover absent signal.

The richer PROVEDIt `[24, 50, 89]` pipeline (stutter features, allele freq,
mixture proportions) remains the correct benchmark — see `docs/data_pipeline.md`
and the NoCNet-v2 report.

## Files

| File | What |
|---|---|
| `best_seqnoc.pt` | base set-Transformer checkpoint (d96/3L) |
| `seqnoc_metrics.json` | base test metrics (acc 0.807, macro-F1 0.437) |
| `seqnoc_ft_test_metrics.json` | donor-disjoint finetune-on-test experiment (worse) |
| `scripts/train_seqnoc.py` | base trainer + TTA + bias tune |
| `scripts/train_seqnoc_v2.py` | CORN + train-aug variant (no improvement) |
| `scripts/finetune_seqnoc_test.py` | donor-disjoint 2-fold test-finetune |
