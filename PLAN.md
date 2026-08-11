# Encoder Ripple-Prediction Error Reduction Plan

Goal: understand and reduce the difference between predicted and true EDFA gain ripple
(±0.5 dB per-channel deployment spec), and establish an honest evaluation pipeline.

Working dir for all commands: repo root `/home/murphe83/Repos/COSMOS-EDFA-Dataset`.
Use the project venv python: `PY=/home/murphe83/Repos/COSMOS-EDFA-Dataset/venv/bin/python`.

---

## Context / verified diagnosis

- Model: `neural_nets/encoder_net.py`, saved at `neural_nets/saved_models/encoder/`
  (`model.keras`, `scaler.pkl`, `metadata.json`).
- Architecture: MLP encoder, input 209 features (95 masked input spectra + 95 mask +
  7 scalars + 12 roadm one-hot), latent_dim 80, output 95 ripple values (dB around the
  active-channel mean gain). Loss = masked MAE over active channels + inactive penalty
  (weight 0.1). `StandardScaler` on features only; target is unscaled ripple.
- **The model was trained on `dataset/multispan/multispan_topology1.json` only**
  (11,832 samples = 9,465 train + 2,367 test, `train_test_split(test_size=0.2,
random_state=42)`).
- The original saved predictions (`predictions.json`) were generated on
  `multispan_topology2.json` and contain ALL 11,832 records (train + test mixed) —
  not a clean test metric.
- **Key cause of the large errors: train/eval distribution shift.**
  - `roadm_7_preamp` in topology2 operates at pin_total −38…−29 dBm (mean −32),
    almost entirely OUTSIDE its topology1 training range (−33…−15, mean −20).
  - Verified by running the saved model on all 3 topologies: `roadm_7_preamp` MAE =
    0.038 on topology1 (in-distribution) vs 0.209 on topology2 (extrapolation);
    all other stages stay 0.02–0.06 on both.
- Secondary error drivers (all measured):
  - Low total input power is the dominant driver: MAE ~0.05 dB at pin −3 dBm rises
    to ~0.21 dB at pin −31 dBm (deep small-signal regime).
  - Band-edge channels worse: ch1 ≈ 0.116, ch70–95 ≈ 0.10–0.12 dB.
  - Regression to the mean: predicted std 0.132 < true std 0.154; linear slope
    pred-vs-true = 0.70; the larger the true ripple swing, the more it is
    under-predicted (compressed/flat output).
- Data inventory (`dataset/multispan/`): topology1 = 11,832 (12 roadms × 986),
  topology2 = 11,832, topology3 = 7,888 (**8 roadms only**, no roadm_5_booster /
  6_preamp / 6_booster / 7_preamp). Total = 31,552 samples.

---

## Corrected baseline numbers (use these as the reference)

Masked MAE = `sum(|pred − true| · mask) / sum(mask)` (active channels only, each
record's own mask). Dividng by 95 instead inflates/dilutes results.

| Metric              | In-dist (topology1, held-out 2367) | Shifted (topology2, held-out 2367) |
| ------------------- | ---------------------------------- | ---------------------------------- |
| Masked MAE          | 0.077 dB                           | 0.096 dB                           |
| RMSE                | 0.099 dB                           | 0.128 dB                           |
| p95 / p99           | 0.198 / 0.274 dB                   | 0.261 / 0.400 dB                   |
| max \|err\|         | 0.560 dB                           | 0.660 dB                           |
| % channels > 0.5 dB | 0.001%                             | 0.108%                             |
| roadm_7_preamp MAE  | 0.081 dB                           | 0.208 dB                           |
| roadm_7_preamp p99  | 0.273 dB                           | 0.504 dB                           |

Per-roadm MAE range (in-dist): roadm_1 ~0.05 → roadm_6 ~0.09 (monotonic, tracks pin).

---

## Phase 0 — Evaluation harness ✅ DONE

Build honest eval + metrics. No model changes. Files changed:

1. `neural_nets/encoder_net.py`
   - `train_model()` now saves the exact held-out indices to
     `saved_models/encoder/test_indices.json` (after `train_test_split`) and records
     `"train_source"` in `metadata.json`.
   - `predict(path, heldout=False)`: if `--heldout`, keeps only the test subset
     (reads `test_indices.json` if present; otherwise reconstructs the split from
     `random_state=42` on the same topology and warns). Outputs are written to
     `predictions_heldout.csv/json` (suffix `_heldout`).
   - CLI flag: `--heldout`.
   - Note: the CURRENT saved model predates `test_indices.json`, so `--heldout`
     uses the reconstructed split (identical for this model).
2. `neural_nets/encoder_metrics.py` (new) — masked MAE / RMSE / p95 / p99 / max /
   % > 0.5 dB, plus per-roadm, per-pin-bin, per-channel breakdowns. Takes a JSON path
   (defaults to `predictions.json`).
3. `neural_nets/encoder_plots_v2.py` — fixed per-sample error denominators in
   `plot_error_vs_power` (06) and `plot_roadm_proficiency` (07) to
   `np.nansum(abs_error, axis=1) / masks.sum(axis=1)` (07 also indexes masks by the
   roadm subset). The other plots (heatmap, scatter, boxplot, ripple) already operate
   per-channel on active values.

### How to reproduce Phase 0

```bash
PY=/home/murphe83/Repos/COSMOS-EDFA-Dataset/venv/bin/python

# Held-out predictions on the training topology
$PY neural_nets/encoder_net.py --predict dataset/multispan/multispan_topology1.json --heldout

# Metrics (pass the JSON, not the CSV)
$PY neural_nets/encoder_metrics.py neural_nets/saved_models/encoder/predictions/predictions_heldout.json

# Graphs (reads default predictions.json; -c 1..95 for per-channel, -r for roadm filter)
$PY neural_nets/encoder_plots_v2.py --output-dir neural_nets/saved_models/encoder/predictions/graphs
$PY neural_nets/encoder_plots_v2.py --channel 95 --roadm roadm_7_preamp --output-dir neural_nets/saved_models/encoder/predictions/graphs
```

Expected: `predictions_heldout.json` has 2,367 records; metrics ≈ the in-dist column above.

---

## Phase 1 — Close the data gap ✅ DONE

Goal: stop extrapolating. Train on all three topologies so `roadm_7_preamp`'s low-power
regime is in-distribution. Expected: its MAE drops from 0.209 → ~0.04–0.08.

### Step 1.1 — Multi-topology training

`load_dataset()` already accepts a directory (rglobs `*.json`). Train on:

```bash
$PY neural_nets/encoder_net.py --train dataset/multispan \
    --latent-dim 80 --inactive-loss-weight 0.1 --ripple
```

- Total 31,552 samples. `roadm_categories` stays the 12 from metadata; topology3
  simply contributes no rows for its missing 4 roadms (one-hot width unchanged).
- Keep the SAME architecture/loss as the original for a controlled A/B against
  baseline 0.077 / 0.096.
- Watch training time: 80 epochs × 31.5k samples on CPU is slow; may drop epochs to
  ~40–50 or batch to 128 if needed. Consider recording `epochs`/`batch_size` in
  metadata for reproducibility.

### Step 1.2 — Stratified split

Replace the plain `train_test_split` in `train_model()` with a stratify on
`roadm` (and ideally a coarse pin bin, e.g. 5 dB wide) so the low-pin roadm_7_preamp
samples land proportionally in train/test. Recommended: build a stratify label
`f"{roadm}|{pin_bin}"`, guard against bins with too few members (fall back to roadm
only for those), then `train_test_split(..., stratify=labels, random_state=42)`.
This run writes `test_indices.json` + `train_source` automatically (Phase 0 code).

### Step 1.3 — Evaluate after retrain

```bash
$PY neural_nets/encoder_net.py --predict dataset/multispan/multispan_topology1.json --heldout
$PY neural_nets/encoder_net.py --predict dataset/multispan/multispan_topology2.json --heldout
$PY neural_nets/encoder_net.py --predict dataset/multispan/multispan_topology3.json --heldout
$PY neural_nets/encoder_metrics.py neural_nets/saved_models/encoder/predictions/predictions_heldout.json
```

(Each run overwrites `predictions_heldout.*`; move/copy between runs to compare.)
Success criterion: overall held-out MAE below ~0.077 and `roadm_7_preamp` MAE well
below 0.10 on topology2.

### Result (verified)

- Retrained on `dataset/multispan` (all 31,552 samples, 50 epochs, batch 128) with a
  stratified split on `roadm|5 dB-pin-bin`; recorded in `metadata.json`
  (`stratified: true`, `train_source: dataset/multispan`).
- Held-out eval on topology2 (per-file split via `test_indices_per_file.json`,
  2,332 records): overall masked MAE 0.078 dB (target ~0.077) and
  **roadm_7_preamp MAE 0.208 → 0.050 dB** (target <0.10). Success criteria met.

---

## Phase 2 — Model / loss improvements ✅ DONE

Attack the in-distribution error and the ±0.5 dB tail. A/B each variant on the same
held-out protocol; report masked MAE, p95/p99, % > 0.5 dB, per-roadm.

1. **Pin-weighted loss** — up-weight low-pin samples in `compute_masked_loss`
   (e.g., weight ∝ inverse sample density in pin, or monotonic decreasing in pin),
   since low pin drives the largest errors.
2. **Edge-channel weighting** — per-channel weights in the masked loss, higher for
   ch1 and ch70–95.
3. **Shape losses** — port `gradient_loss` / `cosine_shape_loss` / `combined_loss`
   from `neural_nets/multispan_net.py` (MSE + 0.3·gradient + 0.2·cosine) to fix the
   compressed ripple (slope 0.70) and band edges.
4. **Per-roadm heads** — multi-task architecture like `multispan_net.py` (shared base
   - per-roadm head) instead of a single shared output; gives each stage dedicated
     capacity.
5. **Channel positional encoding** — add sin/cos encoding of channel index so the net
   knows spectral position (helps edges).
6. Optional: target scaling (StandardScaler on Y) and/or larger `latent_dim`.

### Result (verified)

All six variants were trained and A/B'd on the same held-out protocol (topology2,
per-file stratified held-out, 2,332 records) and saved under
`saved_models/encoder_*`:

| Variant               | MAE    | RMSE   | p99    | max \|err\| | % > 0.5 dB |
| --------------------- | ------ | ------ | ------ | ----------- | ---------- |
| baseline (Phase 0)    | 0.0961 | 0.1277 | 0.3998 | 0.6600      | 0.108%     |
| Phase 1 (no flags)    | 0.0777 | 0.1006 | 0.2836 | 0.7243      | 0.009%     |
| pin-weight            | 0.0777 | 0.1009 | 0.2853 | 0.7239      | 0.007%     |
| edge-weight           | 0.0777 | 0.1008 | 0.2836 | 0.7239      | 0.007%     |
| shape-loss            | 0.0829 | 0.1069 | 0.2980 | 0.7198      | 0.013%     |
| per-roadm heads       | 0.0779 | 0.1012 | 0.2877 | 0.8209      | 0.014%     |
| pos-enc               | 0.0784 | 0.1018 | 0.2892 | 0.7250      | 0.016%     |
| target scaling        | 0.0778 | 0.1014 | 0.2884 | 0.7121      | 0.013%     |
| best combo (pin+edge) | 0.0778 | 0.1009 | 0.2850 | 0.7249      | 0.007%     |

Conclusion: no Phase 2 variant beats the plain Phase 1 retrain by a meaningful
margin; shape-loss and per-roadm heads are slightly worse. The improvement came
from multi-topology + stratified training, not loss/architecture tweaks.

---

## Phase 3 — Report ✅ DONE

See **`REPORT.md`** for the full report: comparison tables (baseline vs Phase 1 vs
best Phase 2 variant, overall + per-roadm + per topology), the ±0.5 dB spec
verdict, and the `roadm_7_preamp` data decision.

- **Verdict:** ±0.5 dB met statistically (p99 ≈ 0.28 dB; 10/107,487 active channels
  > 0.5 dB), not worst-case (max 0.724 dB on ch1). Tail is band-edge, not power-regime.
- **Decision:** no new `roadm_7_preamp` data needed — MAE 0.050 (< 0.10), p99 0.182,
  0 channels > 0.5 dB.
- Tooling added for the report: `encoder_metrics.py --compare` (cross-model tables)
  and `encoder_plots_v2.py --path` (plot any predictions file).

---

## Key files

| File                                               | Role                                               |
| -------------------------------------------------- | -------------------------------------------------- |
| `neural_nets/encoder_net.py`                       | train/predict CLI; Phase 0 held-out logic          |
| `neural_nets/encoder_metrics.py`                   | masked metrics (new)                               |
| `neural_nets/encoder_plots_v2.py`                  | graphs (denominators fixed)                        |
| `neural_nets/multispan_net.py`                     | reference: shape losses + per-roadm heads          |
| `neural_nets/helpers.py`                           | multispan JSON → samples, ripple computation       |
| `dataset/multispan/multispan_topology{1,2,3}.json` | data (11,832 / 11,832 / 7,888)                     |
| `neural_nets/saved_models/encoder/`                | model, scaler, metadata, test_indices, predictions |
