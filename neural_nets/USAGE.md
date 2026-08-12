# Encoder Pipeline Usage

Commands for training, predicting, plotting, and evaluating the encoder
ripple-prediction model. Run everything from the repo root with the project venv:

```bash
PY=/home/murphe83/Repos/COSMOS-EDFA-Dataset/venv/bin/python
```

This covers the Phase 1 (multi-topology retrain), Phase 2 (model/loss variants),
and Phase 3 (report tooling) additions. Related docs: `PLAN.md` (plan + verified
results), `REPORT.md` (Phase 3 report), `PLOT_GUIDE.md` (what each plot shows).

---

## 1. Train

```bash
# Phase 1 baseline — multi-topology retrain, stratified split, ripple target
$PY neural_nets/encoder_net.py --train dataset/multispan \
    --latent-dim 80 --inactive-loss-weight 0.1 --ripple

# Phase 2 variants — each is saved to saved_models/encoder_<variant-name>
$PY neural_nets/encoder_net.py --train dataset/multispan --variant-name pinweight    --pin-weight
$PY neural_nets/encoder_net.py --train dataset/multispan --variant-name edgeweight   --edge-weight
$PY neural_nets/encoder_net.py --train dataset/multispan --variant-name shapeloss    --shape-loss
$PY neural_nets/encoder_net.py --train dataset/multispan --variant-name posenc       --pos-enc
$PY neural_nets/encoder_net.py --train dataset/multispan --variant-name roadmheads   --per-roadm-heads
$PY neural_nets/encoder_net.py --train dataset/multispan --variant-name scaletarget  --scale-target

# Auto A/B harness — trains the Phase 1 baseline, A/Bs each single flag
# against it on topology2 held-out MAE, then trains the combined winners
$PY neural_nets/encoder_net.py --train dataset/multispan --best-combo
```

**What it does / what it checks**

- `--variant-name <name>` (`encoder_net.py:set_model_dir`) redirects all
  save/load to `saved_models/encoder_<name>` so experiments never clobber each
  other. The default (no flag) is `saved_models/encoder`.
- Every run writes `metadata.json` (architecture, `train_source`, and one flag
  per variant, e.g. `"pin_weight": true`), `test_indices.json`, and
  `test_indices_per_file.json` (per-topology held-out indices).
- Variant flags are **train-only**: at predict time their effects (pos-enc
  features, per-roadm heads, target scaling) are re-applied automatically from
  `metadata.json` — never re-pass them when predicting.
- `--best-combo` (`encoder_net.py:best_combo`) runs the full Phase 2 A/B: it
  needs the Phase 1 baseline (retrains it if missing), evaluates each single-flag
  variant on topology2 held-out MAE, prints `winning flags`, then trains
  `encoder_bestcombo` (pin + edge) with the winners.
- Watch: `baseline topo2 held-out MAE` (expect ≈ 0.0777) and `winning flags`.

**Result (verified in PLAN.md):** multi-topology + stratified training fixed the
distribution shift; no Phase 2 variant beats the plain Phase 1 retrain by a
meaningful margin, so `encoder` (Phase 1) is the recommended model.

---

## 2. Predict

```bash
# Per-model held-out predictions on topology2
$PY neural_nets/encoder_net.py --predict dataset/multispan/multispan_topology2.json --heldout
$PY neural_nets/encoder_net.py --predict dataset/multispan/multispan_topology2.json --heldout --variant-name baseline
$PY neural_nets/encoder_net.py --predict dataset/multispan/multispan_topology2.json --heldout --variant-name bestcombo

# Loop: every model x every topology, with per-topology copies
for v in "" baseline bestcombo; do
  for t in 1 2 3; do
    $PY neural_nets/encoder_net.py --predict dataset/multispan/multispan_topology${t}.json \
        --heldout ${v:+--variant-name $v} 2>/dev/null | tail -1
    src="neural_nets/saved_models/encoder${v:+_$v}/predictions"
    [ -f "$src/predictions_heldout.json" ] && \
      cp "$src/predictions_heldout.json" "$src/predictions_heldout_topology${t}.json"
  done
done
```

**What it does / what it checks**

- `--heldout` keeps only the test subset, using the model's saved split:
  `test_indices_per_file.json` when loading a single topology (indexed by
  `source_file`), otherwise `test_indices.json`. Both were written at training
  time, so numbers are reproducible.
- The baseline predates `test_indices.json`: it reconstructs
  `train_test_split(..., random_state=42)` and therefore only evaluates on
  topologies 1 and 2 (topology3's record count differs and it raises).
- Each run **overwrites** `predictions_heldout.json`, so the loop copies it to
  `predictions_heldout_topology{N}.json` — that per-topology file is what the
  metrics and plots tools consume.
- Sanity check the printed `Saved N records`: 2367 for baseline, 2385/2332/1594
  for the Phase 1+ models on topologies 1/2/3. Existing files live in
  `saved_models/*/predictions/predictions_heldout_topology{1,2,3}.json`.

---

## 3. Plot

```bash
J=neural_nets/saved_models/encoder/predictions/predictions_heldout_topology2.json
G=neural_nets/saved_models/encoder/predictions/graphs

# Full 7-plot set for one data source
$PY neural_nets/encoder_plots_v2.py --path $J --output-dir $G/report/all

# Loop: all 12 roadm instances
for r in roadm_1_booster roadm_2_booster roadm_2_preamp roadm_3_booster \
         roadm_3_preamp roadm_4_booster roadm_4_preamp roadm_5_booster \
         roadm_5_preamp roadm_6_booster roadm_6_preamp roadm_7_preamp; do
  $PY neural_nets/encoder_plots_v2.py --path $J -r $r --output-dir $G/roadms/$r
done

# Loop: every channel (1..95) — slower, ~95 invocations
for c in $(seq 1 95); do
  $PY neural_nets/encoder_plots_v2.py --path $J -c $c --output-dir $G/channels
done

# Targeted: worst edge channels + a stage filter, plus roadm x channel combos
$PY neural_nets/encoder_plots_v2.py --path $J -c 1  --output-dir $G/report/ch1
$PY neural_nets/encoder_plots_v2.py --path $J -c 95 --output-dir $G/report/ch95
$PY neural_nets/encoder_plots_v2.py --path $J -r roadm_7_preamp --output-dir $G/report/roadm_7_preamp
$PY neural_nets/encoder_plots_v2.py --path $J -r roadm_7_preamp -c 95 --output-dir $G/roadm7/ch95
```

**What it does / what it checks**

- `--path` (Phase 3) plots any predictions JSON — the held-out topology files or
  a variant dir (`saved_models/encoder_<name>/predictions/...`) — instead of the
  hardcoded default `saved_models/encoder/predictions/predictions.json`. Works
  with every plot function.
- Output layout: with `-c N` plots nest under `<output-dir>/ch<N>/`; with `-r
<roadm>` and no `-c` they write straight into `<output-dir>/`; `-r all`
  (default) plots one sample per stage; a specific roadm plots up to 3 samples.
- Each invocation writes 7 PNGs (`encoder_01..07_*.png`). Verify the final
  `Saved .../encoder_07_roadm_proficiency.png` line and that the output dir
  exists.
- Useful targets: `-c 1` is the worst channel (MAE 0.219 dB in REPORT.md); ch1,
  5, 43, 60, 70, 88, 89, 95 are where the ±0.5 dB breaches occur.

---

## 4. Metrics (`encoder_metrics.py`)

**How it works**

`collect()` (encoder_metrics.py:8) reads a predictions JSON and builds, from each
record's `predicted_gain_spectra`, `true_gain_spectra`, and boolean `mask`, three
arrays: predicted, true, and the mask — producing `err = pred − true` on active
channels only (`NaN` where inactive). Every metric is **masked** over active
channels: `masked_mae()` = `sum(|err|·mask) / sum(mask)`. `metrics()` (Phase 3)
packs MAE, RMSE, p95, p99, max, and % channels > 0.5 dB into a dict for one file.
The default single-file mode prints `summary()` (overall), `by_roadm()`
(per-stage MAE/p99), `by_pin()` (per-1-dB pin-bin MAE), and a per-channel MAE line.

**Usage**

```bash
# 1) Single file (defaults to saved_models/encoder/predictions/predictions.json)
$PY neural_nets/encoder_metrics.py
$PY neural_nets/encoder_metrics.py neural_nets/saved_models/encoder/predictions/predictions_heldout_topology2.json

# 2) Cross-model comparison table (Phase 3)
$PY neural_nets/encoder_metrics.py --compare \
  "baseline:neural_nets/saved_models/encoder_baseline/predictions/predictions_heldout_topology2.json,\
encoder:neural_nets/saved_models/encoder/predictions/predictions_heldout_topology2.json,\
bestcombo:neural_nets/saved_models/encoder_bestcombo/predictions/predictions_heldout_topology2.json"

# 3) Same, plus the per-roadm MAE/p99 matrix (Phase 3 --roadm flag)
$PY neural_nets/encoder_metrics.py --compare "<same pairs>" --roadm
```

**What it does / what it checks**

- `--compare` takes a comma-separated list of `label:path` pairs (quote the whole
  arg, no spaces) and prints a markdown table: label, n, MAE, RMSE, p95, p99, max,
  % > 0.5 dB. `--roadm` adds a matrix with one column per label of `MAE/p99` per
  stage. Both reuse `collect`/`masked_mae`, so numbers always match the
  single-file output.
- Sanity check `n` against the expected held-out record counts (2367 baseline,
  2332 Phase 1+ on topology2) — it confirms which split a file actually used.
- Reference numbers (topology2 held-out, REPORT.md): baseline MAE 0.0961,
  encoder 0.0777, bestcombo 0.0778; `roadm_7_preamp` MAE 0.2084 → 0.0499/0.0469.

---

## Expected held-out record counts (sanity reference)

| Model     | topo1 | topo2 | topo3 |
| --------- | ----- | ----- | ----- |
| baseline  | 2367  | 2367  | — (n/a) |
| encoder   | 2385  | 2332  | 1594  |
| bestcombo | 2385  | 2332  | 1594  |

