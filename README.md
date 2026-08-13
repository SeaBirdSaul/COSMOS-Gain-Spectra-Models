# COSMOS-EDFA Dataset and Neural Net Prediction Pipeline

This repository contains the data, model code, and reporting workflow used to predict EDFA gain spectra and ripple across multispan optical topologies. The project is organized around the measurement dataset in [dataset/multispan](dataset/multispan), the model implementations in [neural_nets](neural_nets), and the evaluation docs in [Docs/PLAN.md](Docs/PLAN.md) and [Docs/REPORT.md](Docs/REPORT.md).

The goal is to estimate the per-channel gain response of an optical system from measured input conditions and compare those predictions against the true measured gain spectrum. This is useful for understanding model drift, stage-specific behavior, edge-channel failures, and deployment suitability under real-world operating conditions.

## Repository overview

The repo includes three main neural-net workflows:

- [neural_nets/neural_net.py](neural_nets/neural_net.py): legacy single-span model workflow
- [neural_nets/multispan_net.py](neural_nets/multispan_net.py): legacy multispan stage-aware model workflow
- [neural_nets/encoder_net.py](neural_nets/encoder_net.py): current primary model used for multispan gain-ripple prediction and evaluation

Supporting tooling:

- [neural_nets/encoder_metrics.py](neural_nets/encoder_metrics.py): metric extraction and comparison tables
- [neural_nets/plots.py](neural_nets/plots.py): legacy single-span plotting workflow
- [neural_nets/multispan_plots.py](neural_nets/multispan_plots.py): legacy multispan plotting workflow
- [neural_nets/encoder_plots_v2.py](neural_nets/encoder_plots_v2.py): current production plotting workflow for the encoder model

Documentation:

- [Docs/PLAN.md](Docs/PLAN.md): experiment plan and diagnosis trail
- [Docs/REPORT.md](Docs/REPORT.md): summary of evaluated models and performance
- [Docs/USAGE.md](Docs/USAGE.md): command examples for training, prediction, and metrics
- [Docs/PLOT_GUIDE.md](Docs/PLOT_GUIDE.md): plot interpretation guide

---

## What each neural net is for

### 1) Legacy single-span model: [neural_nets/neural_net.py](neural_nets/neural_net.py)

This was an earlier single-span EDFA workflow built for a simpler setup where the system was treated as one amplifier stage rather than a whole multispan topology. It uses a dense regression network on a feature set built from input spectra and metadata.

It is mainly useful for:

- older experiments
- single-stage spectra prediction
- basic baseline comparison before moving to multispan settings

The generated outputs live under the single-span saved model directory, typically under:

- [neural_nets/saved_models](neural_nets/saved_models).

### 2) Legacy multispan model: [neural_nets/multispan_net.py](neural_nets/multispan_net.py)

This workflow models multiple ROADM/EDFA stages together and is designed around the multispan dataset. It uses a multitask architecture in which each ROADM stage can have its own output head. The script is useful for understanding how the model behaves across stages and whether stage-specific prediction errors are significant.

This model is useful when you want to:

- compare stage-level prediction quality across different ROADM stages
- test a stage-aware architecture
- evaluate older multispan models before the encoder-based retrain

The older plotting scripts for this workflow are:

- [neural_nets/multispan_plots.py](neural_nets/multispan_plots.py)

### 3) Current main model: [neural_nets/encoder_net.py](neural_nets/encoder_net.py)

This is the primary workflow used in the current project evaluation. It is a dense encoder-style MLP designed to predict the gain spectrum or ripple spectrum from masked input spectra and scalar conditions.

It includes several practical training options such as:

- `--pin-weight`: reweights low-power samples
- `--edge-weight`: increases importance of edge channels
- `--shape-loss`: penalizes spectral-shape mismatch
- `--pos-enc`: exposes channel position information
- `--per-roadm-heads`: uses per-stage heads
- `--scale-target`: scales the output target for training
- `--ripple`: trains on ripple around the active-channel mean rather than raw gain

This is the workflow behind the current report in [Docs/REPORT.md](Docs/REPORT.md).

Important saved-output locations:

- [neural_nets/saved_models](neural_nets/saved_models)
- `encoder/` for the default current model
- `encoder_<variant>/` for variant experiments

Typical outputs include:

- `model.keras`
- `scaler.pkl`
- `metadata.json`
- `predictions/`

---

## What the repo does in practice

### Data loading

The datasets in [dataset/multispan](dataset/multispan) are JSON files containing samples for each topology. Each row typically contains:

- `source_file`
- `roadm`
- `pin_total`
- `target_gain`
- `num_active_channels`
- `input_spectra`
- `gain_spectra`
- `mask`

The loading code converts these into feature arrays ready for model training or prediction.

### Feature preparation

The current encoder workflow builds feature vectors from:

- masked spectra
- active-channel mask
- scalar metadata such as power and gain target values
- one-hot encoding for ROADM stage

This supports a 95-channel spectral regression task, with predictions summarized as the active-channel gain or ripple profile.

### Model training and evaluation

The main pipeline does the following:

1. load multispan dataset
2. split data into train/test subsets using a stratified split if applicable
3. scale feature inputs
4. train the encoder regression model
5. predict the held-out set
6. evaluate masked MAE, RMSE, p95, p99, and out-of-spec rate

The evaluation logic in [neural_nets/encoder_metrics.py](neural_nets/encoder_metrics.py) computes metrics only over active channels, which avoids artificially penalizing inactive channels.

### Plot generation

The plotting scripts produce diagnostic visuals that reveal the type and location of model error. These are key for identifying whether the model struggles with:

- specific ROADM stages
- spectral edge channels
- certain power ranges
- topology shift or unbalanced data

---

## Current plotting status

There are two families of plotting code in this repo:

- legacy plotting scripts for older models: [neural_nets/plots.py](neural_nets/plots.py) and [neural_nets/multispan_plots.py](neural_nets/multispan_plots.py)
- current plotting script for the encoder model: [neural_nets/encoder_plots_v2.py](neural_nets/encoder_plots_v2.py)

The current production workflow uses the encoder plots. The multispan and singlespan plot scripts are still useful as legacy reference tools, especially when working with older saved model outputs or comparing against the original model family. But for the active encoder workflow described in [Docs/REPORT.md](Docs/REPORT.md), [neural_nets/encoder_plots_v2.py](neural_nets/encoder_plots_v2.py) is the one to use.

---

## Quick start

Use a local virtual environment so the dependency versions stay isolated.

### 1) Create and activate a venv

From anywhere on the machine, replace the path with your local checkout:

```bash
cd /path/to/COSMOS-EDFA-Dataset
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the packages the project expects:

```bash
python -m pip install numpy pandas matplotlib scikit-learn tensorflow
```

If the environment already exists, just activate it:

```bash
cd /path/to/COSMOS-EDFA-Dataset
source .venv/bin/activate
```

### 2) Train a model

The main current workflow is the encoder model:

```bash
python neural_nets/encoder_net.py --train dataset/multispan --latent-dim 80 --inactive-loss-weight 0.1 --ripple
```

Optional variants:

```bash
python neural_nets/encoder_net.py --train dataset/multispan --variant-name pinweight --pin-weight
python neural_nets/encoder_net.py --train dataset/multispan --variant-name edgeweight --edge-weight
python neural_nets/encoder_net.py --train dataset/multispan --variant-name bestcombo --best-combo
```

### 3) Predict on a held-out split

```bash
python neural_nets/encoder_net.py --predict dataset/multispan/multispan_topology2.json --heldout
```

This saves the held-out results under the model’s prediction directory.

### 4) Compute metrics

```bash
python neural_nets/encoder_metrics.py neural_nets/saved_models/encoder/predictions/predictions_heldout_topology2.json
```

Optional cross-model comparison:

```bash
python neural_nets/encoder_metrics.py --compare "baseline:neural_nets/saved_models/encoder_baseline/predictions/predictions_heldout_topology2.json,encoder:neural_nets/saved_models/encoder/predictions/predictions_heldout_topology2.json,bestcombo:neural_nets/saved_models/encoder_bestcombo/predictions/predictions_heldout_topology2.json" --roadm
```

### 5) Generate plots

Current encoder graph workflow:

```bash
python neural_nets/encoder_plots_v2.py \
  --path neural_nets/saved_models/encoder/predictions/predictions_heldout_topology2.json \
  --output-dir neural_nets/saved_models/encoder/predictions/graphs/report/all
```

Specific roadm:

```bash
python neural_nets/encoder_plots_v2.py \
  --path neural_nets/saved_models/encoder/predictions/predictions_heldout_topology2.json \
  -r roadm_2_preamp \
  --output-dir neural_nets/saved_models/encoder/predictions/graphs/report/roadm_2_preamp
```

Specific channel:

```bash
python neural_nets/encoder_plots_v2.py \
  --path neural_nets/saved_models/encoder/predictions/predictions_heldout_topology2.json \
  -c 1 \
  --output-dir neural_nets/saved_models/encoder/predictions/graphs/report/ch1
```

Legacy single-span plots:

```bash
python neural_nets/plots.py -t all
```

Legacy multispan plots:

```bash
python neural_nets/multispan_plots.py -r all
```

---

## How to read the results

The project cares about more than a single average error number. The key metrics are typically:

- masked MAE
- RMSE
- p95 / p99
- max absolute error
- out-of-spec rate

This is especially important because edge channels and low-power samples can dominate the worst-case tail even when the average error looks acceptable.

The report in [Docs/REPORT.md](Docs/REPORT.md) is the clearest summary of the project’s final evaluation and should be used as the main reference when comparing model variants.

---

## Sample report excerpt

Below is a brief sample from [Docs/REPORT.md](Docs/REPORT.md):

```text
### Topology 2 (distribution shift — the deployment-relevant case)

| Model     | n    | MAE    | RMSE   | p95   | p99   | max   | % > 0.5 dB |
| --------- | ---- | ------ | ------ | ----- | ----- | ----- | ---------- |
| baseline  | 2367 | 0.0961 | 0.1277 | 0.261 | 0.400 | 0.660 | 0.108%     |
| encoder   | 2332 | 0.0777 | 0.1006 | 0.204 | 0.284 | 0.724 | 0.009%     |
| bestcombo | 2332 | 0.0778 | 0.1009 | 0.204 | 0.285 | 0.725 | 0.007%     |

### roadm_2_preamp (held-out topology 2)

| roadm          | baseline MAE / p99 | encoder MAE / p99 | bestcombo MAE / p99 |
| -------------- | ------------------ | ----------------- | ------------------- |
| roadm_2_preamp | 0.0597 / 0.191     | 0.0588 / 0.195    | 0.0587 / 0.197      |
```

This shows the main pattern: the model is usually strong on average, but a small number of edge-channel or rare operating-condition samples drive the worst-case tail.

---

## Supporting docs

- [Docs/PLAN.md](Docs/PLAN.md)
- [Docs/REPORT.md](Docs/REPORT.md)
- [Docs/USAGE.md](Docs/USAGE.md)
- [Docs/PLOT_GUIDE.md](Docs/PLOT_GUIDE.md)
- [dataset/multispan](dataset/multispan)

For the exact command-level workflow and best-practice usage, read [Docs/USAGE.md](Docs/USAGE.md) and [Docs/PLOT_GUIDE.md](Docs/PLOT_GUIDE.md).
