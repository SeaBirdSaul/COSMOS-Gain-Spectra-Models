import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PREDICTIONS_PATH = Path(__file__).parent / "saved_models" / "singlespan" / "predictions" / "predictions.json"
OUTPUT_PATH = Path(__file__).parent / "saved_models" / "singlespan" / "predictions" / "graphs" 

def load_data(channel_type):
    with open(PREDICTIONS_PATH) as f:
        data = json.load(f)

    if channel_type != "all":
        data = [d for d in data if d["open_channel_type"] == channel_type]
        if not data:
            raise ValueError(f"No samples found for channel type: {channel_type}")

    # ==== Build some arrays ====
    channels = np.arange(1, 96)
    Y_pred = np.array([d["predicted_gain_spectra"] for d in data])
    Y_true = np.array([d["true_gain_spectra"] for d in data])
    error = Y_pred - Y_true
    abs_error = np.abs(error)
    pin_total = np.array([d["pin_total"] for d in data])
    ch_types = np.array([d["open_channel_type"] for d in data])
    unique_types = sorted(set(ch_types))
    masks = []
    for d in data:
        mask = d.get("mask")
        if mask is None:
            masks.append(np.ones(len(d.get("true_gain_spectra", [])), dtype=bool))
        else:
            masks.append(np.asarray(mask, dtype=bool))

    return {
        "channels": channels,
        "Y_pred": Y_pred,
        "Y_true": Y_true,
        "error": error,
        "abs_error": abs_error,
        "pin_total": pin_total,
        "ch_types": ch_types,
        "unique_types": unique_types,
        "masks": masks,
        "channel_type": channel_type
    }

def _active_ripple(values, mask=None):
    values = np.asarray(values, dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0)
    if mask is None:
        mask = np.ones(values.shape[0], dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != values.shape[0]:
            mask = np.ones(values.shape[0], dtype=bool)
    active = values[mask]
    if active.size == 0:
        return np.zeros_like(values, dtype=np.float64)
    return values - np.mean(active)

# 1. Predicted vs True Gain Spectral (one sample per channel type)
def plot_predicted_vs_true_gain(d):
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Prediction vs True Gain Spectra", fontsize=14, fontweight="bold")

    if d["channel_type"] == "all":
        for ct in d["unique_types"]:
            idx = np.where(d["ch_types"] == ct)[0][0]
            ax.plot(d["channels"], d["Y_true"][idx], label=f"True ({ct})", alpha=0.8)
            ax.plot(d["channels"], d["Y_pred"][idx], "--", label=f"Pred ({ct})", alpha=0.8)
        ax.set_title("One sample per channel type")
    else:
        for i in range(min(3, len(d["Y_true"]))):
            ax.plot(d["channels"], d["Y_true"][i], label=f"True (sample {i})", alpha=0.8)
            ax.plot(d["channels"], d["Y_pred"][i], "--", label=f"Pred (sample {i})", alpha=0.8)
        ax.set_title(f"Type: {d['channel_type']}")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Gain (dB)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    return fig

# 2. Error Heatmap
def plot_error_heatmap(d):
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Prediction Error Heatmap (pred - true)", fontsize=14, fontweight="bold") 

    if d["channel_type"] == "all":
        order = np.argsort(d["ch_types"])
        error_sorted = d["error"][order]
        ch_sorted = d["ch_types"][order]
    else:
        order = np.arange(len(d["error"]))
        error_sorted = d["error"]
        ch_sorted = d["ch_types"]

    im = ax.imshow(error_sorted, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Sample")
    plt.colorbar(im, ax=ax, label="Error (dB)")

    # Annotate group boundaries
    if d["channel_type"] == "all":
        prev, boundary = ch_sorted[0], 0
        for i, ct in enumerate(ch_sorted):
            if ct!= prev:
                ax.axhline(y=i, color="black", linewidth=0.8)
                ax.text(96, (boundary + i) / 2, prev, fontsize=6, va="center", ha="left")
                prev = ct
                boundary = i
        ax.text(96, (boundary + len(order)) / 2, prev, fontsize=6, va="center", ha="left")

    return fig

# 3. Error Distribution by Channel Type
def plot_error_distribution(d):
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Error Distribution by Channel Type", fontsize=14, fontweight="bold")

    by_type = [d["abs_error"][d["ch_types"] == ct].flatten() for ct in d["unique_types"]]
    bp = ax.boxplot(by_type, tick_labels=[ct.replace("_", "\n") for ct in d["unique_types"]], patch_artist=True)
    colors = plt.cm.Set3(np.linspace(0, 1, len(d["unique_types"])))
    for patch, color in zip(bp["boxes"],colors):
        patch.set_facecolor(color)

    ax.set_ylabel("Absolute Error (dB)")
    ax.set_xlabel("Channel Type")
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(True, alpha=0.3, axis="y")
    return fig

# 4. Predicted vs True Scatter
def plot_scatter(d):
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.suptitle("Predicted vs True Scatter", fontsize=14, fontweight="bold")

    if d["channel_type"] == "all":
        for ct in d["unique_types"]:
            mask = d["ch_types"] == ct
            ax.scatter(d["Y_true"][mask], d["Y_pred"][mask], s=4, alpha=0.4, label=ct)
    else:
        ax.scatter(d["Y_true"], d["Y_pred"], s=4, alpha=0.4)
        ax.set_title(f"Type: {d['channel_type']}")

    lims = (min(d["Y_true"].min(), d["Y_pred"].min()) - 0.5, max(d["Y_true"].max(), d["Y_pred"].max()) + 0.5)
    ax.plot(lims, lims, "k--", alpha=0.5, label="Perfect")
    ax.set_xlabel("True Gain (dB)")
    ax.set_ylabel("Predicted Gain (dB)")
    ax.legend(fontsize=6, markerscale=3)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(lims)
    ax.set_ylim(lims)   
    return fig
    

# 5. Ripple Comparison
def plot_ripple_comparison(d):
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Ripple Comparison (Predicted vs True)", fontsize=14, fontweight="bold")

    if d["channel_type"] == "all":
        for ct in d["unique_types"]:
            idx = np.where(d["ch_types"] == ct)[0][0]
            mask = d["masks"][idx]
            true_ripple = _active_ripple(d["Y_true"][idx], mask)
            pred_ripple = _active_ripple(d["Y_pred"][idx], mask)
            ax.plot(d["channels"], true_ripple, label=f"True ripple ({ct})", alpha=0.8)
            ax.plot(d["channels"], pred_ripple, "--", label=f"Pred ripple ({ct})", alpha=0.8)
        ax.set_title("One sample per channel type")
    else:
        idx = 0
        mask = d["masks"][idx] if idx < len(d["masks"]) else None
        true_ripple = _active_ripple(d["Y_true"][idx], mask)
        pred_ripple = _active_ripple(d["Y_pred"][idx], mask)
        ax.plot(d["channels"], true_ripple, label="True ripple", alpha=0.8)
        ax.plot(d["channels"], pred_ripple, "--", label="Pred ripple", alpha=0.8)
        ax.set_title(f"Type: {d['channel_type']}")

    ax.set_xlabel("Channel")
    ax.set_ylabel("Ripple (dB)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    return fig

# 6. Mean Error vs Input Power
def plot_error_vs_power(d):
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.suptitle("Error vs Input Power", fontsize=14, fontweight="bold")

    mae_per_sample = d["abs_error"].mean(axis=1)

    if d["channel_type"] == "all":
        for ct in d["unique_types"]:
            mask = d["ch_types"] == ct
            ax.scatter(d["pin_total"][mask], mae_per_sample[mask], s=20, alpha=0.6, label=ct)
        ax.legend(fontsize=6)
    else: 
        ax.scatter(d["pin_total"], mae_per_sample, s=20, alpha=0.6)
        ax.set_title(f"Type: {d['channel_type']}")

    ax.set_xlabel("Input Power (dB)")
    ax.set_ylabel("Mean Absolute Error (dB)")
    ax.grid(True, alpha=0.3)
    return fig

def main():
    parser = argparse.ArgumentParser(description="Generate prediction analysis plots")
    parser.add_argument(
        "-t", "--channel-type",
        default="fully_loaded_channel_wdm",
        help="Channel type to plot, or 'all' for all types (default: fully_loaded_channel_wdm)"
    )
    args = parser.parse_args()

    d = load_data(args.channel_type)

    plots = [
        ("01_predicted_vs_true_gain.png", plot_predicted_vs_true_gain),
        ("02_error_heatmap.png", plot_error_heatmap),
        ("03_error_distrinutiom_by_type.png", plot_error_distribution),
        ("04_predicted_vs_true_scatter.png", plot_scatter),
        ("05_ripple_comparison.png", plot_ripple_comparison),
        ("06_error_vs_input_power.png", plot_error_vs_power),
    ]

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    for filename, plot_fn in plots:
        fig = plot_fn(d)
        out = OUTPUT_PATH / filename
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")

if __name__ == "__main__":
    main()