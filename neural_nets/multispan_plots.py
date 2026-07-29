import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PREDICTIONS_PATH = Path(__file__).parent / "saved_models" / "multispan" / "predictions" / "predictions.json"
OUTPUT_PATH = Path(__file__).parent / "saved_models" / "multispan" / "predictions" / "graphs" 

def load_data(roadm_filter=None, channel=None):
    with open(PREDICTIONS_PATH) as f:
        data = json.load(f)

    if roadm_filter and roadm_filter != "all":
        data = [d for d in data if d["roadm"] in roadm_filter]
        if not data:
            raise ValueError(f"No samples found for roadm: {roadm_filter}")

    # ==== Build some arrays ====
    channels = np.arange(1, 96)
    Y_pred = np.array([d["predicted_gain_spectra"] for d in data])
    Y_true = np.array([d["true_gain_spectra"] for d in data])
    error = Y_pred - Y_true
    abs_error = np.abs(error)
    pin_total = np.array([d["pin_total"] for d in data])
    roadms = np.array([d["roadm"] for d in data])
    unique_types = sorted(set(roadms))
    masks = []
    for d in data:
        mask = d.get("mask")
        if mask is None:
            masks.append(np.ones(len(d.get("true_gain_spectra", [])), dtype=bool))
        else:
            masks.append(np.asarray(mask, dtype=bool))

    channel_idx = None
    Y_pred_full = None
    Y_true_full = None
    if channel is not None:
        channel_idx = channel - 1
        if channel_idx < 0 or channel_idx >= Y_pred.shape[1]:
            raise ValueError(f"Channel {channel} out of range (1-95)")
        Y_pred_full = Y_pred.copy()
        Y_true_full = Y_true.copy()
        Y_pred = Y_pred[:, channel_idx]
        Y_true = Y_true[:, channel_idx]
        error = error[:, channel_idx]
        abs_error = abs_error[:, channel_idx]

    return {
        "channels": channels,
        "Y_pred": Y_pred,
        "Y_true": Y_true,
        "Y_pred_full": Y_pred_full,
        "Y_true_full": Y_true_full,
        "error": error,
        "abs_error": abs_error,
        "pin_total": pin_total,
        "roadms": roadms,
        "unique_types": unique_types,
        "masks": masks,
        "roadm_filter": roadm_filter,
        "channel_num": channel,
        "channel_idx": channel_idx,
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

    if d.get("channel_num"):
        ax.set_title(f"Ch {d['channel_num']} - Predicted vs True Gain by ROADM", fontsize=14, fontweight="bold")
        x = np.arange(len(d["unique_types"]))
        width = 0.35
        true_means, pred_means = [], []
        true_stds, pred_stds = [], []
        for ct in d["unique_types"]:
            mask = d["roadms"] == ct
            true_means.append(d["Y_true"][mask].mean())
            pred_means.append(d["Y_pred"][mask].mean())
            true_stds.append(d["Y_true"][mask].std())
            pred_stds.append(d["Y_pred"][mask].std())
        ax.bar(x - width / 2, true_means, width, yerr=true_stds, capsize=3, label="True", alpha=0.8)
        ax.bar(x + width / 2, pred_means, width, yerr=pred_stds, capsize=3, label="Pred", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([ct.replace("_", "\n") for ct in d["unique_types"]], fontsize=7)
        ax.set_ylabel("Gain (dB)")
    else:
        fig.suptitle("Prediction vs True Gain Spectra", fontsize=14, fontweight="bold")

        if d["roadm_filter"] == "all" or isinstance(d["roadm_filter"], list):
            for ct in d["unique_types"]:
                idx = np.where(d["roadms"] == ct)[0][0]
                ax.plot(d["channels"], d["Y_true"][idx], label=f"True ({ct})", alpha=0.8)
                ax.plot(d["channels"], d["Y_pred"][idx], "--", label=f"Pred ({ct})", alpha=0.8)
            ax.set_title("One sample per ROADM stage")
        else:
            for i in range(min(3, len(d["Y_true"]))):
                ax.plot(d["channels"], d["Y_true"][i], label=f"True (sample {i})", alpha=0.8)
                ax.plot(d["channels"], d["Y_pred"][i], "--", label=f"Pred (sample {i})", alpha=0.8)
            ax.set_title(f"Type: {d['roadm_filter']}")
        ax.set_xlabel("Channel")
        ax.set_ylabel("Gain (dB)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    return fig

# 2. Error Heatmap
def plot_error_heatmap(d):
    fig, ax = plt.subplots(figsize=(12, 6))

    if d.get("channel_num"):
        ax.set_title(f"Ch {d['channel_num']} - Predicted Error per Sample", fontsize=14, fontweight="bold")
        colors = []
        color_map = plt.cm.tab10
        unique = d["unique_types"]
        for r in d["roadms"]:
            colors.append(color_map(unique.index(r) % 10))
        ax.bar(np.arange(len(d["error"])), d["error"], color=colors, width=1.0, alpha=0.8)
        handles = [plt.Rectangle((0, 0), 1, 1, fc=color_map(i % 10), alpha=0.8) for i in range(len(unique))]
        ax.legend(handles, [u.replace("_", "\n") for u in unique], fontsize=6, loc="upper right")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Error (dB)")
    else:
        fig.suptitle("Prediction Error Heatmap (pred - true)", fontsize=14, fontweight="bold") 
        if d["roadm_filter"] == "all" or isinstance(d["roadm_filter"], list):
            order = np.argsort(d["roadms"])
            error_sorted = d["error"][order]
            ch_sorted = d["roadms"][order]
        else:
            order = np.arange(len(d["error"]))
            error_sorted = d["error"]
            ch_sorted = d["roadms"]

        im = ax.imshow(error_sorted, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
        ax.set_xlabel("Channel")
        ax.set_ylabel("Sample")
        plt.colorbar(im, ax=ax, label="Error (dB)")

        # Annotate group boundaries
        if d["roadm_filter"] == "all":
            prev, boundary = ch_sorted[0], 0
            for i, ct in enumerate(ch_sorted):
                if ct!= prev:
                    ax.axhline(y=i, color="black", linewidth=0.8)
                    ax.text(96, (boundary + i) / 2, prev, fontsize=6, va="center", ha="left")
                    prev = ct
                    boundary = i
            ax.text(96, (boundary + len(order)) / 2, prev, fontsize=6, va="center", ha="left")
    ax.grid(True, alpha=0.3)
    return fig

# 3. Error Distribution by Channel Type
def plot_error_distribution(d):
    fig, ax = plt.subplots(figsize=(12, 6))

    if d.get("channel_num"):
        ax.set_title(f"Ch {d['channel_num']} - Error Distribution by ROADM", fontsize=14, fontweight="bold")
    else:
        fig.suptitle("Error Distribution bt Channel Type", fontsize=14, fontweight="bold")

    by_type = [d["abs_error"][d["roadms"] == ct].flatten() for ct in d["unique_types"]]
    bp = ax.boxplot(by_type, tick_labels=[ct.replace("_", "\n") for ct in d["unique_types"]], patch_artist=True)
    colors = plt.cm.Set3(np.linspace(0, 1, len(d["unique_types"])))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_ylabel("Absolute Error (dB)")
    ax.set_xlabel("Channel Type")
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(True, alpha=0.3, axis="y")
    return fig

# 4. Predicted vs True Scatter
def plot_scatter(d):
    fig, ax = plt.subplots(figsize=(8, 8))

    if d.get("channel_num"):
        ax.set_title(f"Ch {d['channel_num']} - Predicted vs True Scatter", fontsize=14, fontweight="bold")
    else:
        fig.suptitle("Predicted vs True Scatter", fontsize=14, fontweight="bold")

    if d["roadm_filter"] == "all" or isinstance(d["roadm_filter"], list):
        for ct in d["unique_types"]:
            mask = d["roadms"] == ct
            ax.scatter(d["Y_true"][mask], d["Y_pred"][mask], s=4, alpha=0.4, label=ct)
    else:
        ax.scatter(d["Y_true"], d["Y_pred"], s=4, alpha=0.4)
        ax.set_title(f"Type: {d['roadm_filter']}")

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

    if d.get("channel_num"):
        ax.set_title(f"Ch {d['channel_num']} - Ripple Comparison by ROADM", fontsize=14, fontweight="bold")
        labels = [ct.replace("_", "\n") for ct in d["unique_types"]]
        true_rip, pred_rip = [], []
        true_rip_std, pred_rip_std = [], []
        for ct in d["unique_types"]:
            idxs = np.where(d["roadms"] == ct)[0]
            ch = d["channel_idx"]
            tr, pr = [], []
            for i in idxs:
                m = d["masks"][i].astype(bool)
                active_true = d["Y_true_full"][i][m]
                active_pred = d["Y_pred_full"][i][m]
                tr.append(d["Y_true_full"][i][ch] - np.mean(active_true))
                pr.append(d["Y_pred_full"][i][ch] - np.mean(active_pred))
            true_rip.append(np.mean(tr))
            pred_rip.append(np.mean(pr))
            true_rip_std.append(np.std(tr))
            pred_rip_std.append(np.std(pr))
        x = np.arange(len(labels))
        ax.plot(x, true_rip, "o-", label="True", alpha=0.8)
        ax.plot(x, pred_rip, "s--", label="Pred", alpha=0.8)
        ax.fill_between(x, np.array(true_rip) - np.array(true_rip_std), np.array(true_rip) + np.array(true_rip_std), alpha=0.15)
        ax.fill_between(x, np.array(pred_rip) - np.array(pred_rip_std), np.array(pred_rip) + np.array(pred_rip_std), alpha=0.15)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("Ripple (dB)")
    else:
        fig.suptitle("Ripple Comparison (Predicted vs True)", fontsize=14, fontweight="bold")
        if d["roadm_filter"] == "all" or isinstance(d["roadm_filter"], list):
            for ct in d["unique_types"]:
                idx = np.where(d["roadms"] == ct)[0][0]
                mask = d["masks"][idx]
                true_ripple = _active_ripple(d["Y_true"][idx], mask)
                pred_ripple = _active_ripple(d["Y_pred"][idx], mask)
                ax.plot(d["channels"], true_ripple, label=f"True ripple ({ct})", alpha=0.8)
                ax.plot(d["channels"], pred_ripple, "--", label=f"Pred ripple ({ct})", alpha=0.8)
            ax.set_title("One sample per ROADM stage")
        else:
            idx = 0
            mask = d["masks"][idx] if idx < len(d["masks"]) else None
            true_ripple = _active_ripple(d["Y_true"][idx], mask)
            pred_ripple = _active_ripple(d["Y_pred"][idx], mask)
            ax.plot(d["channels"], true_ripple, label="True ripple", alpha=0.8)
            ax.plot(d["channels"], pred_ripple, "--", label="Pred ripple", alpha=0.8)
            ax.set_title(f"Type: {d['roadm_filter']}")
        ax.set_xlabel("Channel")
        ax.set_ylabel("Ripple (dB)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    return fig

# 6. Mean Error vs Input Power
def plot_error_vs_power(d):
    fig, ax = plt.subplots(figsize=(8, 8))

    if d.get("channel_num"):
        ax.set_title(f"Ch {d['channel_num']} - Error vs Input Power", fontsize=14, fontweight="bold")
        if d["roadm_filter"] == "all" or isinstance(d["roadm_filter"], list):
            for ct in d["unique_types"]:
                mask = d["roadms"] == ct
                ax.scatter(d["pin_total"][mask], d["abs_error"][mask], s=20, alpha=0.6, label=ct)
            ax.legend(fontsize=6)
        else:
            ax.scatter(d["pin_total"], d["abs_error"], s=20, alpha=0.6)
        ax.set_ylabel("Absolute Error (dB)")
    else:
        fig.suptitle("Error vs Input Power", fontsize=14, fontweight="bold")
        mae_per_sample = d["abs_error"].mean(axis=1)
        if d["roadm_filter"] == "all" or isinstance(d["roadm_filter"], list):
            for ct in d["unique_types"]:
                mask = d["roadms"] == ct
                ax.scatter(d["pin_total"][mask], mae_per_sample[mask], s=20, alpha=0.6, label=ct)
            ax.legend(fontsize=6)
        else: 
            ax.scatter(d["pin_total"], mae_per_sample, s=20, alpha=0.6)
            ax.set_title(f"Type: {d['roadm_filter']}")
        ax.set_ylabel("Mean Absolute Error (dB)")
    ax.set_xlabel("Input Power (dB)")
    ax.grid(True, alpha=0.3)
    return fig

def main():
    parser = argparse.ArgumentParser(description="Generate prediction analysis plots")
    parser.add_argument(
        "-r", "--roadm",
        default="all",
        help="Comma-separated ROADM stages (e.g. 'roadm_1_booster,roadm_2_preamp'), or 'all'"
    )
    parser.add_argument(
        "-c", "--channel", type=int, default=None,
        help="Plot for a single channel (1-95). Omit for spectral plots."
    )
    args = parser.parse_args()

    roadm_filter = args.roadm if args.roadm == "all" else [r.strip() for r in args.roadm.split(",")]
    
    d = load_data(roadm_filter, channel=args.channel)

    plots = [
        ("multispan_01_predicted_vs_true_gain.png", plot_predicted_vs_true_gain),
        ("multispan_02_error_heatmap.png", plot_error_heatmap),
        ("multispan_03_error_distrinutiom_by_type.png", plot_error_distribution),
        ("multispan_04_predicted_vs_true_scatter.png", plot_scatter),
        ("multispan_05_ripple_comparison.png", plot_ripple_comparison),
        ("multispan_06_error_vs_input_power.png", plot_error_vs_power),
    ]

    out = OUTPUT_PATH
    if args.channel:
        out = OUTPUT_PATH / f"ch{args.channel}"
    out.mkdir(parents=True, exist_ok=True)

    for filename, plot_fn in plots:
        fig = plot_fn(d)
        fig.savefig(out / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out / filename}")

if __name__ == "__main__":
    main()