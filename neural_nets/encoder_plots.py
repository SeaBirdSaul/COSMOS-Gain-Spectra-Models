import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PREDICTIONS_PATH = Path(__file__).parent / "saved_models" / "encoder" / "predictions" / "predictions.json"
OUTPUT_PATH = Path(__file__).parent / "saved_models" / "encoder" / "predictions" / "graphs" 

def load_data():
    with open(PREDICTIONS_PATH) as f:
        data = json.load(f)

    # ==== Build some arrays ====
    channels = np.arange(1, 96)
    Y_pred = np.array([d["predicted_gain_spectra"] for d in data], dtype=float)
    Y_true = np.array([d["true_gain_spectra"] for d in data], dtype=float)
    if "mask" in data[0]:
        mask = np.array([d["mask"] for d in data], dtype=float)
    else:
        mask = (Y_true != 0).astype(float)
    error = np.where(mask > 0, Y_pred - Y_true, np.nan)
    abs_error = np.abs(error)
    pin_total = np.array([d["pin_total"] for d in data], dtype=float)
    
    return {
        "channels": channels,
        "Y_pred": Y_pred,
        "Y_true": Y_true,
        "error": error,
        "abs_errors": abs_error,
        "pin_total": pin_total,
        "mask": mask,
    }

def plot_prediction_vs_true(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    idx = 0
    ax.plot(data["channels"], data["Y_true"][idx], label="True", alpha=0.8)
    ax.plot(data["channels"], data["Y_pred"][idx], "--", label="Pred", alpha=0.8)
    ax.set_title("Encoder Predicted vs True Gain Spectrum")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Gain (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig

def plot_error_distribution(data):
    channel_mean = np.nanmean(data["abs_errors"], axis=0)
    channel_q25, channel_q75 = np.nanpercentile(data["abs_errors"], [25, 75], axis=0)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data["channels"], channel_mean, label="Mediam abs errro", color="C0")
    ax.fill_between(data["channels"], channel_q25, channel_q75, alpha=0.25, label="25-75 percentile")
    ax.set_title("Channel Error Summary")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Absolute Error (dB)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

def plot_error_vs_power(data):
    mae_per_sample = np.nanmean(data["abs_errors"], axis=1)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(data["pin_total"], mae_per_sample, s=10, alpha=0.6)
    z = np.polyfit(data["pin_total"], mae_per_sample, 1)
    ax.plot(data["pin_total"], np.polyval(z, data["pin_total"]), color="red", lw=1)
    ax.set_title("Mean Absolute Error vs Input Power")
    ax.set_xlabel("Input Power (dB)")
    ax.set_ylabel("Mean Absolute Error (dB)")
    ax.grid(True, alpha=0.3)
    return fig

def main():
    parser = argparse.ArgumentParser(description="Generate plots for encoder predictions")
    parser.add_argument("--output-dir", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    data = load_data()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, plot_fn in [
        ("encoder_01_prediction_vs_true.png", plot_prediction_vs_true),
        ("encoder_02_error_distribution.png", plot_error_distribution),
        ("encoder_03_error_vs_power.png", plot_error_vs_power),
    ]:
        fig = plot_fn(data)
        fig.savefig(out_dir / name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_dir / name}")

if __name__ == "__main__":
    main()