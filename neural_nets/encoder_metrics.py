import argparse
import json
import numpy as np
from pathlib import Path

PREDICTIONS_PATH = Path(__file__).parent / "saved_models" / "encoder" / "predictions" / "predictions.json"

def collect(data):
    yp = np.array([r["predicted_gain_spectra"] for r in data], dtype=float)
    yt = np.array([r["true_gain_spectra"] for r in data], dtype=float)
    m = np.array([np.asarray(r["mask"], dtype=bool) for r in data])
    err = np.where(m, yp - yt, np.nan)
    return err, m

def masked_mae(err, m):
    return float(np.nansum(np.abs(err)) / m.sum())

def summary(err, m, data):
    a = np.abs(err[m])
    print(f"samples: {len(data)} active channels: {int(m.sum())}")
    print(f"masked MAE : {masked_mae(err, m):.4f} dB")
    print(f"RMSE       : {float(np.sqrt(np.nanmean(err[m] ** 2))):.4f} dB")
    print(f"p95 / p99  : {float(np.percentile(a, 95)):.4f} / {float(np.percentile(a, 99)):.4f} dB")
    print(f"max |err|  : {float(np.nanmax(a)):.4f} dB")
    print(f"% channels > 0.5 dB : {100 * float((a > 0.5).mean()):.3f}%")

def by_roadm(err, m, data):
    print("per-roadm masked MAE:")
    for rm in sorted({r["roadm"] for r in data}):
        idx = np.array([i for i, r in enumerate(data) if r["roadm"] == rm])
        print(f"  {rm:20s} mae={masked_mae(err[idx], m[idx]):.4f} "
              f"p99={float(np.percentile(np.abs(err[idx][m[idx]]), 99)):.4f}")

def by_pin(err, m, data, width=1):
    pins = np.array([r["pin_total"] for r in data])
    print("per-pin-bin masked MAE:")
    for lo in range(int(np.floor(pins.min())), int(np.ceil(pins.max())), width):
        idx = (pins >= lo) & (pins < lo + width)
        if not idx.any():
            continue
        print(f"  pin[{lo:4d},{lo + width:4d}] n={int(m[idx].sum()):6d} "
        f"mae={masked_mae(err[idx], m[idx]):.4f}")

def metrics(err, m, data):
    a = np.abs(err[m])
    return {
        "samples": len(data),
        "mae": masked_mae(err, m),
        "rmse": float(np.sqrt(np.nanmean(err[m] ** 2))),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(np.nanmax(a)),
        "pct_gt_0p5": 100 * float((a > 0.5).mean()),
    }

def row(v):
    return (f"| {v['name']:26s} | {v['samples']:5d} | {v['mae']:.4f} | {v['rmse']:.4f} "
            f"| {v['p95']:.3f} | {v['p99']:.3f} | {v['max']:.3f} | {v['pct_gt_0p5']:.3f}% |")

def compare(pairs):
    print("| Model                    | n     | MAE    | RMSE   | p95   | p99   | max   | % > 0.5 dB |")
    print("| ------------------------ | ----- | ------ | ------ | ----- | ----- | ----- | ---------- |")
    for name, path in pairs:
        with open(path) as f:
            data = json.load(f)
        err, m = collect(data)
        v = metrics(err, m, data)
        v["name"] = name
        print(row(v))

def by_roadm_table(pairs):
    roadms = sorted({r["roadm"] for r in json.load(open(pairs[0][1]))})
    print(f"{'roadm':22s} " + " ".join(f"{name:>14s}" for name, _ in pairs))
    print(f"{'':22s} " + " ".join(f"{'MAE / p99':>14s}" for _ in pairs))
    for rm in roadms:
        cells = []
        for _, path in pairs:
            with open(path) as f:
                data = [r for r in json.load(f) if r["roadm"] == rm]
            err, m = collect(data)
            a = np.abs(err[m])
            cells.append(f"{masked_mae(err, m):.4f}/{float(np.percentile(a, 99)):.3f}")
        print(f"{rm:22s} " + " ".join(f"{c:>14s}" for c in cells))

def main():
    ap = argparse.ArgumentParser(description="Masked error metrics for encoder predictions")
    ap.add_argument("path", nargs="?", default=str(PREDICTIONS_PATH))
    ap.add_argument("--compare", default="", help="comma-separated name:path pairs; print cross-model markdown tables")
    ap.add_argument("--roadm", action="store_true", help="with --compare, also print the per-roadm MAE/p99 table")
    args = ap.parse_args()
    if args.compare:
        pairs = [tuple(p.split(":", 1)) for p in args.compare.split(",")]
        compare(pairs)
        if args.roadm:
            print()
            by_roadm_table(pairs)
        return
    with open(args.path) as f:
        data = json.load(f)
    err, m = collect(data)
    summary(err, m, data)
    by_roadm(err, m, data)
    by_pin(err, m, data)
    mae = np.nansum(np.abs(err), axis=0) / m.sum(axis=0)
    print("per channel MAE (1..95):")
    print(" " + " ".join(f"ch{i+1}={mae[i]:.3f}" for i in range(len(mae))))

if __name__ == "__main__":
    main()