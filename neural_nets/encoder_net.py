import os
import json
import argparse
import pickle
import sys
import subprocess
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from tensorflow import keras
from tensorflow.keras import layers, models
from collections import Counter

try:
    from .helpers import extract_samples_from_multispan_json
except ImportError:
    from helpers import extract_samples_from_multispan_json

@tf.keras.utils.register_keras_serializable(package="Custom", name="MaskedEncoderModel")
class MaskedEncoderModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.masked_mae = tf.keras.metrics.Mean(name="masked_mae")

    @property
    def metrics(self):
        return [self.masked_mae]

    # ==== Opt-in inactive penalty for model ====
    def compute_masked_loss(self, Y, Y_pred, sw):
        sw = tf.cast(sw, Y_pred.dtype)
        binary = tf.cast(sw > 0.0, Y_pred.dtype)
        if bool(getattr(self, "shape_loss", False)):
            loss = combined_loss(Y, Y_pred)
        else:
            loss = tf.abs(Y - Y_pred)
        loss = tf.reduce_sum(loss * sw) / tf.maximum(tf.reduce_sum(sw), 1.0)
        w = float(getattr(self, "inactive_loss_weight", 0.0))
        if w > 0.0:
            inactive = 1.0 - binary
            loss += w * tf.reduce_sum(tf.abs(Y_pred * inactive)) / tf.maximum(tf.reduce_sum(inactive), 1.0)
        loss += tf.add_n(self.losses) if self.losses else loss * 0.0
        return loss

    def train_step(self, data):
        x, y, mask = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_masked_loss(y, y_pred, mask)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.masked_mae.update_state(loss)
        return {"loss": loss, "masked_mae": self.masked_mae.result()}

    def test_step(self, data):
        x, y, mask = data
        y_pred = self(x, training=False)
        loss = self.compute_masked_loss(y, y_pred, mask)
        self.masked_mae.update_state(loss)
        return {"loss": loss, "masked_mae": self.masked_mae.result()}

# ==== Custom loss functions ====
def gradient_loss(Y_true, Y_pred):
    grad_true = Y_true[:, 1:] - Y_true[:, :-1]
    grad_pred = Y_pred[:, 1:] - Y_pred[:, :-1]
    mask = tf.cast(Y_true != 0, tf.float32)
    grad_mask = mask[:, 1:] * mask[:, :-1]
    loss = tf.square(grad_true - grad_pred) * grad_mask
    return tf.pad(loss, [[0, 0], [0, 1]])

def cosine_shape_loss(Y_true, Y_pred):
    Y_true_n = tf.math.l2_normalize(Y_true, axis=-1)
    Y_pred_n = tf.math.l2_normalize(Y_pred, axis=-1)
    cos = tf.reduce_sum(Y_true_n * Y_pred_n, axis=-1)
    loss = 1.0 - cos
    return tf.repeat(tf.expand_dims(loss, axis=-1), tf.shape(Y_true)[-1], axis=-1)

def combined_loss(Y_true, Y_pred):
    mse = tf.square(Y_true - Y_pred)
    return mse + 0.3 * gradient_loss(Y_true, Y_pred) + 0.2 * cosine_shape_loss(Y_true, Y_pred) 

@tf.keras.utils.register_keras_serializable(package="Custom", name="HeadRoute")
class HeadRoute(layers.Layer):
    def __init__(self, num_heads, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.output_dim = output_dim
    
    def build(self, input_shape):
        self.heads = [layers.Dense(self.output_dim, activation="linear") for _ in range(self.num_heads)]
        super().build(input_shape)
    
    def call(self, args):
        enc, rid = args
        outs = tf.stack([h(enc) for h in self.heads], axis=1)
        idx = tf.stack([tf.range(tf.shape(rid)[0]), tf.reshape(rid, (-1,))], axis=1)
        return tf.gather_nd(outs, idx)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_heads": self.num_heads, "output_dim": self.output_dim})
        return cfg

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "saved_models" / "encoder"
MODEL_PATH = MODEL_DIR / "model.keras"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"
TARGET_SCALER_PATH = MODEL_DIR / "target_scaler.pkl"
PREDICTIONS_DIR = MODEL_DIR / "predictions"

def set_model_dir(name):
    global MODEL_DIR, MODEL_PATH, SCALER_PATH, METADATA_PATH, TARGET_SCALER_PATH, PREDICTIONS_DIR
    MODEL_DIR = HERE / "saved_models" / ("encoder_" + name if name else "encoder")
    MODEL_PATH = MODEL_DIR / "model.keras"
    SCALER_PATH = MODEL_DIR / "scaler.pkl"
    METADATA_PATH = MODEL_DIR / "metadata.json"
    TARGET_SCALER_PATH = MODEL_DIR / "target_scaler.pkl"
    PREDICTIONS_DIR = MODEL_DIR / "predictions"

# ==== Build Encoder Model ====
def build_encoder_model(input_dim, latent_dim=64, output_dim=95):
    inputs = layers.Input(shape=(input_dim,), name="input_features")
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.15)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    encoded = layers.Dense(latent_dim, activation="relu", name="encoder_latent")(x)
    outputs = layers.Dense(output_dim, activation="linear", name="output_spectrum")(encoded)
    return models.Model(inputs=inputs, outputs=outputs, name="encoder_net")

def build_multitask_encoder(input_dim, latent_dim, output_dim, roadm_categories):
    feat = layers.Input(shape=(input_dim,), name="features")
    rid = layers.Input(shape=(1,), dtype="int32", name="roadm_id")
    x = layers.Dense(256, activation='relu')(feat)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.15)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    encoded = layers.Dense(latent_dim, activation="relu", name="encoder_latent")(x)
    out = HeadRoute(len(roadm_categories), output_dim)([encoded, rid])
    return models.Model(inputs=[feat, rid], outputs=out, name="encoder_net")

def pin_weights(pins, width=5):
    pins = np.nan_to_num(np.asarray(pins, dtype=float), nan=-60.0)
    lo = np.floor(pins.min() / width) * width
    hi = np.ceil(pins.max() / width) * width
    hist, edges = np.histogram(pins, bins=np.arange(lo, hi + width, width))
    density = np.interp(pins, edges[:-1], hist.astype(float))
    w = 1.0 / np.sqrt(np.maximum(density, 1.0))
    return (w / w.mean()).astype(np.float32)

def edge_weights(n=95, gain=1.0):
    ch = np.arange(n)
    bump = np.exp(-(ch / 8.0) ** 2) + np.exp(-((ch - (n - 1)) / 8.0) ** 2)
    return (1.0 + gain * bump).astype(np.float32)

def pos_encoding(n=95, freqs=(1, 3)):
    ch = np.arange(n, dtype=float)
    cols = []
    for k in freqs:
        cols.append(np.sin(2 * np.pi * k * ch / n))
        cols.append(np.cos(2 * np.pi * k * ch / n))
    return np.stack(cols, axis=1).astype(np.float32)

def pos_features(n):
    pe = pos_encoding()
    return np.broadcast_to(pe.reshape(1, -1), (n, pe.size)).astype(np.float32)


# ==== Load dataset ====
def load_dataset(path, calculate_ripple=False):
    if os.path.isfile(path) and path.endswith('.json'):
        samples = extract_samples_from_multispan_json(path, calculate_ripple=calculate_ripple)
        return pd.DataFrame(samples)
    if os.path.isdir(path):
        json_files = sorted(str(p) for p in Path(path).rglob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {path}")
        all_samples = []
        for file in json_files:
            all_samples.extend(extract_samples_from_multispan_json(file, calculate_ripple=calculate_ripple))
        return pd.DataFrame(all_samples)
    raise FileNotFoundError(f"Cannot load dataset from {path}")


# ==== Prepare data ====
def prepare_features(df, roadm_categories=None, use_ripple=False):
    df = df.copy()

    df['input_spectra'] = df['input_spectra'].apply(lambda x: np.asarray(x, dtype=np.float32))
    df['mask'] = df['mask'].apply(lambda x: np.asarray(x, dtype=np.int8))
    df['gain_spectra'] = df['gain_spectra'].apply(lambda x: np.asarray(x, dtype=np.float32))

    # Masked spectra (95) + mask(95) = 190 spectral features
    X_spectra = np.stack(list(df['input_spectra'].values))
    X_spectra[np.isnan(X_spectra)] = -60.0
    X_mask = np.stack(list(df['mask'].values))
    X_spectra_masked = X_spectra * X_mask

    # Scalar features 
    scalar_cols = [
        "pin_total",
        "target_gain",
        "target_power",
        "voa_attenuation",
        "output_power",
        "voa_input_power",
        "voa_output_power"
    ]
    scalar_features = df[scalar_cols].fillna(0.0).to_numpy(dtype=np.float32)

    # One-hot encode roadm stage (12 unique prefixes)
    if roadm_categories is None:
        roadm_categories = sorted(df['roadm'].unique().tolist())
    df['roadm'] = pd.Categorical(df['roadm'], categories=roadm_categories)
    roadm_dummies = pd.get_dummies(df['roadm'], dtype=np.float32).to_numpy(dtype=np.float32)

    # Concatenate:
    # masked spectra (95) + mask (95) + pin (1) + target_gain (1) + target_power (1) + vao (1) + roadm one-hot (N)
    X = np.hstack([X_spectra_masked, X_mask, scalar_features, roadm_dummies])
    
    # Target Y
    target_col = "ripple_spectra" if use_ripple else "gain_spectra"
    Y_raw = np.stack(df[target_col].values)
    Y = np.nan_to_num(Y_raw, nan=0.0).astype(np.float32)

    return X, Y, roadm_categories, X_mask.astype(np.float32)

# ==== Model training ====
def train_model(train_path, epochs=80, batch_size=64, latent_dim=64, inactive_loss_weight=0.0, ripple=False,
pin_weight=False, edge_weight=False, shape_loss=False, pos_enc=False, per_roadm_heads=False, scale_target=False):
    print(f"Loading training data from {train_path}")
    df = load_dataset(train_path, calculate_ripple=ripple)
    if df.empty:
        raise ValueError("No data found in training set")
    X, Y, roadm_categories, X_mask = prepare_features(df, use_ripple=ripple)

    print(f"Features: {X.shape[1]} (spectra: 95, mask: 95, pin: 1, target_gain: 1, target_power: 1, voa: 1, roadm: {len(roadm_categories)})")
    print(f"ROADM categories: {roadm_categories}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Stratified split on ROADM / 5 dB pin bin so the low-pin stages land
    # proportionally in train/test. Sparse bins fall back to roadm-only
    pins = np.nan_to_num(df["pin_total"].to_numpy(), nan=-60.0)
    bins = (np.floor(pins / 5.0) * 5.0).astype(int)
    roadms = df["roadm"].astype(str).to_numpy()
    labels = np.array([f"{r}|{b}" for r, b in zip(roadms, bins)])
    cnt = Counter(labels)
    labels = np.array([l if cnt[l] >= 20 else l.split("|")[0] for l in labels])

    indices = np.arange(len(X_scaled))
    X_train, X_test, Y_train, Y_test, mask_train, mask_test, idx_train, test_idx = train_test_split(
        X_scaled,
        Y,
        X_mask,
        indices,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    print("test per roadm:", dict(Counter(roadms[test_idx])))
    if pos_enc:
        X_train = np.hstack([X_train, pos_features(len(X_train))])
        X_test = np.hstack([X_test, pos_features(len(X_test))])
    
    if per_roadm_heads:
        codes = pd.Categorical(df["roadm"], categories=roadm_categories).codes.astype("int32")
        codes_train = codes[idx_train][:, None]
        codes_test = codes[test_idx][:, None]
        model = build_multitask_encoder(X_train.shape[1], latent_dim, Y.shape[1], roadm_categories)
    else:
        codes_train = codes_test = None
        model = build_encoder_model(X_train.shape[1], latent_dim, Y.shape[1])
    model = MaskedEncoderModel(inputs=model.inputs, outputs=model.outputs, name=model.name)
    model.inactive_loss_weight = inactive_loss_weight
    model.shape_loss = shape_loss
    print("Model Building complete.")

    Y_test_raw = Y_test.copy()
    if scale_target:
        train_act = mask_train.astype(bool)
        means = np.nanmean(np.where(train_act, Y_train, np.nan), axis=0)
        stds = np.maximum(np.nanstd(np.where(train_act, Y_train, np.nan), axis=0), 1e-6)
        Y_train = np.where(train_act, (Y_train - means) / stds, 0.0).astype(np.float32)
        test_act = mask_test.astype(bool)
        Y_test = np.where(test_act, (Y_test - means) / stds, 0.0).astype(np.float32)
    
    sw_train = mask_train.astype(np.float32)
    if pin_weight:
        sw_train = sw_train * pin_weights(df["pin_total"].to_numpy()[idx_train])[:, None]
    if edge_weight:
        sw_train = sw_train * edge_weights()[None, :]

    model.compile(optimizer="adam")
    # ==== Train the model ====
    model.fit(
        [X_train, codes_train] if per_roadm_heads else X_train,
        Y_train,
        sample_weight=sw_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    preds = model.predict([X_test, codes_test] if per_roadm_heads else X_test, verbose=0)
    preds_eval = (preds * stds + means) if scale_target else preds
    mae = float(np.sum(np.abs(Y_test_raw - preds_eval) * mask_test) / np.maximum(np.sum(mask_test), 1.0))
    print(f"Test MAE: {mae:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "test_indices.json", "w") as f:
        json.dump([int(i) for i in test_idx], f)

    # Per-file held-out indices so --heldout works on a single topology file
    # After multi-topology training (global test_indices no longer maps 1:1)
    is_test = np.zeros(len(df), dtype=bool)
    is_test[test_idx] = True
    src = df["source_file"].to_numpy()
    per_file = {}
    for fname in df["source_file"].unique():
        f_idx = np.flatnonzero(src == fname)
        per_file[fname] = [int(i) for i in f_idx[is_test[f_idx]] - f_idx[0]]
    with open(MODEL_DIR / "test_indices_per_file.json", "w") as f:
        json.dump(per_file, f, indent=2)
    
    model.save(MODEL_PATH)

    with open(SCALER_PATH, "wb") as handle:
        pickle.dump(scaler, handle)

    if scale_target:
        with open(TARGET_SCALER_PATH, "wb") as handle:
            pickle.dump({"mean": means, "std": stds}, handle)

    metadata = {
        "input_dim": int(X_train.shape[1]),
        "output_dim": int(Y.shape[1]),
        "roadm_categories": roadm_categories,
        "latent_dim": latent_dim,
        "ripple": ripple,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "inactive_loss_weight": inactive_loss_weight,
        "epochs": epochs,
        "batch_size": batch_size,
        "stratified": True,
    }
    metadata["train_source"] = str(train_path)

    if pin_weight:
        metadata["pin_weight"] = True
    if edge_weight:
        metadata["edge_weight"] = True
    if shape_loss:
        metadata["shape_loss"] = True
    if per_roadm_heads:
        metadata["per_roadm_heads"] = True
    if pos_enc:
        metadata["pos_enc"] = True
    if scale_target:
        metadata["scale_target"] = True

    with open(METADATA_PATH, "w") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved model to {MODEL_PATH}")

# ==== Load saved models ====
def load_saved_model():
    """Load the saved model and scaler"""
    model = models.load_model(MODEL_PATH)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    return model, scaler, metadata


# ==== Predicting using loaded model ====
def predict(path, heldout=False):
    """Run inference on a new CSV file using the saved model."""
    print(f"\nLoading saved model from {MODEL_PATH}")
    model, scaler, metadata = load_saved_model()
    print(f"Loading evaluation data from {path}")

    # rebuild the same feature matrix as training
    use_ripple = bool(metadata.get("ripple", False))
    per_roadm_heads = bool(metadata.get("per_roadm_heads", False))
    pos_enc = bool(metadata.get("pos_enc", False))
    scale_target = bool(metadata.get("scale_target", False))

    df = load_dataset(path, calculate_ripple=use_ripple)
    X, Y_true, roadm_categories, X_mask = prepare_features(df, roadm_categories=metadata.get("roadm_categories"), use_ripple=use_ripple)
    X_scaled = scaler.transform(X)
    if pos_enc:
        X_scaled = np.hstack([X_scaled, pos_features(len(X_scaled))])
    
    if per_roadm_heads:
        codes = pd.Categorical(df["roadm"], categories=metadata.get("roadm_categories")).codes.astype("int32")[:, None]
        preds = model.predict([X_scaled, codes], verbose=0)
    else:
        preds = model.predict(X_scaled, verbose=0)
    
    if scale_target:
        with open(TARGET_SCALER_PATH, "rb") as f:
            ts = pickle.load(f)
        preds = preds * ts["std"] + ts["mean"]
    preds = preds * X_mask

    keep = slice(None)
    if heldout:
        idx_path = MODEL_DIR / "test_indices.json"
        per_file_path = MODEL_DIR / "test_indices_per_file.json"
        if per_file_path.exists() and df["source_file"].nunique() == 1:
            fname = df["source_file"].iloc[0]
            keep = np.array(json.load(open(per_file_path)).get(fname, []), dtype=int)
            if keep.size == 0:
                print("WARNING: file not in training held-out map; keeping all records")
                keep = slice(None)
        elif idx_path.exists():
            keep = np.array(json.load(open(idx_path)), dtype=int)
        else:
            n = len(df)
            if n != int(metadata["train_samples"]) + int(metadata["test_samples"]):
                raise ValueError("--heldout requires the same topology used for training"
                "(or a saved test_indices.json)"
                )
            _, keep = train_test_split(np.arange(n), test_size=0.2, random_state=42)
            print("WARNING: reconstructing split from random_state=42; retrain to save test_indices.json")
        preds, Y_true, X_mask = preds[keep], Y_true[keep], X_mask[keep]
        df = df.iloc[keep].reset_index(drop=True)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    pred_cols = [f"pred_gain_ch{i+1}" for i in range(preds.shape[1])]
    true_cols = [f"true_gain_ch{i+1}" for i in range(Y_true.shape[1])]

    result = pd.DataFrame(preds, columns=pred_cols)
    true_df = pd.DataFrame(Y_true, columns=true_cols)
    meta = df[["source_file", "pin_total", "num_active_channels", "roadm", "target_gain"]].reset_index(drop=True)
    output_df = pd.concat([meta, result, true_df], axis=1)
    suffix = "_heldout" if heldout else ""
    output_df.to_csv(PREDICTIONS_DIR / f"predictions{suffix}.csv", index=False)
    print(f"Saved {len(result)} rows")

    records = []
    for idx, row in output_df.iterrows ():
        records.append({
            "sample_index": int(idx),
            "source_file": row["source_file"],
            "pin_total": float(row["pin_total"]),
            "num_active_channels": int(row["num_active_channels"]),
            "roadm": row["roadm"],
            "target_gain": float(row["target_gain"]),
            "target_type": "ripple" if use_ripple else "gain",
            "mask": X_mask[idx].tolist(),
            "predicted_gain_spectra": preds[idx].tolist(),
            "true_gain_spectra": Y_true[idx].tolist()
        })
        
    with open(PREDICTIONS_DIR / f"predictions{suffix}.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {len(records)} records to {PREDICTIONS_DIR / f'predictions{suffix}.json'}")

FLAGS = [
    ("--pin-weight", "pinweight"),
    ("--edge-weight", "edgeweight"),
    ("--shape-loss", "shapeloss"),
    ("--pos-enc", "posenc"),
    ("--per-roadm-heads", "roadmheads"),
    ("--scale-target", "scaletarget"),
]

def eval_heldout_mae(tag):
    from encoder_metrics import collect, masked_mae
    d = HERE / "saved_models" / ("encoder_" + tag if tag else "encoder") / "predictions" / "predictions_heldout.json"
    err, m = collect(json.load(open(d)))
    return masked_mae(err, m)

def best_combo(train_path, epochs=80, batch_size=64, latent_dim=64, inactive_loss_weight=0.0, ripple=False):
    topo2 =str(Path(train_path) / "multispan_topology2.json")
    base = [sys.executable, __file__, "--train", train_path, "--epochs", str(epochs), "--batch-size", str(batch_size), 
    "--latent-dim", str(latent_dim), "--inactive-loss-weight", str(inactive_loss_weight)]
    if ripple:
        base.append("--ripple")
    pred = [sys.executable, __file__, "--predict", topo2, "--heldout"]

    meta_path = HERE / "saved_models" / "encoder" / "metadata.json"
    if meta_path.exists() and json.load(open(meta_path)).get("train_source") == train_path:
        print("reusing baseline model from saved_models.encoder")
    else:
        subprocess.run(base, check=True)
    subprocess.run(pred, check=True)
    base_mae = eval_heldout_mae("")
    print(f"baseline topo2 held-out MAE: {base_mae:.4f}")

    wins = []
    for flag, tag in FLAGS:
        if not (HERE / "saved_models" / f"encoder_{tag}" / "model.keras").exists():
            subprocess.run(base + [flag, "--variant-name", tag], check=True)
        subprocess.run(pred + ["--variant-name", tag], check=True)
        mae = eval_heldout_mae(tag)
        keep = mae < base_mae
        print(f"{tag:14s} MAE={mae:.4f} {'KEEP' if keep else 'drop'}")
        if keep:
            wins.append(flag)
    print("winning flags:", wins)

    subprocess.run(base + ["--variant-name", "bestcombo"] + wins, check=True)
    subprocess.run(pred + ["--variant-name", "bestcombo"], check=True)
    combo_mae = eval_heldout_mae("bestcombo")
    print(f"bestcombo topo2 held-out MAE: {combo_mae:.4f} (baseline {base_mae:.4f})")

def main():
    # ==== CLI ====
    parser = argparse.ArgumentParser(description="Train or predict with an encoder-only multispan model")
    parser.add_argument("--train", default=None, help="Multispan JSON file or directory for training")
    parser.add_argument("--predict", default=None, help="Multispan JSON file or directory for prediction")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--latent-dim", type=int, default=80)
    parser.add_argument("--inactive-loss-weight", type=float, default=0.0, help="Weight to force inactive-channel predictions toward 0 (opt-in, requires retraining)")
    parser.add_argument("--ripple", action="store_true", help="Load gain spectra as ripple around the active gain mean")
    parser.add_argument("--heldout", action="store_true", help="Only save predictions for the held-out split")
    parser.add_argument("--pin-weight", action="store_true", help="Reweight loss by inverse pin density (5 dB bins)")
    parser.add_argument("--edge-weight", action="store_true", help="Up-weight edge channels (ch1, ch70-95)")
    parser.add_argument("--shape-loss", action="store_true", help="Use MSE + 0.3*gradient + 0.2*cosine shape loss")
    parser.add_argument("--pos-enc", action="store_true", help="Add sin/cos channel positional encoding")
    parser.add_argument("--per-roadm-heads", action="store_true", help="Per-roadm output heads")
    parser.add_argument("--scale-target", action="store_true", help="Scale Y by per-channel active-only stats")
    parser.add_argument("--variant-name", default="", help="Save/load under saved_models/encoder_<name>")
    parser.add_argument("--best-combo", action="store_true", help="Auto A/B single-factor variants, then train combined winners")
    args = parser.parse_args()

    set_model_dir(args.variant_name)

    if args.best_combo:
        if not args.train:
            raise SystemExit("--best-combo requires --train")
        best_combo(args.train, epochs=args.epochs, batch_size=args.batch_size, latent_dim=args.latent_dim, inactive_loss_weight=args.inactive_loss_weight, 
                   ripple=args.ripple)
        return
    
    if args.train:
        train_model(args.train, epochs=args.epochs, batch_size=args.batch_size, latent_dim=args.latent_dim, inactive_loss_weight=args.inactive_loss_weight, 
        ripple=args.ripple, pin_weight=args.pin_weight, edge_weight=args.edge_weight, shape_loss=args.shape_loss, per_roadm_heads=args.per_roadm_heads,
        pos_enc=args.pos_enc, scale_target=args.scale_target)

    if args.predict:
        predict(args.predict, heldout=args.heldout)

if __name__ == '__main__':
    main()