import os
import json
import argparse
import pickle
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
    def compute_masked_loss(self, Y, Y_pred, mask):
        mask = tf.cast(mask, Y_pred.dtype)
        loss = tf.abs(Y - Y_pred) * mask
        loss = tf.reduce_sum(loss)
        mask_sum = tf.reduce_sum(mask)
        loss = loss / tf.maximum(mask_sum, 1.0)
        w = float(getattr(self, "inactive_loss_weight", 0.0))
        if w > 0.0:
            inactive = 1.0 - mask
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

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "saved_models" / "encoder"
MODEL_PATH = MODEL_DIR / "model.keras"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"
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

# ==== Load dataset ====
def load_dataset(path, calculate_ripple=False):
    if os.path.isfile(path) and path.endswith('.json'):
        samples = extract_samples_from_multispan_json(path, calculate_ripple=calculate_ripple)
        return pd.DataFrame(samples)
    if os.path.isdir(path):
        json_files = [str(p) for p in Path(path).rglob("*.json")]
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
def train_model(train_path, epochs=80, batch_size=64, latent_dim=64, inactive_loss_weight=0.0, ripple=False):
    print(f"Loading training data from {train_path}")
    df = load_dataset(train_path, calculate_ripple=ripple)
    if df.empty:
        raise ValueError("No data found in training set")
    X, Y, roadm_categories, X_mask = prepare_features(df, use_ripple=ripple)

    print(f"Features: {X.shape[1]} (spectra: 95, mask: 95, pin: 1, target_gain: 1, target_power: 1, voa: 1, roadm: {len(roadm_categories)})")
    print(f"ROADM categories: {roadm_categories}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, Y_train, Y_test, mask_train, mask_test = train_test_split(
        X_scaled,
        Y,
        X_mask,
        test_size=0.2,
        random_state=42,
    )

    _, test_idx = train_test_split(np.arange(len(X_scaled)), test_size=0.2, random_state=42)

    model = build_encoder_model(
        input_dim=X_train.shape[1],
        latent_dim=latent_dim,
        output_dim=Y.shape[1],
    )
    # wrap into subclass so train_step/test_step are used
    model = MaskedEncoderModel(inputs=model.inputs, outputs=model.outputs, name=model.name)
    model.inactive_loss_weight = inactive_loss_weight
    print("Model building complete.")

# compile only optimizer; custom train_step/test_step handle masked loss and metrics
    model.compile(optimizer="adam")

    # ==== Train the model ====
    model.fit(
        X_train,
        Y_train,
        sample_weight=mask_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    preds = model.predict(X_test, verbose=0)
    mae = float(np.sum(np.abs(Y_test - preds) * mask_test) / np.maximum(np.sum(mask_test), 1.0))
    print(f"Test MAE: {mae:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "test_indices.json", "w") as f:
        json.dump([int(i) for i in test_idx], f)

    model.save(MODEL_PATH)

    with open(SCALER_PATH, "wb") as handle:
        pickle.dump(scaler, handle)

    metadata = {
        "input_dim": int(X_train.shape[1]),
        "output_dim": int(Y.shape[1]),
        "roadm_categories": roadm_categories,
        "latent_dim": latent_dim,
        "ripple": ripple,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "inactive_loss_weight": inactive_loss_weight,
    }
    metadata["train_source"] = str(train_path)
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
    df = load_dataset(path, calculate_ripple=use_ripple)
    X, Y_true, roadm_categories, X_mask = prepare_features(df, roadm_categories=metadata.get("roadm_categories"), use_ripple=use_ripple)
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled, verbose=0)
    preds = preds * X_mask

    keep = slice(None)
    if heldout:
        idx_path = MODEL_DIR / "test_indices.json"
        if idx_path.exists():
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
    args = parser.parse_args()

    if args.train:
        train_model(args.train, epochs=args.epochs, batch_size=args.batch_size, latent_dim=args.latent_dim, inactive_loss_weight=args.inactive_loss_weight, ripple=args.ripple)

    if args.predict:
        predict(args.predict, heldout=args.heldout)

if __name__ == '__main__':
    main()