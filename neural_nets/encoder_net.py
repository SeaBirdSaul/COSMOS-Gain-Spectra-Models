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
def load_dataset(path):
    if os.path.isfile(path) and path.endswith('.json'):
        samples = extract_samples_from_multispan_json(path, calculate_ripple=False)
        return pd.DataFrame(samples)
    if os.path.isdir(path):
        json_files = [str(p) for p in Path(path).rglob("*.json")]
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {path}")
        all_samples = []
        for file in json_files:
            all_samples.extend(extract_samples_from_multispan_json(file, calculate_ripple=False))
        return pd.DataFrame(all_samples)
    raise FileNotFoundError(f"Cannot load dataset from {path}")


# ==== Prepare data ====
def prepare_features(df, roadm_categories=None):
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
    Y_raw = np.stack(df["gain_spectra"].values)
    Y = np.nan_to_num(Y_raw, nan=0.0).astype(np.float32)

    return X, Y, roadm_categories

# ==== Model training ====
def train_model(train_path, epochs=80, batch_size=64, latent_dim=64):
    print(f"Loading training data from {train_path}")
    df = load_dataset(train_path)
    if df.empty:
        raise ValueError("No data found in training set")
    X, Y, roadm_categories = prepare_features(df)

    print(f"Features: {X.shape[1]} (spectra: 95, mask: 95, pin: 1, target_gain: 1, target_power: 1, voa: 1, roadm: {len(roadm_categories)})")
    print(f"ROADM categories: {roadm_categories}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled,
        Y,
        test_size=0.2,
        random_state=42,
    )

    model = build_encoder_model(
        input_dim=X_train.shape[1],
        latent_dim=latent_dim,
        output_dim=Y.shape[1],
    )
    print("Model building complete.")

    model.compile(optimizer="adam", loss="mae")

    # ==== Train the model ====
    model.fit(
        X_train,
        Y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    preds = model.predict(X_test, verbose=0)
    mae = mean_absolute_error(Y_test, preds)
    print(f"Test MAE: {mae:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)

    with open(SCALER_PATH, "wb") as handle:
        pickle.dump(scaler, handle)

    metadata = {
        "input_dim": int(X_train.shape[1]),
        "output_dim": int(Y.shape[1]),
        "roadm_categories": roadm_categories,
        "latent_dim": latent_dim,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
    }
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
def predict(path):
    """Run inference on a new CSV file using the saved model."""
    print(f"\nLoading saved model from {MODEL_PATH}")
    model, scaler, metadata = load_saved_model()
    print(f"Loading evaluation data from {path}")
    df = load_dataset(path)

    # rebuild the same feature matrix as training
    X, Y_true, roadm_categories = prepare_features(df, roadm_categories=metadata.get("roadm_categories"))
    X_scaled = scaler.transform(X) 
    preds = model.predict(X_scaled, verbose=0)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    pred_cols = [f"pred_gain_ch{i+1}" for i in range(preds.shape[1])]
    true_cols = [f"true_gain_ch{i+1}" for i in range(Y_true.shape[1])]

    result = pd.DataFrame(preds, columns=pred_cols)
    true_df = pd.DataFrame(Y_true, columns=true_cols)
    meta = df[["source_file", "pin_total", "num_active_channels", "roadm", "target_gain"]].reset_index(drop=True)
    output_df = pd.concat([meta, result, true_df], axis=1)
    output_df.to_csv(PREDICTIONS_DIR / "predictions.csv", index=False)
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
            "predicted_gain_spectra": preds[idx].tolist(),
            "true_gain_spectra": Y_true[idx].tolist()
        })
        
    with open(PREDICTIONS_DIR / "predictions.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {len(records)} records to {PREDICTIONS_DIR / 'predictions.json'}")

def main():
    # ==== CLI ====
    parser = argparse.ArgumentParser(description="Train or predict with an encoder-only multispan model")
    parser.add_argument("--train", default=None, help="Multispan JSON file or directory for training")
    parser.add_argument("--predict", default=None, help="Multispan JSON file or directory for prediction")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--latent-dim", type=int, default=80)
    args = parser.parse_args()

    if args.train:
        train_model(args.train, epochs=args.epochs, batch_size=args.batch_size, latent_dim=args.latent_dim)

    if args.predict:
        predict(args.predict)

if __name__ == '__main__':
    main()