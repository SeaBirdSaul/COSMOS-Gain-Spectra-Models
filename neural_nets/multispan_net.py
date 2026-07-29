import tensorflow as tf
import keras
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import os
import json
import glob
import argparse
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from helpers import spectra_dict_to_array, active_idx_to_mask, extract_samples_from_multispan_json

# ==== Directories ====
# data_dir = './dataset/booster/15dB/fix'
MODEL_DIR = os.path.join("neural_nets", "saved_models", "multispan")
MODEL_PATH = os.path.join(MODEL_DIR, "model.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")

# ==== CLI ====
parser = argparse.ArgumentParser()
parser.add_argument("--train", default=None, help="Multispan JSON file or directory for training")
parser.add_argument("--predict", default=None, help="Multispan JSON file or directory for prediction")
parser.add_argument("--skip-training", action="store_true")
parser.add_argument("--full-output", action="store_true", help="Include input/output spectra in prediction output")
parser.add_argument("--ripple", action="store_true", help="Load gain spectra as ripple around the active gain mean")
args = parser.parse_args()

# ==== Load dataset ====
def load_dataset(path):
    if os.path.isfile(path) and path.endswith('.json'):
        samples = extract_samples_from_multispan_json(path, calculate_ripple=args.ripple)
        df = pd.DataFrame(samples)
    elif os.path.isdir(path):
        json_files = glob.glob(os.path.join(path, '**' , '*.json'), recursive=True)
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {path}")
        all_samples = []
        for file in json_files:
            all_samples.extend(extract_samples_from_multispan_json(file, calculate_ripple=args.ripple))
        df = pd.DataFrame(all_samples)
    elif not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")
    else:
        raise FileNotFoundError(f"Cannot load dataset from {path}: expected a .json file or directory")
    if df.empty:
        raise ValueError(f"No valid samples found in {path}")
    return df

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

    # Scalar features (3)
    X_pin = df[['pin_total']].fillna(-60.0).values
    X_target_gain = df[['target_gain']].fillna(0).values
    X_target_power = df[['target_power']].fillna(0).values
    X_vao = df[['voa_attenuation']].fillna(0).values

    X_output_power = df[['output_power']].fillna(0).values
    X_voa_input = df[['voa_input_power']].fillna(0).values
    X_voa_output = df[['voa_output_power']].fillna(0).values

    # One-hot encode roadm stage (12 unique prefixes)
    if roadm_categories is None:
        roadm_categories = sorted(df['roadm'].unique().tolist())
    df['roadm'] = pd.Categorical(df['roadm'], categories=roadm_categories)
    roadm_dummies = pd.get_dummies(df['roadm'], dtype=np.float32).values

    # Concatenate:
    # masked spectra (95) + mask (95) + pin (1) + target_gain (1) + target_power (1) + vao (1) + roadm one-hot (N)
    X = np.hstack([X_spectra_masked, X_mask, X_pin, X_target_gain, X_target_power, X_vao, X_output_power, X_voa_input, X_voa_output, roadm_dummies])
    
    # Target Y
    target_col = 'ripple_spectra' if args.ripple and 'ripple_spectra' in df.columns else 'gain_spectra'
    Y_raw = np.stack(list(df[target_col].values))
    Y = np.nan_to_num(Y_raw, nan=0.0)
    
    return X, Y, roadm_categories, X_mask


# ==== Custom loss functions ====
@keras.saving.register_keras_serializable()
def gradient_loss(Y_true, Y_pred, mask=None):
    grad_true = Y_true[:, 1:] - Y_true[:, :-1]
    grad_pred = Y_pred[:, 1:] - Y_pred[:, :-1]
    loss = tf.reduce_mean(tf.square(grad_true - grad_pred), axis=-1)
    if mask is not None:
        grad_mask = mask[:, 1:] * mask[:, :-1]
        loss = loss * tf.reduce_mean(grad_mask, axis=-1)
    return tf.reduce_mean(loss)

@keras.saving.register_keras_serializable()
def cosine_shape_loss(Y_true, Y_pred, mask=None):
    if mask is not None:
        Y_true = Y_true * mask
        Y_pred = Y_pred * mask
    Y_true_n = tf.math.l2_normalize(Y_true, axis=-1)
    Y_pred_n = tf.math.l2_normalize(Y_pred, axis=-1)
    return 1.0 - tf.reduce_mean(tf.reduce_sum(Y_true_n * Y_pred_n, axis=-1))

@keras.saving.register_keras_serializable()
def combined_loss(Y_true, Y_pred, sample_weight=None):
    # Derive per-channel mask from sample_weight (shappe: (batch, 95))
    # sample_weight is 1 for active channels, 0 for inactive channels
    mask = tf.cast(sample_weight > 0.5, tf.float32) if sample_weight is not None else None

    if mask is not None:
        Y_true_m = Y_true * mask
        Y_pred_m = Y_pred * mask
        active_count = tf.reduce_sum(mask, axis=-1) + 1e-8
    else:
        Y_true_m = Y_true
        Y_pred_m = Y_pred
        active_count = tf.cast(tf.shape(Y_true)[-1], tf.float32)

    channel_mse = tf.reduce_sum(tf.square(Y_true_m - Y_pred_m), axis=-1) / active_count
    mse = tf.reduce_mean(channel_mse)

    return mse + 0.3 * gradient_loss(Y_true, Y_pred, mask) + 0.2 * cosine_shape_loss(Y_true, Y_pred, mask) 

# ==== Model training ====
def train_model(train):
    print(f"Loading training data from {train}")
    df = load_dataset(train)
    X, Y, roadm_categories, X_mask = prepare_features(df)

    print(f"Features: {X.shape[1]} (spectra: 95, mask: 95, pin: 1, target_gain: 1, target_power: 1, voa: 1, roadm: {len(roadm_categories)})")
    print(f"ROADM categories: {roadm_categories}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_mask_for_loss = X[:, 95:190]

    # Split by file if multiple, else random
    unique_files = df['source_file'].unique()
    if len(unique_files) > 1:
        train_files, test_files = train_test_split(unique_files, test_size=0.2, random_state=42)
        train_mask = df['source_file'].isin(train_files)
        test_mask = df['source_file'].isin(test_files)
        X_train, X_test = X_scaled[train_mask], X_scaled[test_mask]
        Y_train, Y_test = Y[train_mask], Y[test_mask]
        X_mask_train = X_mask_for_loss[train_mask]
        X_mask_test = X_mask_for_loss[test_mask]
    else:
        print("Warning: Single file, falling back to random shuffle split.")
        indices = np.arange(len(X_scaled))
        idx_train, idx_test = train_test_split(indices, test_size=0.2, random_state=42)
        X_train, X_test = X_scaled[idx_train], X_scaled[idx_test]
        Y_train, Y_test = Y[idx_train], Y[idx_test]
        X_mask_train = X_mask_for_loss[idx_train]
        X_mask_test = X_mask_for_loss[idx_test]

    print(f"\nTrain/Test Split: {len(X_train)} train, {len(X_test)} test.")

    # ==== Build the model ====
    output_dim = Y.shape[1]
    input_dim = X_train.shape[1]
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dense(output_dim, activation='linear')
    ])

    print("Model building complete.")
    model.compile(optimizer='adam', loss=combined_loss)

    # ==== Train the model ====
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    history = model.fit(
        X_train,
        Y_train,
        sample_weight = X_mask_train,
        validation_split=0.2,
        epochs=200,
        batch_size=64,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    print("Model training complete.")

    # ==== Evaluate the model ====
    Y_pred = model.predict(X_test, verbose=0)
    Y_pred_masked = Y_pred * X_mask_test
    Y_test_masked = Y_test * X_mask_test

    mae = mean_absolute_error(Y_test_masked, Y_pred_masked)
    max_err = np.max(np.abs(Y_test_masked - Y_pred_masked), axis=0)
    shape_corr = np.array([
        np.dot(Y_pred_masked[i], Y_test_masked[i]) / (np.linalg.norm(Y_pred_masked[i]) * np.linalg.norm(Y_test_masked[i]) + 1e-8) 
        for i in range(len(Y_test))
        if np.linalg.norm(Y_pred_masked[i]) > 1e-8 and np.linalg.norm(Y_test_masked[i]) > 1e-8
    ])
    print(f"Test MAE: {mae}")
    print(f"Max channel error: min={max_err.min():.4f} max={max_err.max():.4f} mean={max_err.mean():.4f}")
    print(f"Shape correlation: min={shape_corr.min():.4f} mean={shape_corr.mean():.4f}")

    # ==== Save the model and scaler ====
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)

    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    metadata = {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "roadm_categories" : roadm_categories
    }

    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model saved to {MODEL_PATH}")

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
def predict(path, full_output=False):
    """Run inference on a new CSV file using the saved model."""
    print(f"\nLoading saved model from {MODEL_PATH}")
    model, scaler, metadata = load_saved_model()
    print(f"Loading evaluation data from {path}")
    df = load_dataset(path)

    out_dir = os.path.join(MODEL_DIR, "predictions")
    os.makedirs(out_dir, exist_ok=True)
    csv_out = os.path.join(out_dir, "predictions.csv")
    json_out = os.path.join(out_dir, "predictions.json")

    # rebuild the same feature matrix as training
    roadm_categories = metadata.get("roadm_categories")
    X, Y_true, _, X_mask = prepare_features(df, roadm_categories=roadm_categories)
    X_scaled = scaler.transform(X) 

    Y_pred = model.predict(X_scaled, verbose=0)

    pred_cols = [f"pred_gain_ch{i+1}" for i in range(Y_pred.shape[1])]
    true_cols = [f"true_gain_ch{i+1}" for i in range(Y_true.shape[1])]

    result = pd.DataFrame(Y_pred, columns=pred_cols)
    true_df = pd.DataFrame(Y_true, columns=true_cols)

    meta = df[["source_file", "pin_total", "num_active_channels", "roadm", "target_gain"]].reset_index(drop=True)
    result = pd.concat([meta, result, true_df], axis=1)

    result.to_csv(csv_out, index=True, index_label="sample_index")
    print(f"Saved {len(result)} rows to {csv_out}")

    records = []
    for i in range(len(df)):
        rec = {
            "sample_index": i,
            "source_file": df.iloc[i]["source_file"],
            "pin_total": float(df.iloc[i]["pin_total"]),
            "num_active_channels": int(df.iloc[i]["num_active_channels"]),
            "roadm": df.iloc[i]["roadm"],
            "target_gain": float(df.iloc[i]["target_gain"]),
            "mask": np.asarray(df.iloc[i]["mask"], dtype=np.int8).tolist(),
            "predicted_gain_spectra": Y_pred[i].tolist(),
            "true_gain_spectra": Y_true[i].tolist()
        }
        if full_output:
            rec["input_spectra"] = df.iloc[i]["input_spectra"].tolist()
            rec["output_spectra"] = df.iloc[i]["output_spectra"].tolist()
        records.append(rec)
    
    with open(json_out, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {len(records)} records to {json_out}")

    print("\nPrediction preview (first 3 samples):")
    print(pd.DataFrame(Y_pred[:3]).head())

    # Compare with targets if available
    if Y_true.shape[1] == Y_pred.shape[1]:
        Y_pred_masked = Y_pred * X_mask
        Y_true_masked = Y_true * X_mask
        mae = mean_absolute_error(Y_true_masked, Y_pred_masked)
        max_err = np.max(np.abs(Y_true_masked - Y_pred_masked), axis=0)
        shape_corr = np.array([
            np.dot(Y_pred_masked[i], Y_true_masked[i]) / (np.linalg.norm(Y_pred_masked[i]) * np.linalg.norm(Y_true_masked[i]) + 1e-8) 
            for i in range(len(Y_true_masked))
            if np.linalg.norm(Y_pred_masked[i]) > 1e-8 and np.linalg.norm(Y_true_masked[i]) > 1e-8
        ])
        print(f"\nPrediction MAE against available targets: {mae}")
        print(f"Max channel error: min={max_err.min():.4f} max={max_err.max():.4f} mean={max_err.mean():.4f}")
        print(f"Shape correlation: min={shape_corr.min():.4f} mean={shape_corr.mean():.4f}")
    else:
        print("\nNo target columns available for MAE comparison.")

    return Y_pred

if __name__ == '__main__':
    if args.train and not args.skip_training:
        train_model(args.train)

    if args.predict:
        predict(args.predict, full_output=args.full_output)