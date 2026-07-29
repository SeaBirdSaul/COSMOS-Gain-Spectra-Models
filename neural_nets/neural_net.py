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
import re
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from helpers import spectra_dict_to_array, active_idx_to_mask, get_path_metadata, extract_samples_from_json, extract_samples_from_multispan_json


# data_dir = './dataset/booster/15dB/fix'
MODEL_DIR = os.path.join("neural_nets", "saved_models", "singlespan")
MODEL_PATH = os.path.join(MODEL_DIR, "model.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")

# ==== CLI ====
parser = argparse.ArgumentParser()
parser.add_argument("--train", default=None, help="CSV file or JSON directory for training")
parser.add_argument("--predict", default=None, help="CSV file or JSON directory for prediction")
parser.add_argument("--skip-training", action="store_true")
parser.add_argument("--full-output", action="store_true", help="Include input/output spectra in prediction output")
parser.add_argument("--ripple", action="store_true", help="Load gain spectra as ripple around the active gain mean")
args = parser.parse_args()

# ==== Load CSV data (CSV fallback) ====
# path = os.path.join('misc', 'ML_features', 'booster', 'rdm-1-co1_train.csv')
def load_dataset(path):
    """Load dataset from a CSV file, single JSON file or directory of JSONs."""
    if os.path.isfile(path) and path.endswith('.csv'):
        df = _load_csv(path)
    elif os.path.isfile(path) and path.endswith('.json'):
        df = pd.DataFrame(_load_json(path))
    elif os.path.isdir(path):
        json_files = glob.glob(os.path.join(path, '**', '*.json'), recursive=True)
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {path}")
        all_samples = []
        for jf in json_files:
            all_samples.extend(_load_json(jf))
        df = pd.DataFrame(all_samples)
    elif not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")
    else:
        raise FileNotFoundError(f"Cannot load dataset from {path}: not a .csv file, .json file, or directory")
    return df

def _load_csv(csv_path):
        df_csv = pd.read_csv(csv_path)

        # detect per-channel columns by regex and sort by channel index
        def sorted_cols(prefix):
            cols = [c for c in df_csv.columns if c.startswith(prefix)]
            cols_sorted = sorted(cols, key=lambda s: int(m.group(1)) if (m := re.search(r'(\d+)$', s)) else -1)
            return cols_sorted
        
        input_cols = sorted_cols('EDFA_input_spectra_')
        mask_cols = sorted_cols('DUT_WSS_activated_channel_index_')
        gain_cols = sorted_cols('calculated_gain_spectra_')

        # assemble arrays
        X_spectra = df_csv[input_cols].values.astype(np.float32)
        X_mask = df_csv[mask_cols].values.astype(np.int8)
        Y = df_csv[gain_cols].values.astype(np.float32)

        # scalar features
        if 'EDFA_input_power_total' in df_csv.columns:
            pin_total = df_csv['EDFA_input_power_total'].values
        else:
            pin_total = np.full((len(df_csv),), np.nan)
        
        # target_gain (e.g. rdm1-co1_train.csv has this column)
        if 'target_gain' in df_csv.columns:
            target_gain = df_csv['target_gain'].values
        else:
            target_gain = np.full((len(df_csv),), np.nan)
        
        # Exact roadm name from filename (e.g. rdm1-co1_train.csv -> rdm1-co1)
        basename = os.path.basename(csv_path)
        roadm_match = re.match(r'([^.]+)', basename)
        roadm = roadm_match.group(1).replace('_train', '').replace('_test', '').replace('_augm', '') if roadm_match else 'unknown'

        # Create DataFrame rows so downstream logic can remain unchanged
        rows = []
        for i in range(len(df_csv)):
            rows.append({
                'input_spectra': X_spectra[i],
                'mask': X_mask[i],
                'gain_spectra': Y[i],
                'pin_total': pin_total[i],
                'target_gain': target_gain[i],
                'roadm': roadm,
                'source_file': os.path.basename(csv_path),
                'num_active_channels': int(X_mask[i].sum())
            })
        return pd.DataFrame(rows)

def _load_json(path):
    with open(path, "r") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return extract_samples_from_multispan_json(path, calculate_ripple=args.ripple)
    return extract_samples_from_json(path, calculate_ripple=args.ripple)

# ==== Prepare the data ====
def prepare_features(df, roadm_categories=None, open_channel_categories=None):
    """
    Prepare features from a loaded DataFrame.

    Args:
        df: Datafram with spectra, mask, pin_total, target_gain, roadm columns.
        roadm_categories: Optional list of roadm names for consistent one-hot encoding
    
    If None, derivced from df (training mode)
    """
    df = df.copy()

    # Ensure arrays are numpy arrays and consistent dtype
    df['input_spectra'] = df['input_spectra'].apply(lambda x: np.asarray(x, dtype=np.float32))
    df['mask'] = df['mask'].apply(lambda x: np.asarray(x, dtype=np.int8))
    df['gain_spectra'] = df['gain_spectra'].apply(lambda x: np.asarray(x, dtype=np.float32))

    # Input features, filled -60 for inactive channels
    X_spectra = np.stack(list(df['input_spectra'].values))
    X_spectra[np.isnan(X_spectra)] = -60.0

    # Mask (0/1) for active channels
    X_mask = np.stack(list(df['mask'].values))

    # Zero out inactive channels in spectra so the model sees clean input
    X_spectra_masked = X_spectra * X_mask

    # Scalar features
    X_pin = df[['pin_total']].fillna(-60.0).values
    X_gain = df[['target_gain']].fillna(0).values

    # One-hot encode roadm identity
    if roadm_categories is None:
        roadm_categories = sorted(df['roadm'].unique().tolist())
    df['roadm'] = pd.Categorical(df['roadm'], categories=roadm_categories)
    roadm_dummies = pd.get_dummies(df['roadm'], dtype=np.float32).values

    # One-hot encode open_channel_type
    if 'open_channel_type' in df.columns:
        if open_channel_categories is None:
            open_channel_categories = sorted(df['open_channel_type'].unique().tolist())
        df['open_channel_type'] = pd.Categorical(df['open_channel_type'], categories=open_channel_categories)
        open_ch_dummies = pd.get_dummies(df['open_channel_type'], dtype=np.float32).values
    else:
        open_ch_dummies = np.zeros((len(df), 0), dtype=np.float32)
        open_channel_categories = []

    # Concatenate all features:
    # masked spectra (95) + mask (95) + pin_total (1) + target_gain (1) + roadm one-hot (N) + open_channel_type one-hot (M)
    X = np.hstack([X_spectra_masked, X_mask, X_pin, X_gain, roadm_dummies, open_ch_dummies])

    # Target Y
    target_col = 'ripple_spectra' if args.ripple and 'ripple_spectra' in df.columns else 'gain_spectra'
    Y_raw = np.stack(list(df[target_col].values))
    Y = np.nan_to_num(Y_raw, nan=0.0)
    
    return X, Y, roadm_categories, open_channel_categories

# ==== Custom loss functions ====
@keras.saving.register_keras_serializable()
def gradient_loss(Y_true, Y_pred):
    grad_true = Y_true[:, 1:] - Y_true[:, :-1]
    grad_pred = Y_pred[:, 1:] - Y_pred[:, :-1]
    return tf.reduce_mean(tf.square(grad_true - grad_pred))

@keras.saving.register_keras_serializable()
def cosine_shape_loss(Y_true, Y_pred):
    Y_true_n = tf.math.l2_normalize(Y_true, axis=-1)
    Y_pred_n = tf.math.l2_normalize(Y_pred, axis=-1)
    return 1.0 - tf.reduce_mean(tf.reduce_sum(Y_true_n * Y_pred_n, axis=-1))

@keras.saving.register_keras_serializable()
def combined_loss(Y_true, Y_pred):
    mse = tf.reduce_mean(tf.square(Y_true - Y_pred))
    return mse + 0.3 * gradient_loss(Y_true, Y_pred) + 0.2 * cosine_shape_loss(Y_true, Y_pred) 

# ==== Training the model ====
def train_model(train):
    """Train the model and save it along with the scaler"""
    print(f"Loading training data from {train}")
    df = load_dataset(train)
    X, Y, roadm_categories, open_channel_categories = prepare_features(df)

    print(f"Features: {X.shape[1]} (spectra: 95, mask: 95, pin_total: 1, target_gain: 1, roadm: {len(roadm_categories)}, open_channel_type: {len(open_channel_categories)})")
    print(f"ROADM categories: {roadm_categories}")
    print(f"Open channel categories: {open_channel_categories}")

    # Standardize X
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    unique_files = df['source_file'].unique()
    if len(unique_files) > 5:
        train_files, test_files = train_test_split(unique_files, test_size=0.2, random_state=42)
        train_mask = df['source_file'].isin(train_files)
        test_mask = df['source_file'].isin(test_files)

        X_train, X_test = X_scaled[train_mask], X_scaled[test_mask]
        Y_train, Y_test = Y[train_mask], Y[test_mask]
    else:
        print("Warning: Small dataset, falling back to random shuffle split.")
        indices = np.arange(len(X_scaled))
        idx_train, idx_test = train_test_split(indices, test_size=0.2, random_state=42)

        X_train, X_test = X_scaled[idx_train], X_scaled[idx_test]
        Y_train, Y_test = Y[idx_train], Y[idx_test]
    
    print(f"\nTrain/Test Split: {len(X_train)} train samples, {len(X_test)} test samples.")

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
        validation_split=0.2,
        epochs=200,
        batch_size=64,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    print("Model training complete.")

    # ==== Evaluate the model ====
    Y_pred = model.predict(X_test, verbose=0)
    mae = mean_absolute_error(Y_test, Y_pred)
    max_err = np.max(np.abs(Y_test - Y_pred), axis=0)
    shape_corr = np.array([
        np.dot(Y_pred[i], Y_test[i]) / (np.linalg.norm(Y_pred[i]) * np.linalg.norm(Y_test[i]) + 1e-8) 
        for i in range(len(Y_test))
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
        "roadm_categories" : roadm_categories,
        "open_channel_categories": open_channel_categories
    }

    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model saved to {MODEL_PATH}")

# ==== Load the model and scaler ====
def load_saved_model():
    """Load the saved model and scaler"""
    model = models.load_model(MODEL_PATH)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    return model, scaler, metadata

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
    open_channel_categories = metadata.get("open_channel_categories")
    X, Y_true, _, _ = prepare_features(df, roadm_categories=roadm_categories, open_channel_categories=open_channel_categories)
    X_scaled = scaler.transform(X) 

    Y_pred = model.predict(X_scaled, verbose=0)

    pred_cols = [f"pred_gain_ch{i+1}" for i in range(Y_pred.shape[1])]
    true_cols = [f"true_gain_ch{i+1}" for i in range(Y_true.shape[1])]

    result = pd.DataFrame(Y_pred, columns=pred_cols)
    true_df = pd.DataFrame(Y_true, columns=true_cols)

    meta = df[["source_file", "pin_total", "num_active_channels", "open_channel_type", "gain_folder", "target_gain"]].reset_index(drop=True)
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
            "open_channel_type": df.iloc[i]["open_channel_type"],
            "gain_folder": df.iloc[i]["gain_folder"],
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
        mae = mean_absolute_error(Y_true, Y_pred)
        max_err = np.max(np.abs(Y_true - Y_pred), axis=0)
        shape_corr = np.array([
            np.dot(Y_pred[i], Y_true[i]) / (np.linalg.norm(Y_pred[i]) * np.linalg.norm(Y_true[i]) + 1e-8) 
            for i in range(len(Y_true))
        ])
        print(f"\nPrediction MAE against available targets: {mae}")
        print(f"Max channel error: min={max_err.min():.4f} max={max_err.max():.4f} mean={max_err.mean():.4f}")
        print(f"Shape correlation: min={shape_corr.min():.4f} mean={shape_corr.mean():.4f}")
    else:
        print("\nNo target columns available for MAE comparison.")

    return Y_pred


# ==== Main workflow ====
if __name__ == '__main__':
    if args.train and not args.skip_training:
        train_model(args.train)
    
    if args.predict:
        predict(args.predict, full_output=args.full_output)