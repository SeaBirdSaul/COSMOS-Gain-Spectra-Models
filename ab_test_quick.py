import sys
import os

sys.argv = ["multispan_net.py", "--ripple"]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "neural_nets"))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json

import multispan_net as M

tf.get_logger().setLevel("ERROR")
np.random.seed(42)
tf.random.set_seed(42)

DATA = "dataset/multispan/multispan_topology1.json"

df = M.load_dataset(DATA)
X, Y, roadm_categories, X_mask, stages = M.prepare_features(df)

rng = np.random.RandomState(42)
keep = []
for s in roadm_categories:
    idx = np.where(stages == s)[0]
    pick = rng.choice(idx, size=min(200, len(idx)), replace=False)
    keep.extend(pick)
keep = np.array(keep)
X, Y, X_mask, stages = X[keep], Y[keep], X_mask[keep], stages[keep]
print(f"Subset: {len(X)} rows, {len(roadm_categories)} stages")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

idx_train, idx_test = train_test_split(np.arange(len(X_scaled)), test_size=0.2, random_state=42)
X_train, X_test = X_scaled[idx_train], X_scaled[idx_test]
Y_train, Y_test = Y[idx_train], Y[idx_test]
mask_train, mask_test = X_mask[idx_train], X_mask[idx_test]
stages_train, stages_test = stages[idx_train], stages[idx_test]

idx_tr, idx_val = train_test_split(np.arange(len(X_train)), test_size=0.2, random_state=42)
X_tr, X_val = X_train[idx_tr], X_train[idx_val]
Y_tr, Y_val = Y_train[idx_tr], Y_train[idx_val]
mask_tr, mask_val = mask_train[idx_tr], mask_train[idx_val]
stages_tr, stages_val = stages_train[idx_tr], stages_train[idx_val]

model = M.build_multitask_model(X_tr.shape[1], roadm_categories)
model.compile(
    optimizer="adam",
    loss={name: M.combined_loss for name in roadm_categories},
    loss_weights={name: 1.0 for name in roadm_categories},
)

train_gen = M.MultiStageGenerator(X_tr, Y_tr, stages_tr, mask_tr, roadm_categories)
val_gen = M.MultiStageGenerator(X_val, Y_val, stages_val, mask_val, roadm_categories)

history = model.fit(train_gen, validation_data=val_gen, epochs=10, verbose=1)

preds = model.predict(X_test, verbose=0)
Y_pred = np.zeros_like(Y_test)
for i, s in enumerate(stages_test):
    Y_pred[i] = preds[s][i]

Y_pred_masked = Y_pred * mask_test
Y_test_masked = Y_test * mask_test
mae = mean_absolute_error(Y_test_masked, Y_pred_masked)
true_vals = Y_test_masked[Y_test_masked != 0]
pred_vals = Y_pred_masked[Y_pred_masked != 0]
true_std = np.std(true_vals)
pred_std = np.std(pred_vals)
true_range = np.percentile(true_vals, 99) - np.percentile(true_vals, 1)
pred_range = np.percentile(pred_vals, 99) - np.percentile(pred_vals, 1)

vl = [round(v, 4) for v in history.history["val_loss"]]
print("\n==== A/B TEST RESULTS ====")
print(f"val_loss trajectory: {vl}")
print(f"Test MAE:      {mae:.4f}   (pre-fix: 0.152, predict-mean baseline ~0.16)")
print(f"pred_std:      {pred_std:.4f}   (pre-fix: 0.020)")
print(f"true_std:      {true_std:.4f}")
print(f"pred 1-99% range: {pred_range:.4f} dB   (pre-fix: ~0.09)")
print(f"true 1-99% range: {true_range:.4f} dB")

with open("ab_test_history.json", "w") as f:
    json.dump({"val_loss": history.history["val_loss"], "loss": history.history["loss"]}, f, indent=2)
