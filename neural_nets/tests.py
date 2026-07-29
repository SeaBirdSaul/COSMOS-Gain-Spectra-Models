from helpers import extract_samples_from_json
from pathlib import Path

here = Path(__file__).parent
dataset_json = here / ".." / "dataset" / "booster" / "15dB" / "fix" / "edfa_meas_rdm1-co1.bed_booster_2022.06.27.17.18.25.json"

rows = extract_samples_from_json(dataset_json)

print(f"Samples extracted: {len(rows)}")
print(f"Keys: {sorted(rows[0].keys())}")
print(f"open_channel_type values: {[r['open_channel_type'] for r in rows]}")

print("======================================================================================")

import pandas as pd
from neural_net import prepare_features

df = pd.DataFrame(rows)
X, Y, roadm_cats, open_ch_cats = prepare_features(df)

print(f"X shape: {X.shape}")
print(f"open_channel_categories: {open_ch_cats}")

print("======================================================================================")

import json
with open(here / "saved_models" / "metadata.json" ) as f:
    m = json.load(f)
print(m["open_channel_categories"])

print("======================================================================================")
