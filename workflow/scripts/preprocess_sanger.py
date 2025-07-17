"""
Preprocess Sanger drug response dataset (GDSC).

This script:
1. Selects one measurement per drug (GDSC2 preferred over GDSC1).
2. Filters valid cell lines using external metadata.
3. Creates a cell line × drug response matrix (AUC).
4. Outputs a standardized (z-score) version of the matrix.
"""

import os
import pandas as pd
import numpy as np

# paths
sanger_inputs_path = "../data/sanger/sanger-dose-response.csv"
drugs_info_path = "../data/sanger/screened_compounds_rel_8.5.csv"
metadata_path = "../data/Model.csv"
output_folder = "data/sanger_processed"
os.makedirs(output_folder, exist_ok=True)

response_path = os.path.join(output_folder, "drug_response_sanger.csv")
standardized_path = os.path.join(output_folder, "drug_response_sanger_standardized.csv")
drug_net_path = os.path.join(output_folder, "drug_net_sanger.csv")  # placeholder, if needed later

# Load data
df_raw = pd.read_csv(sanger_inputs_path)
df_meta = pd.read_csv(metadata_path, index_col=0)

# Normalize drug names 
df_raw["DRUG_NAME"] = df_raw["DRUG_NAME"].str.lower()

#Filter by preferred dataset order (GDSC2 > GDSC1)
screen_priority = ["GDSC2", "GDSC1"]
seen = set()
filtered = []

for screen in screen_priority:
    subset = df_raw[df_raw["DATASET"] == screen]
    new = subset[~subset["DRUG_NAME"].isin(seen)]
    seen.update(new["DRUG_NAME"])
    filtered.append(new)

df_filtered = pd.concat(filtered, ignore_index=True)

# Use ARXSPAN_ID as depmap_id
df_filtered = df_filtered[df_filtered["ARXSPAN_ID"].notna()]
df_filtered["depmap_id"] = df_filtered["ARXSPAN_ID"]

# Filter valid cell lines (present in metadata)
valid_cells = set(df_meta.index)
df_filtered = df_filtered[df_filtered["depmap_id"].isin(valid_cells)]

# Clean BROAD_ID (keep first if multiple)
df_filtered["broad_id"] = df_filtered["BROAD_ID"].str.split(",", n=1).str[0]

# Check duplicate measurements
dup = df_filtered.groupby(["depmap_id", "broad_id"]).size()
dup = dup[dup > 1]
print(f"🔍 Duplicate measurements per (cell, drug): {len(dup)}")
if len(dup) > 0:
    print(dup.head())

# Create AUC matrix 
df_auc = df_filtered.groupby(["depmap_id", "broad_id"])["auc"].mean().unstack()
df_auc.to_csv(response_path)

# Standardize (z-score across cell lines per drug)
drug_means = df_auc.mean(axis=0)
drug_stds = df_auc.std(axis=0)
df_auc_std = (df_auc - drug_means) / drug_stds
df_auc_std.to_csv(standardized_path)


print("✅ Sanger preprocessing complete.")

