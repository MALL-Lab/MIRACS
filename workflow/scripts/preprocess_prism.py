"""
Preprocess PRISM drug response dataset.

This script:
1. Selects one measurement per drug based on preferred screen order.
2. Creates a cell line × drug response matrix (AUC).
3. Constructs a drug annotation table (drug_net) with MOA and names.
4. Outputs a standardized version of the response matrix (Z-score).
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# paths
preprocess_prisminput_path = r"C:\Users\Sara GC\Máster Bioinformática\TFM\data\prism\secondary-screen-dose-response-curve-parameters19q.csv"
output_folder = "data/prism_processed"
os.makedirs(output_folder, exist_ok=True)

response_path = os.path.join(output_folder, "drug_response_prism.csv")
drug_net_path = os.path.join(output_folder, "drug_net_prism.csv")
standardized_path = os.path.join(output_folder, "drug_response_prism_standardized.csv")

# load input
df = pd.read_csv(input_path)

# Priority to recent data
screen_priority = ["MTS010", "MTS006", "MTS005", "HTS002"]
df["screen_id"] = pd.Categorical(df["screen_id"], categories=screen_priority, ordered=True)
df_sorted = df.sort_values("screen_id")

# Select one row per drug (broad_id) according to priority
seen = set()
filtered_rows = []

for screen in screen_priority:
    subset = df_sorted[df_sorted["screen_id"] == screen]
    new_entries = subset[~subset["broad_id"].isin(seen)]
    seen.update(new_entries["broad_id"])
    filtered_rows.append(new_entries)

df_filtered = pd.concat(filtered_rows, ignore_index=True)

# Create drug response matrix (DepMap ID × Broad ID) 
df_response = df_filtered.groupby(["depmap_id", "broad_id"])["auc"].mean().unstack()

#Create drug_net table with MOA and names 
drug_net = df_filtered[["broad_id", "name", "moa"]].drop_duplicates()
drug_net["moa"] = drug_net["moa"].astype(str).str.split(",")
drug_net = drug_net.explode("moa")
drug_net["moa"] = drug_net["moa"].str.strip()

# Save cleaned outputs 
df_response.to_csv(response_path)
drug_net.to_csv(drug_net_path)

# Standardize response matrix (Z-score across cell lines per drug)
drug_means = df_response.mean(axis=0)
drug_stds = df_response.std(axis=0)
df_response_std = (df_response - drug_means) / drug_stds
df_response_std.to_csv(standardized_path)


print(f"Standardized matrix saved to: {standardized_path}")
print(f"Unique MOAs: {drug_net['moa'].nunique()}")
