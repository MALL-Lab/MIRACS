import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from drug_enrichment import run_drug_enrichment  # Import the reusable enrichment pipeline

# Configuration
model_name = "mofa_model_groups_30f"
results_dir = os.path.join("..", "results", f"enrichment_{model_name}")
os.makedirs(results_dir, exist_ok=True)

# Load input data 
Z_filtered_factors = pd.read_csv("../data/Z_filtered_factors.csv", index_col=0)
Z_filtered_lineage = pd.read_csv("../data/Z_filtered_lineage.csv", index_col=0, squeeze=True)
drug_response_filtered = pd.read_csv("../data/drug_response_filtered.csv", index_col=0)
drug_net_filtered = pd.read_csv("../data/filtered_drug_net.csv")

# Run enrichment by lineage
results_by_lineage = {}
pvals_by_lineage = {}
cormatrix_by_lineage = {}

for lineage in Z_filtered_lineage["lineage"].unique():
    cells = Z_filtered_lineage[Z_filtered_lineage["lineage"] == lineage].index
    shared = cells.intersection(drug_response_filtered.index).intersection(Z_filtered_factors.index)

    if len(shared) < 5:
        continue

    Z_sub = Z_filtered_factors.loc[shared]
    drug_sub = drug_response_filtered.loc[shared]

    try:
        es, pvals, corr = run_drug_enrichment(
            df_drugs=drug_sub,
            df_counts=Z_sub,
            drug_net=drug_net_filtered,
            shared_elements='broad_id',
            group='moa',
            min_elements=5,
            number_of_threads=12,
            return_cormatrix=True,
            method='spearman',
            iteration=10000
        )
        results_by_lineage[lineage] = es
        pvals_by_lineage[lineage] = pvals
        cormatrix_by_lineage[lineage] = corr
        print(f"{lineage}: {es.shape[0]} drugs x {es.shape[1]} factors")
    except Exception as e:
        print(f"{lineage} skipped: {e}")

# === Save results ===
for lineage in results_by_lineage:
    lineage_dir = os.path.join(results_dir, lineage)
    os.makedirs(lineage_dir, exist_ok=True)

    results_by_lineage[lineage].to_csv(os.path.join(lineage_dir, "enrichment.csv"))
    pvals_by_lineage[lineage].to_csv(os.path.join(lineage_dir, "pvalues.csv"))
    cormatrix_by_lineage[lineage].to_csv(os.path.join(lineage_dir, "correlations.csv"))

# === Plot correlation distributions ===
correlation_long = []
for lineage, corr in cormatrix_by_lineage.items():
    flat_vals = corr.values.flatten()
    flat_vals = flat_vals[~pd.isna(flat_vals)]
    for val in flat_vals:
        correlation_long.append({"Lineage": lineage, "Correlation": val})

cor_df = pd.DataFrame(correlation_long)

plt.figure(figsize=(8, 4.5))
sns.kdeplot(
    data=cor_df,
    x="Correlation",
    hue="Lineage",
    common_norm=False,
    fill=True,
    alpha=0.4,
    bw_adjust=0.7
)
plt.title("Distribution of correlation values (drug vs. MOFA factors)")
plt.xlabel("Correlation (Spearman)")
plt.ylabel("Density")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, f"correlation_distribution_{model_name}.png"), dpi=300)
plt.show()

# === Analyze enrichment ===
df_list = []
df_pval_list = []
pval_threshold = 0.05
enrichment_threshold = 1.0

for lineage in results_by_lineage:
    enrich = results_by_lineage[lineage]
    pvals = pvals_by_lineage[lineage]

    enrich_t = enrich.T
    enrich_t.columns = [f"{col}_{lineage}" for col in enrich_t.columns]
    pvals_t = pvals.T
    pvals_t.columns = enrich_t.columns

    df_list.append(enrich_t)
    df_pval_list.append(pvals_t)

df_enrichment_all = pd.concat(df_list, axis=1)
df_pvals_all = pd.concat(df_pval_list, axis=1)

# Apply significance thresholds
mask_significant = df_pvals_all < pval_threshold
mask_enrichment = df_enrichment_all.abs() >= enrichment_threshold
df_enrichment_sig = df_enrichment_all.where(mask_significant & mask_enrichment)

# === Save filtered enrichment heatmap ===
plt.figure(figsize=(60, 20))
sns.heatmap(df_enrichment_sig, cmap="coolwarm", center=0)
plt.title(f"MOA Enrichment (p < {pval_threshold}, |NES| ≥ {enrichment_threshold})")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, f"heatmap_enrichment_{model_name}.png"), dpi=300)
plt.show()

# === Sensitivity/Resistance classification ===
all_vals = df_enrichment_all.values.flatten()
all_vals = all_vals[~np.isnan(all_vals)]
q10 = np.percentile(all_vals, 10)
q90 = np.percentile(all_vals, 90)

sensitivity_mask = df_enrichment_all < q10
resistance_mask = df_enrichment_all > q90

factor_names = df_enrichment_all.columns.to_series().str.extract(r'(Factor\d+)_')[0]
factors_unique = factor_names.unique()

sensitivity_counts = pd.DataFrame(0, index=df_enrichment_all.index, columns=factors_unique)
resistance_counts = pd.DataFrame(0, index=df_enrichment_all.index, columns=factors_unique)

for col, factor in zip(df_enrichment_all.columns, factor_names):
    sensitivity_counts[factor] += sensitivity_mask[col].astype(int)
    resistance_counts[factor] += resistance_mask[col].astype(int)

# Save result tables
sensitivity_counts.to_csv(os.path.join(results_dir, f"sensitivity_counts_{model_name}.csv"))
resistance_counts.to_csv(os.path.join(results_dir, f"resistance_counts_{model_name}.csv"))

# Print top associations
print("Top sensitive associations:")
print(sensitivity_counts.stack().sort_values(ascending=False).head(10))

print("\nTop resistant associations:")
print(resistance_counts.stack().sort_values(ascending=False).head(10))
