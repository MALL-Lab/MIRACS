import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import os
import mofax as mfx
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ================================
# 1. Load MOFA model and factor matrix Z
# ================================
model_name = "mofa_model_groups_30f"
models_dir = "../models"
model_path = os.path.join(models_dir, f"{model_name}.hdf5")

m = mfx.mofa_model(model_path)
Z_array = m.get_factors()
Z = pd.DataFrame(Z_array, index=m.get_samples())
Z.columns = [f"Factor_{i+1}" for i in range(Z.shape[1])]
Z.index = [i[1] if isinstance(i, tuple) else i for i in Z.index]
Z.index.name = "depmap_id"
df_z = Z

# ================================
# 2. Load drug response data
# ================================
prism_path = "../data/prism_processed/drug_response_prism.csv"
prism_drug_net_path = "../data/prism_processed/drug_net_prism.csv"
sanger_path = "../data/sanger/sanger-dose-response.csv"
sanger_drugs_path = "../data/sanger/screened_compounds_rel_8.5.csv"

df_prism = pd.read_csv(prism_path, index_col=0)
df_drugnet_prism = pd.read_csv(prism_drug_net_path)
df_sanger_raw = pd.read_csv(sanger_path)
df_sanger_drugs = pd.read_csv(sanger_drugs_path)

# ================================
# 3. Preprocess PRISM
# ================================
broad_to_name = df_drugnet_prism.drop_duplicates("broad_id").set_index("broad_id")["name"].str.lower()
df_prism_named = df_prism.rename(columns=broad_to_name)
df_prism_grouped = df_prism_named.groupby(axis=1, level=0).mean()
df_prism_std = (df_prism_grouped - df_prism_grouped.mean()) / df_prism_grouped.std()

# ================================
# 4. Preprocess SANGER
# ================================
df_sanger_raw['DRUG_NAME'] = df_sanger_raw['DRUG_NAME'].str.lower()
df_sanger_raw['depmap_id'] = df_sanger_raw['ARXSPAN_ID']

# Prioritize GDSC2 over GDSC1
seen = set()
rows = []
for source in ['GDSC2', 'GDSC1']:
    temp = df_sanger_raw[df_sanger_raw['DATASET'] == source]
    new = temp[~temp['DRUG_NAME'].isin(seen)]
    seen.update(new['DRUG_NAME'])
    rows.append(new)
df_sanger_filtered = pd.concat(rows)

df_sanger_mat = df_sanger_filtered.groupby(['depmap_id', 'DRUG_NAME'])['auc'].max().unstack()
df_sanger_std = (df_sanger_mat - df_sanger_mat.mean()) / df_sanger_mat.std()

# ================================
# 5. Intersect samples
# ================================
common_cells = df_z.index.intersection(df_prism_std.index).intersection(df_sanger_std.index)
df_z_common = df_z.loc[common_cells]
df_prism_std = df_prism_std.loc[common_cells]
df_sanger_std = df_sanger_std.loc[common_cells]

# ================================
# 6. Compute correlation matrix (Z × drug)
# ================================
def compute_cor_matrix(df_auc, df_z):
    result = pd.DataFrame(index=df_z.columns, columns=df_auc.columns)
    for factor in df_z.columns:
        for drug in df_auc.columns:
            x, y = df_auc[drug], df_z[factor]
            x, y = x.align(y, join='inner')
            if len(x) < 2:
                result.loc[factor, drug] = np.nan
            else:
                rho, _ = spearmanr(x, y, nan_policy='omit')
                result.loc[factor, drug] = rho
    return result.astype(float)

cormatrix_prism = compute_cor_matrix(df_prism_std, df_z_common)
cormatrix_sanger = compute_cor_matrix(df_sanger_std, df_z_common)

# ================================
# 7. Compare PRISM vs SANGER
# ================================
shared_drugs = cormatrix_prism.columns.intersection(cormatrix_sanger.columns)
flat1 = cormatrix_prism[shared_drugs].values.flatten()
flat2 = cormatrix_sanger[shared_drugs].values.flatten()

final_corr, pval = spearmanr(flat1, flat2)

# ================================
# 8. Save results
# ================================
results_dir = "../results/mofa_drug_correlation_comparison"
os.makedirs(results_dir, exist_ok=True)

# Save correlation statistics
stats_path = os.path.join(results_dir, "correlation_stats.txt")
with open(stats_path, "w") as f:
    f.write("Spearman correlation between PRISM and SANGER drug correlation matrices:\n")
    f.write(f"r = {final_corr:.4f}\n")
    f.write(f"p-value = {pval:.4e}\n")

print(f" Spearman correlation saved to {stats_path}")

# ================================
# 9. Plot scatter with regression
# ================================
plt.figure(figsize=(3.5, 3))
plt.scatter(flat1, flat2, alpha=0.5, edgecolors='k')
plt.xlabel("Correlation MOFA–AUC (PRISM)", fontsize=10)
plt.ylabel("Correlation MOFA–AUC (SANGER)", fontsize=10)
plt.title("PRISM vs SANGER correlation", fontsize=10)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.7)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.7)

# Linear regression
X = flat1.reshape(-1, 1)
y = flat2
sort_idx = np.argsort(flat1)
X_sorted = X[sort_idx]
y_sorted = y[sort_idx]
model = LinearRegression().fit(X_sorted, y_sorted)
y_pred = model.predict(X_sorted)
residuals = y_sorted - y_pred
std_err = np.std(residuals)
ci = 1.96 * std_err

# Plot regression line and confidence band
plt.plotcompare_prism_sanger (X_sorted, y_pred, color='red', linewidth=1.5, label="Linear fit")
plt.fill_between(X_sorted.ravel(), y_pred - ci, y_pred + ci, color='red', alpha=0.2, label='95% CI')
plt.legend(fontsize=8)
plt.tight_layout()

# Save figure
plot_path = os.path.join(results_dir, "scatter_prism_vs_sanger.png")
plt.savefig(plot_path, dpi=300)
plt.show()
print(f"Scatter plot saved to {plot_path}")
