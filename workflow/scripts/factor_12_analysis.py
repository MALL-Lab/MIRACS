"""
This script analyzes drug response correlations for a specific MOFA factor across multiple MOAs (mechanism of action)
and cancer lineages. It performs the following:
- Computes drug-factor correlations per lineage (barplots)
- Aggregates these correlations into forest plots
- Compares factor expression between sensitive and resistant cell lines using boxplots

The results are saved in a structured folder: barplots (per MOA), forestplots (one per MOA), and boxplots (per MOA).
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu

# Configuration
FACTOR_OF_INTEREST = "Factor12"

MOAS_OF_INTEREST = [
    "topoisomerase inhibitor",
    "Bcr-Abl kinase inhibitor",
    "Aurora kinase inhibitor",
    "DNA inhibitor",
    "HSP inhibitor"
]

LINEAGES_BY_MOA = {
    "topoisomerase inhibitor": ["Breast", "Esophagus_Stomach", "Kidney", "Skin", "Uterus"],
    "Bcr-Abl kinase inhibitor": ["Bowel", "Esophagus_Stomach", "Kidney", "Lung"],
    "Aurora kinase inhibitor": ["Bowel", "Lung", "Pancreas", "Skin"],
    "DNA inhibitor": ["Bowel", "Lung", "Uterus"],
    "HSP inhibitor": ["Kidney", "Pancreas", "Skin"]
}

# Create result folders
RESULTS_BASE = f"results_{FACTOR_OF_INTEREST.lower()}"
os.makedirs(os.path.join(RESULTS_BASE, "forestplots"), exist_ok=True)

# Load preprocessed data
Z_filtered_factors = pd.read_csv("../data/Z_filtered_factors.csv", index_col=0)
Z_filtered_lineage = pd.read_csv("../data/Z_filtered_lineage.csv", index_col=0, squeeze=True)
drug_response_filtered = pd.read_csv("../data/drug_response_filtered.csv", index_col=0)
drug_net_filtered = pd.read_csv("data/prism_processed/drug_net_prism.csv")

# Clean MOA labels and map drug names
drug_net_filtered['moa'] = drug_net_filtered['moa'].astype(str).str.strip()
drug_name_map = dict(zip(drug_net_filtered['broad_id'], drug_net_filtered['name']))

# Plot: Forest plot of drug correlations across lineages
def forest_plot_per_drug(moa_name, factor_name, moa_correlations, drug_name_map, save_path):
    all_drugs = sorted({d for lineage_corr in moa_correlations.values() for d in lineage_corr.index})
    all_lineages = list(moa_correlations.keys())
    data_matrix = pd.DataFrame(index=all_drugs, columns=all_lineages)

    for lineage, corr in moa_correlations.items():
        data_matrix.loc[corr.index, lineage] = corr.values

    means = data_matrix.astype(float).mean(axis=1)
    stds = data_matrix.astype(float).std(axis=1)
    labels = pd.Series(data_matrix.index.map(lambda x: drug_name_map.get(x, x)), index=data_matrix.index)

    sorted_idx = means.sort_values().index
    y_pos = np.arange(len(sorted_idx))

    plt.figure(figsize=(4, len(sorted_idx) * 0.3 + 1))
    plt.errorbar(
        x=means.loc[sorted_idx], y=y_pos,
        xerr=stds.loc[sorted_idx],
        fmt='o', color='black', ecolor='red', capsize=4,
    )
    plt.yticks(y_pos, labels.loc[sorted_idx], fontsize=7)
    plt.axvline(x=0, color='blue', linestyle='--')
    plt.xlabel(f"Correlation with {factor_name}")
    plt.title(f"Forest plot – {moa_name}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# Plot: Barplot of absolute drug correlations for a given lineage and MOA
def plot_moa_barplot(moa_name, lineage, correlations, drug_name_map, save_path):
    abs_corr = correlations.abs().sort_index()
    drug_names = abs_corr.index.map(lambda x: drug_name_map.get(x, x))
    plt.figure(figsize=(6, 3))
    sns.barplot(x=drug_names, y=abs_corr.values, palette="tab20")
    plt.title(f"{moa_name} – {lineage}")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Absolute correlation")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# Plot: Boxplot comparing factor expression between sensitive and resistant groups
def plot_drug_boxplot(drug, drug_name, lineage, factor_values, bin_labels, save_path, factor_name, test_result):
    plot_df = pd.DataFrame({
        factor_name: factor_values,
        "ResponseGroup": bin_labels
    })

    plt.figure(figsize=(3.5, 3.5))
    sns.boxplot(data=plot_df, x="ResponseGroup", y=factor_name, order=["Resistant", "Sensitive"], palette="Set2")
    sns.stripplot(data=plot_df, x="ResponseGroup", y=factor_name, order=["Resistant", "Sensitive"],
                  color='black', alpha=0.3, jitter=0.2)

    plt.title(f"{drug_name} - {lineage}\n{test_result}")
    plt.xlabel("Group")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# Main analysis loop
for moa in MOAS_OF_INTEREST:
    lineage_list = LINEAGES_BY_MOA[moa]
    moa_drugs = drug_net_filtered[drug_net_filtered["moa"] == moa]["broad_id"].unique()
    forest_data = {}

    # Barplots per lineage
    for lineage in lineage_list:
        cells = Z_filtered_lineage[Z_filtered_lineage == lineage].index
        shared = cells.intersection(drug_response_filtered.index).intersection(Z_filtered_factors.index)
        if len(shared) < 5:
            continue

        Z_sub = Z_filtered_factors.loc[shared]
        drug_sub = drug_response_filtered.loc[shared]
        lineage_corr = {}

        lineage_barplot_dir = os.path.join(RESULTS_BASE, "barplots", moa)
        os.makedirs(lineage_barplot_dir, exist_ok=True)

        for drug in moa_drugs:
            if drug not in drug_sub.columns:
                continue
            corr_val = np.corrcoef(Z_sub[FACTOR_OF_INTEREST], drug_sub[drug])[0, 1]
            lineage_corr[drug] = corr_val

        if lineage_corr:
            corr_series = pd.Series(lineage_corr).dropna()
            forest_data[lineage] = corr_series

            barplot_path = os.path.join(lineage_barplot_dir, f"{lineage}_barplot.png")
            plot_moa_barplot(moa, lineage, corr_series, drug_name_map, barplot_path)

    # Forest plot for MOA
    forest_path = os.path.join(RESULTS_BASE, "forestplots", f"{moa}_forestplot.png")
    forest_plot_per_drug(moa, FACTOR_OF_INTEREST, forest_data, drug_name_map, forest_path)

    # Boxplots for each drug in the MOA
    boxplot_dir = os.path.join(RESULTS_BASE, "boxplots", moa)
    os.makedirs(boxplot_dir, exist_ok=True)

    for drug in moa_drugs:
        if drug not in drug_response_filtered.columns:
            continue

        auc_vals = drug_response_filtered[drug].dropna()
        if len(auc_vals) < 5:
            continue

        p33, p67 = auc_vals.quantile([0.33, 0.67])
        sensitive = auc_vals[auc_vals <= p33].index
        resistant = auc_vals[auc_vals >= p67].index

        binarized = pd.Series("Other", index=auc_vals.index)
        binarized.loc[sensitive] = "Sensitive"
        binarized.loc[resistant] = "Resistant"

        for lineage in lineage_list:
            cells = Z_filtered_lineage[Z_filtered_lineage == lineage].index
            shared = binarized.index.intersection(Z_filtered_factors.index).intersection(cells)

            bins = binarized.loc[shared]
            if (bins == "Sensitive").sum() < 3 or (bins == "Resistant").sum() < 3:
                continue

            values = Z_filtered_factors.loc[shared, FACTOR_OF_INTEREST]

            sens = values[bins == "Sensitive"]
            res = values[bins == "Resistant"]

            if (shapiro(sens).pvalue > 0.05 and shapiro(res).pvalue > 0.05 and levene(sens, res).pvalue > 0.05):
                stat, pval = ttest_ind(sens, res)
                test = "t-test"
            else:
                stat, pval = mannwhitneyu(sens, res)
                test = "Mann-Whitney U"

            drug_name = drug_name_map.get(drug, drug)
            plot_path = os.path.join(boxplot_dir, f"{drug}_{lineage}_boxplot.png")
            test_result = f"{test} p = {pval:.2e}"

            plot_drug_boxplot(drug, drug_name, lineage, values.loc[bins.index], bins, plot_path,
                              FACTOR_OF_INTEREST, test_result)
