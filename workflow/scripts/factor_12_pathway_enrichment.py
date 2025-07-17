"""
This script performs pathway and transcription factor enrichment for MOFA Factor12
using PROGENy and CoLLECTRI signatures. It uses MOFA weights (W matrix) from RNA
and protein views and visualizes significant scores in heatmaps.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import decoupler as dc
import anndata as ad
import mofax as mfx

from funct_enrich_W import run_enrichment_W  # Import enrichment function

# --- Configuration ---
data_dir = "../data"
models_dir = "../models"
model_name = "mofa_model_groups_30f"
model_path = os.path.join(models_dir, f"{model_name}.hdf5")
factor = "Factor12"

# --- Load data ---
rna_file = os.path.join(data_dir, "OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected.csv")
protein_file = os.path.join(data_dir, "prot", "protein_quant_current_normalized.csv")
model_df = pd.read_csv(os.path.join(data_dir, "Model.csv"))

rna_df = pd.read_csv(rna_file, index_col=0)
protein_df = pd.read_csv(protein_file, index_col=0)

# --- Load MOFA weights ---
m = mfx.mofa_model(model_path)
W = m.get_weights(df=True).T

# --- Split and clean RNA and protein views ---
cols_rna = [col for col in W.columns if 'rna' in col]
cols_prot = [col for col in W.columns if 'prot' in col]

W_rna = W[cols_rna].copy()
W_prot = W[cols_prot].copy()

W_rna.columns = W_rna.columns.str.extract(r'^([^(]+)')[0].str.strip()
W_prot.columns = W_prot.columns.str.replace('protprotprotprot', '', regex=False)

# --- Map protein features to gene symbols ---
valid_prot_cols = [col for col in W_prot.columns if col in protein_df.index]
W_prot = W_prot[valid_prot_cols]
W_prot.columns = protein_df.loc[W_prot.columns, 'Gene_Symbol'].values
W_prot = W_prot.groupby(W_prot.columns, axis=1).min()

print("W_rna shape:", W_rna.shape)
print("W_prot shape (deduplicated):", W_prot.shape)

# --- Get signatures ---
net_progeny = dc.op.progeny(organism='human')
net_collectri = dc.op.collectri(organism='human')

net_p_rna = net_progeny[net_progeny['target'].isin(W_rna.columns)]
net_p_prot = net_progeny[net_progeny['target'].isin(W_prot.columns)]
net_c_rna = net_collectri[net_collectri['target'].isin(W_rna.columns)]
net_c_prot = net_collectri[net_collectri['target'].isin(W_prot.columns)]

# --- Run enrichment ---
scores_rna_prog, pvals_rna_prog = run_enrichment_W(W_rna, net_p_rna, return_pvals=True)
scores_prot_prog, pvals_prot_prog = run_enrichment_W(W_prot, net_p_prot, return_pvals=True)
scores_rna_coll, pvals_rna_coll = run_enrichment_W(W_rna, net_c_rna, return_pvals=True)
scores_prot_coll, pvals_prot_coll = run_enrichment_W(W_prot, net_c_prot, return_pvals=True)

# --- Extract Factor12 scores and p-values ---
def extract_factor_df(adata, sources, factor_name):
    scores_df = pd.DataFrame(adata.obsm['score_ulm'], index=adata.obs_names, columns=sources)
    pvals_df = pd.DataFrame(adata.obsm['padj_ulm'], index=adata.obs_names, columns=sources)
    return scores_df.loc[factor_name], pvals_df.loc[factor_name]

sources_prog = sorted(net_p_rna['source'].unique())
sources_coll = sorted(net_c_rna['source'].unique())

scores_factor12_rna_prog, pvals_factor12_rna_prog = extract_factor_df(scores_rna_prog, sources_prog, factor)
scores_factor12_prot_prog, pvals_factor12_prot_prog = extract_factor_df(scores_prot_prog, sources_prog, factor)
scores_factor12_rna_coll, pvals_factor12_rna_coll = extract_factor_df(scores_rna_coll, sources_coll, factor)
scores_factor12_prot_coll, pvals_factor12_prot_coll = extract_factor_df(scores_prot_coll, sources_coll, factor)

# --- Filter significant features ---
def filter_significant(scores, pvals, alpha=0.05):
    mask = (pvals < alpha)
    return scores.where(mask)

# --- Plot PROGENy heatmap ---
df_prog = pd.concat([
    scores_factor12_rna_prog.rename('RNA'),
    scores_factor12_prot_prog.rename('Protein')
], axis=1)

mask_prog = (pvals_factor12_rna_prog < 0.05) | (pvals_factor12_prot_prog < 0.05)
df_prog = df_prog[mask_prog]

plt.figure(figsize=(3, len(df_prog) * 0.35))
sns.heatmap(df_prog, annot=True, cmap='coolwarm', center=0, cbar_kws={'label': 'Activity Score'})
plt.title("Factor12 – PROGENy\nSignaling Pathways", fontsize=10)
plt.tight_layout()
plt.show()

# --- Plot CoLLECTRI heatmap ---
df_coll = pd.concat([
    scores_factor12_rna_coll.rename('RNA'),
    scores_factor12_prot_coll.rename('Protein')
], axis=1)

mask_coll = (pvals_factor12_rna_coll < 0.05) | (pvals_factor12_prot_coll < 0.05)
df_coll = df_coll[mask_coll]

plt.figure(figsize=(2.5, len(df_coll) * 0.25))
sns.heatmap(df_coll, annot=True, cmap='coolwarm', center=0, cbar_kws={'label': 'Activity Score'})
plt.title("Factor12 – CoLLECTRI\nTranscription Factor Activity", fontsize=10)
plt.tight_layout()
plt.show()
