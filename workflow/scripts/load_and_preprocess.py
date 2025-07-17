"""
load_and_preprocess.py

Loads and preprocesses RNA-seq, proteomics, and metadata, and formats data for MOFA input.

Inputs:
- Trasncriptómics: OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected.csv
- Proteomics: protein_quant_current_normalized.csv
- Metadata: Model.csv

Outputs:
- rna_df: Raw RNA-seq data (samples x genes)
- protein_df_ready: Cleaned proteomics (samples x proteins)
- mofa_input_df: Tidy format input for MOFA with view and group columns
- metadata_df: Metadata with sample annotations
"""

import os
import pandas as pd
import re

def load_and_preprocess(data_dir="data"):
    # File paths
    rna_path = os.path.join(data_dir, "OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected.csv")
    protein_path = os.path.join(data_dir, "prot", "protein_quant_current_normalized.csv")
    metadata_path = os.path.join(data_dir, "Model.csv")

    # Load data
    rna_df = pd.read_csv(rna_path, index_col=0)
    protein_df = pd.read_csv(protein_path, index_col=0)
    metadata_df = pd.read_csv(metadata_path)

    # ----------------------------
    # Preprocessing Proteomics
    # ----------------------------
    preferred_replicates = [
        'SW948_LARGE_INTESTINE_TenPx20',
        'CAL120_BREAST_TenPx28',
        'HCT15_LARGE_INTESTINE_TenPx18'
    ]

    columns_to_keep = [
        col for col in protein_df.columns
        if not any(rep in col for rep in ['SW948_LARGE_INTESTINE', 'CAL120_BREAST', 'HCT15_LARGE_INTESTINE'])
    ] + preferred_replicates
    protein_df = protein_df[columns_to_keep]
    protein_df = protein_df.reset_index()

    ccle_to_depmap = dict(zip(metadata_df["CCLEName"], metadata_df["ModelID"]))

    expression_columns = ["Protein_Id"] + [
        col for col in protein_df.columns
        if re.match(r"^[A-Z0-9\-]+_[A-Z_]+_TenPx\d+$", col)
    ]
    expr_df = protein_df[expression_columns]
    proteins = expr_df["Protein_Id"].values

    expr_transposed = expr_df.drop(columns="Protein_Id").transpose()
    expr_transposed.columns = proteins
    expr_transposed.index.name = "CCLEName"
    expr_transposed["CCLEName"] = expr_transposed.index.str.replace(r"_TenPx\d+$", "", regex=True)
    expr_transposed["ModelID"] = expr_transposed["CCLEName"].map(ccle_to_depmap)
    expr_clean = expr_transposed.dropna(subset=["ModelID"])
    expr_clean = expr_clean.drop(columns=["CCLEName"]).set_index("ModelID")
    expr_clean = expr_clean.loc[:, ~expr_clean.columns.duplicated()]
    expr_clean = expr_clean.dropna(axis=1, thresh=int(0.5 * expr_clean.shape[0]))
    protein_df_ready = expr_clean.copy()

    # ----------------------------
    # Prepare MOFA input (tidy format)
    # ----------------------------
    rna_df.index.name = "ModelID"
    rna_long = rna_df.reset_index().melt(id_vars="ModelID", var_name="feature", value_name="value")
    rna_long = rna_long.rename(columns={"ModelID": "sample"})
    rna_long["view"] = "rna"

    prot_long = protein_df_ready.reset_index().melt(id_vars="ModelID", var_name="feature", value_name="value")
    prot_long = prot_long.rename(columns={"ModelID": "sample"})
    prot_long["view"] = "prot"

    # Assign groups from metadata
    group_map = metadata_df.set_index("ModelID")["OncotreeLineage"].to_dict()
    for df in [rna_long, prot_long]:
        df["group"] = df["sample"].map(group_map).fillna("Unassigned")
        df["group"] = df["group"].str.replace("/", "_").str.replace(" ", "_")

    # Concatenate views
    mofa_input_df = pd.concat([rna_long, prot_long], ignore_index=True)

    return rna_df, protein_df_ready, mofa_input_df, metadata_df