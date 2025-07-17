"""
funct_enrich_W.py

This module provides a utility function to perform pathway or transcription factor
enrichment using decoupler's model-based methods (e.g., ULM, MLM) on a MOFA W matrix.

The input W matrix should be shaped as (factors × genes), where each row is a latent
factor and each column is a gene symbol. The function supports filtering the network
by gene presence and allows returning both enrichment scores and adjusted p-values.

Typical use cases:
- Pathway activity scoring with PROGENy signatures
- Transcription factor activity estimation with CoLLECTRI
"""

import pandas as pd
import anndata as ad
import decoupler as dc

def run_enrichment_W(W_mat, net, method='ulm', tmin=5, return_pvals=False, verbose=True):
    """
    Runs decoupler enrichment (e.g., ULM or MLM) on a W matrix.

    Parameters:
        W_mat (pd.DataFrame): Weight matrix (factors × genes).
        net (pd.DataFrame): Signature network with 'source', 'target', and 'weight' columns.
        method (str): Method name in decoupler.mt (e.g., 'ulm', 'mlm').
        tmin (int): Minimum number of genes per pathway or TF.
        return_pvals (bool): If True, returns both scores and adjusted p-values.
        verbose (bool): Whether to show progress messages.

    Returns:
        pd.DataFrame or tuple: Scores (factors × pathways or TFs), and optionally p-values.
    """

    # Ensure gene names are string and properly formatted
    W_mat.columns = W_mat.columns.astype(str).str.strip()
    W_mat.columns.name = None

    # Convert to AnnData format required by decoupler
    adata = ad.AnnData(X=W_mat.values)
    adata.obs_names = W_mat.index
    adata.var_names = W_mat.columns
    adata.var_names.name = None

    # Filter network to keep only genes present in W
    net_filtered = net[net['target'].isin(adata.var_names)]

    # Validate network columns
    required_cols = {'source', 'target', 'weight'}
    if not required_cols.issubset(net_filtered.columns):
        raise ValueError(f"Network must contain the columns: {required_cols}")

    # Run selected enrichment method from decoupler
    enrichment_func = getattr(dc.mt, method)
    enrichment_func(data=adata, net=net_filtered, tmin=tmin, verbose=verbose)

    # Extract scores and optionally adjusted p-values
    scores = dc.pp.get_obsm(adata, key=f'score_{method}')
    if return_pvals:
        pvals = dc.pp.get_obsm(adata, key=f'padj_{method}')
        return scores, pvals
    return scores
