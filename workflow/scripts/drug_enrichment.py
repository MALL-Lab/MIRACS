"""
drug_enrichment.py

1. Correlation analysis between drug responses and MOFA factors.
2. GSEA-based enrichment of drug mechanisms of action (MOA) using decoupler >2.0.
3. High-level function to execute the full enrichment pipeline.

Functions:
- compute_correlation(): compute correlation between one drug and all features.
- correlation_drugs(): compute correlation matrix for all drugs × features.
- gsea_enrichment(): run GSEA enrichment using decoupler's multivariate tools.
- run_drug_enrichment(): high-level function that combines correlation and GSEA.

"""
import pandas as pd
from scipy.stats import pearsonr, kendalltau, spearmanr, zscore
from joblib import Parallel, delayed
import anndata as ad
import decoupler as dc

# 0.1 - Correlation between drug response and factors
def compute_correlation(drug_col, df_counts, df_drugs, min_len, method='spearman'):
    correlations = []
    df_suboutputs = df_drugs.loc[:, drug_col].dropna()
    
    for feature in df_counts.columns:
        if len(df_suboutputs.unique()) > min_len:
            df_subfeature = df_counts[feature].dropna()
            shared_cells = df_suboutputs.index.intersection(df_subfeature.index)

            if len(shared_cells) > 1:
                x = df_suboutputs[shared_cells].sort_index().values
                y = df_subfeature[shared_cells].sort_index().values
                if method == 'pearson':
                    correlation, _ = pearsonr(x, y)
                elif method == 'spearman':
                    correlation, _ = spearmanr(x, y)
                elif method == 'kendall':
                    correlation, _ = kendalltau(x, y)
                else:
                    raise ValueError("Unsupported correlation method.")
                correlations.append((feature, correlation))
            else:
                correlations.append((feature, np.nan))
        else:
            correlations.append((feature, np.nan))

    return drug_col, correlations

# 0.2 - Correlation matrix (drugs × factors)
def correlation_drugs(df_drugs, df_counts, threads, min_len=1, method='spearman'):
    correlation_matrix = pd.DataFrame(index=df_drugs.columns, columns=df_counts.columns)
    results = Parallel(n_jobs=threads)(
        delayed(compute_correlation)(drug_col, df_counts, df_drugs, min_len, method)
        for drug_col in df_drugs.columns
    )

    for drug_col, correlations in results:
        for feature, correlation in correlations:
            correlation_matrix.at[drug_col, feature] = correlation

    return correlation_matrix.astype(float)

# 0.3 - GSEA enrichment using decoupler (≥ 2.0)
def gsea_enrichment(matrix, net, shared_element, net_group, n_min, iter):
    # Drop rows (drugs) with NaNs
    matrix = matrix.dropna(axis=0, how='any')

    # Drop columns (factors) with NaNs
    matrix = matrix.dropna(axis=1, how='any')

    # Check if matrix is empty
    if matrix.empty:
        raise ValueError("❌ Matrix is empty after removing NaNs.")

    shared_elements = set(matrix.index).intersection(set(net[shared_element]))
    if not shared_elements:
        raise ValueError("❌ No shared elements between matrix and network.")

    net = net[net[shared_element].isin(shared_elements)].copy()
    net = net.rename(columns={net_group: 'source', shared_element: 'target'})

    matrix = matrix.loc[list(shared_elements), :]
    matrix = matrix.dropna(axis=1, how='any')

    if matrix.empty:
        raise ValueError("❌ Matrix is empty after selecting shared elements and cleaning.")

    adata = ad.AnnData(matrix.T.astype('float32'))

    dc.mt.gsea(
        data=adata,
        net=net,
        tmin=n_min,
        times=iter,
        verbose=True
    )

    if 'score_gsea' in adata.obsm:
        result_df_gsea = adata.obsm['score_gsea']
    else:
        raise KeyError(f"'score_gsea' not found. Available keys: {list(adata.obsm.keys())}")

    if 'padj_gsea' in adata.obsm:
        p_value_gsea = adata.obsm['padj_gsea']
    else:
        raise KeyError("'padj_gsea' not found in adata.obsm.")

    return result_df_gsea, p_value_gsea

# 0.4 - Enrichment of drug groups based on NES
def run_drug_enrichment(df_drugs, df_counts, drug_net, shared_elements, group,
                        min_elements=5, number_of_threads=-1, return_cormatrix=False,
                        min_len=10, method='spearman', iteration=100000):
    cormatrix = correlation_drugs(df_drugs, df_counts, number_of_threads, min_len, method)

    df_scores, df_pvalue = gsea_enrichment(
        matrix=cormatrix,
        net=drug_net,
        shared_element=shared_elements,
        net_group=group,
        n_min=min_elements,
        iter=iteration
    )

    if return_cormatrix:
        return df_scores, df_pvalue, cormatrix
    else:
        return df_scores, df_pvalue
