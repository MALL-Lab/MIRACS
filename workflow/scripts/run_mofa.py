"""
run_mofa.py

Train a single MOFA model from tidy input data using mofapy2.

Inputs:
- df: tidy-format DataFrame with columns ['sample', 'feature', 'value', 'view', 'group']
- n_factors: number of latent factors for MOFA model
- output_path: file path to save trained model (.hdf5)

Requirements:
- mofapy2 must be installed and working
"""

import os
from mofapy2.run.entry_point import entry_point

def run_mofa(df, n_factors, output_path):
    # Initialize MOFA entry point
    ent = entry_point()

    # Set data options
    ent.set_data_options(scale_views=True)

    # Set tidy-format input (long format with sample, feature, value, view, group)
    ent.set_data_df(df, likelihoods=['gaussian'] * df["view"].nunique())

    # Set model options
    ent.set_model_options(
        factors=n_factors,
        spikeslab_weights=True,
        ard_weights=True,
        ard_factors=True
    )

    # Set training options
    ent.set_train_options(
        convergence_mode="fast",
        dropR2=None,
        gpu_mode=False,
        seed=1993,
        save_interrupted=True
    )

    # Build and run
    ent.build()
    ent.run()

    # Save model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ent.save(outfile=output_path)

def get_model_filename(n_factors: int) -> str:
    return f"mofa_model_{n_factors}f.hdf5"