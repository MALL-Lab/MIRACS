from load_and_preprocess import load_and_preprocess
from run_mofa import run_mofa, get_model_filename

# 1. Load and preprocess RNA and protein data
rna_df, protein_df, mofa_input_df, metadata_df = load_and_preprocess()

# 2. Train MOFA model with 30 factors
model_path = f"models/{get_model_filename(30)}"
run_mofa(mofa_input_df, n_factors=30, output_path=model_path)
