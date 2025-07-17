Project: Identificación de factores biológicos que influyen en
la respuesta a fármacos anticancerígenos mediante
análisis multi-ómico no supervisado

This project integrates transcriptomics and proteomics data from cancer cell lines 
to analyze drug response patterns using MOFA (Multi-Omics Factor Analysis) and 
pathway enrichment methods.


Data Used (https://depmap.org/portal/)
1. RNA-seq data 24Q4:
   - OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected.csv
2. Proteomics data:
   - protein_quant_current_normalized.csv
3. Cell line metadata:
   - Model.csv

4. Drug response data:
   - PRISM 19Q4: secondary-screen-dose-response-curve-parameters19q.csv

   - Sanger: sanger-dose-response.csv + screened_compounds_rel_8.5.csv


How to Run
1. Run the script: run_train_mofa.py
2. Run: drug_enrichment_analysis.py
3. Run: factor_12_analysis.py
4. Run: factor_12_pathway_enrichment.py
optional. Run: compare_prism_sanger.py 
------------------------
Environment Setup
------------------------
This project was run in a conda environment called `env_miracs` (included in scripts file).
