
Explanation of scripts in actlearn:
* select.py: Script for performing Kriging-Believer selection
* utils.py: utility functions for select.py

Explanation of jupyter notebooks:
* batch-1.ipynb, batch-2.ipynb, and batch-3.ipynb are the first 3 rounds of active learning selections on the base 8 core set
* batch-4.ipynb is the 4th round of active learning, which occured on the expanded core set
* batch-5_RF_ALvsRandomPicks.ipynb is a batch for evaluating the active learning and random sampling models (corresponding to the results shown in figure 9 of the paper)
* cross_picks.ipynb is an analysis script of the results from batch-5
* feature_importance.ipynb looked at fingerprints and QM features in how relevant they were for model performance
* partial_batch_yieldpred.ipynb tried to address whether we could have gotten a decent model with the expanded core set if we'd made fewer selections during batch 4
* model_hyperopt.ipynb contains the hyperparameter optimization calculations and figures related to model performance
* replicate_batch_yield.ipynb generates figures related to model performance with varying amounts of data

Explanation of data files:
* experimental_results_final.csv: Batches for all experimental data collected in this study. This is the file to use for most machine learning and analysis on the data collected.
* base_reaction_domain.csv: the reaction domain for the base core set, as used by several jupyter notebooks for batch selections
* batch2_experimental_results.csv: the reaction yield results used for making batch 2 selections
* BrBr_dft_selectfeatures_alkylBr.csv: DFT features for alkyl bromide search space
* BrBr_dft_selectfeatures_aryl.csv: DFT features for aryl bromides in base core set
* BrBr_dft_selectfeatures_aryl_cores9t12.csv: DFT features for aryl bromides in expanded core set
* brbr_reaction_domain_batch2.csv up to batch 5 are snapshots of the domain for in batches 2 to 5 as they are augmented with experimental data
* experimental_results_batch5.csv: the experimental results after all selections made by active learning or random sampling, augmented with results from cross picks
* Excel files starting with SOUZALUH are outputs from the experiments which are incorporated during some of the batch notebook, and are unfortunately necessary for reproducing the notebook outputs. 

Although it would have been preferable to condense these files down into fewer files, we found that due to rounding during file I/O within the selection process could have an effect on the model selections, so we instead reproduce here copies of the original files to make the batch selection notebooks more reproducible.