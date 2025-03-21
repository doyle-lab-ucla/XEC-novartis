from joblib import Parallel, delayed, dump, load
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, AllChem
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import tqdm
from typing import List
from .utils import complement


def predict_posterior(model: RandomForestRegressor, X: np.ndarray):
    """
    Produces a predictive distribution from the trees in a random foret model.

    Arguments:
    - model: sklearn.ensemble.RandomForestRegressor
    - X: Input to the model. Shape: num_samples x num_features

    """
    # Because we want check_input=False below
    X = model._validate_X_predict(X)

    pred_dist = Parallel(n_jobs=-1, verbose=0, backend="threading")(
        delayed(tree.predict)(X, check_input=False) for tree in model.estimators_
    )

    # Each row corresponds to the tree predictions for a sample.
    # Shape: num_samples x num_trees
    pred_dist = np.transpose(np.vstack(pred_dist))

    # Compute the mean and average over the trees.
    mean = np.mean(pred_dist, axis=1)
    std = np.std(pred_dist, axis=1)

    return mean, std


def score_regression(y_val, y_pred):
    r2 = metrics.r2_score(y_val, y_pred)
    mae = metrics.mean_absolute_error(y_val, y_pred)
    rmse = metrics.mean_squared_error(y_val, y_pred, squared=False)
    return r2, mae, rmse


def diverse_pick(df, n_molecules, fp_col="fp_bitvec"):
    if fp_col not in df.columns:
        df[fp_col] = utils.add_fpvec(df, fpl=2, mol_col="mol")
    fp_vec = list(df["fp_bitvec"])
    picker = MaxMinPicker()
    indices = picker.LazyBitVectorPick(fp_vec, len(fp_vec), n_molecules)
    return df.loc[list(indices)]


def random_pick(df, n):
    return df.sample(n)


def max_rfr_stdev(regressor, X):
    _, std = predict_posterior(regressor.estimator, X)
    query_idx = np.argmax(std)
    return query_idx


def train_predict_rf(df_train, df_val, x_cols, target):
    model = RandomForestRegressor(n_estimators=150)
    model.fit(df_train[x_cols], df_train[target])
    y_pred = model.predict(df_val[x_cols])
    return score_regression(df_val[target], y_pred)


def evaluate_sampler(
    df_pool, df_test, x_cols, target, n_samples, strategy="random", replicates=10
):
    avg_rmse_div, avg_r2_div = {}, {}
    for n in n_samples:
        r2_all, rmse_all = [], []
        for i in range(replicates):
            if strategy == "random":
                df_sample = df_pool.sample(n)
            elif strategy == "diverse":
                df_sample = diverse_pick(df_pool, n)
            r2, mae, rmse = train_predict_rf(df_sample, df_test, x_cols, target)
            r2_all.append(r2)
            rmse_all.append(rmse)

        avg_rmse_div[n] = np.mean(rmse_all)
        avg_r2_div[n] = np.mean(r2_all)
    return avg_r2_div, avg_rmse_div


def kriging_believer(
    domain: pd.DataFrame,
    results: pd.DataFrame,
    x_cols: List[str],
    batch_size: int,
    id_column: str,
    target_column: str,
    bb_smiles_column: str,
    output_dir: str,
):

    assert id_column is not None
    assert target_column is not None

    domain_copy = domain.copy()
    results_copy = results.copy()

    results_copy.drop(x_cols, axis=1).to_csv(f"{output_dir}/results.csv")

    proposed_experiments = pd.DataFrame()
    for i in tqdm.tqdm(range(batch_size)):

        train_X = results_copy[x_cols].values
        train_y = results_copy[target_column].values

        model = RandomForestRegressor(n_jobs=-1, n_estimators=500, random_state=42)
        model.fit(train_X, train_y)
        dump(model, f"{output_dir}/model_{i:02}.pkl")

        domain_complement_results = complement(domain_copy, results_copy, [id_column])
        X = domain_complement_results[x_cols].values
        mean, std = predict_posterior(model, X)

        domain_complement_results["std"] = std
        domain_complement_results[target_column] = mean

        domain_complement_results.drop(x_cols, axis=1).to_csv(
            f"{output_dir}/domain_{i:02}.csv", index=False
        )

        bb_ = (
            domain_complement_results.groupby(by=[bb_smiles_column])["std"]
            .agg("mean")
            .sort_values(ascending=False)
            .index[0]
        )

        proposed = domain_complement_results[
            domain_complement_results[bb_smiles_column] == bb_
        ]
        proposed_experiments = pd.concat([proposed_experiments, proposed])
        results_copy = pd.concat([results_copy, proposed.drop(columns=["std"])])

    proposed_experiments.drop(x_cols, axis=1).to_csv(
        f"{output_dir}/proposed_experiments.csv"
    )

    with Chem.SDWriter(f"{output_dir}/selections.sdf") as w:
        for smi in proposed_experiments[bb_smiles_column].unique():
            w.write(Chem.MolFromSmiles(smi))

    return proposed_experiments


def rxn_dfp(rsmi):
    r, _, p = rsmi.split('>')
    rfp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(r), 4, useChirality=True)
    pfp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(p), 4, useChirality=True)
    return np.array(pfp) - np.array(rfp)