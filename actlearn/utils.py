import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, inchi
from rdkit.Chem.MolStandardize import rdMolStandardize
import scipy
import sklearn

params = rdMolStandardize.CleanupParameters()
params.maxTautomers = 100
params.maxTransforms = 100
te = rdMolStandardize.TautomerEnumerator(params)

def rxn_dfp(rsmi):
    r, _, p = rsmi.split('>')
    rfp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(r), 4, useChirality=True)
    pfp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(p), 4, useChirality=True)
    return np.array(pfp) - np.array(rfp)


def add_fpvec(df, fpl, mol_col="mol"):
    if mol_col not in df.columns:
        df["mol"] = df["smiles"].apply(Chem.MolFromSmiles)
    return df["mol"].apply(lambda m: Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(m,fpl))


def add_reaction_fp(df_in, reaction_smiles_col="reaction_smarts", reaction_col="reaction", reaction_fp_col="rxnfp"):
    df = df_in.copy()
    fp_params = Chem.rdChemReactions.ReactionFingerprintParams()
    fp_params.fpType = Chem.rdChemReactions.FingerprintType.MorganFP
    if reaction_col not in df.columns:
        df[reaction_col] = df[reaction_smiles_col].apply(Chem.rdChemReactions.ReactionFromSmarts)
    df[reaction_fp_col] = df[reaction_col].apply(AllChem.CreateDifferenceFingerprintForReaction, ReactionFingerPrintParams=fp_params)
    return df

def complement(df1, df2, columns):
    return pd.concat([df1, df2, df2]).drop_duplicates(subset = columns, keep = False)


def inchi_to_smiles(inchi):
    if not inchi:
        return None
    mol = Chem.MolFromInchi(inchi)
    if not mol:
        return None
    return Chem.MolToSmiles(te.Canonicalize(mol))


def add_inchikeys(inrep):
    if type(inrep) == str:
        mol = Chem.MolFromSmiles(inrep)
    else:
        mol = inrep
    try:
        inchikey = inchi.MolToInchiKey(mol, options="/FixedH")
    except:
        inchikey = None
    return inchikey


def score_model(y_val, y_pred):
    performance = {}
    performance["r2"] = sklearn.metrics.r2_score(y_val, y_pred)
    performance["mae"] = sklearn.metrics.mean_absolute_error(y_val, y_pred)
    performance["rmse"] = sklearn.metrics.mean_squared_error(y_val, y_pred, squared=False)
    performance["spearmanr"] = scipy.stats.spearmanr(y_val, y_pred)
    performance["mean"] = np.mean(y_pred)

    for threshold in [10, 20, 30, 40, 50]:
        performance[f"{threshold}_acc"] = sklearn.metrics.accuracy_score(y_val > threshold, y_pred > threshold)
        performance[f"{threshold}_recall"] = sklearn.metrics.recall_score(y_val > threshold, y_pred > threshold, zero_division=0)
        performance[f"{threshold}_f1"] = sklearn.metrics.f1_score(y_val > threshold, y_pred > threshold)
        performance[f"{threshold}_precision"] = sklearn.metrics.precision_score(y_val > threshold, y_pred > threshold, zero_division=0)
        precision, recall, _ = sklearn.metrics.precision_recall_curve(y_val > threshold, y_pred > threshold)
        performance[f"{threshold}_prc-auc"] = sklearn.metrics.auc(recall, precision)

    return performance


def fit_and_score(train, test, target, feature_cols, n_trees=80, depth=6):
    rf_model = sklearn.ensemble.RandomForestRegressor(n_jobs=-1, n_estimators=n_trees, max_depth=depth)
    rf_model.fit(train[feature_cols], train[target])
    y_pred = rf_model.predict(test[feature_cols])
    return score_model(test[target], y_pred)


def replicate_score(train, test, feature_cols, repeat=50, n_trees=80, depth=6):
    scores = {}
    for i in range(repeat):
        scores[i] = (fit_and_score(train, test, feature_cols, n_trees, depth))
    return pd.DataFrame.from_dict(scores, orient="index").drop(columns="spearmanr").mean()


#def reshape_f1_df(score):
#    df = pd.DataFrame.from_dict(score, orient="index")
#    df_f1 = df[["10_f1", "20_f1", "30_f1", "40_f1", "50_f1"]].T
#    dfd1Tf1[test_name]["Threshold"] = [10,20,30,40,50]