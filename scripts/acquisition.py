import numpy as np

from rdkit import Chem

from random import sample

from utils import fingerprints_from_mol
from scripts.simulated_expert_delta import logPEvaluationModel

def local_idx_to_fulldata_idx(N, selected_feedback, idx):
    all_idx = np.arange(N)
    mask = np.ones(N, dtype=bool)
    mask[selected_feedback] = False
    pred_idx = all_idx[mask]
    return pred_idx[idx]

def uncertainty_sampling(N, n, smiles, fit, selected_feedback, is_counts = True, rng = None, t = None):
    """
    N: number of molecules in the generated pool
    n: number of queries to select
    smiles: array-like object of high-scoring smiles
    selected_feedback: previously selected in previous feedback rounds
    is_counts: depending on whether the model was fitted on counts (or binary) molecular features
    """
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fps = fingerprints_from_mol(mols)
    if not is_counts:
        fps = fingerprints_from_mol(mols, type = 'binary')
    estimated_unc = fit._uncertainty(fps)
    print(estimated_unc)
    query_idx = np.argsort(estimated_unc)[::-1][:n]
    return local_idx_to_fulldata_idx(N, selected_feedback, query_idx)

def vote_entropy_qbc(N, n ,smiles, fit, selected_feedback, is_counts = True, rng = None, t = None):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fps = fingerprints_from_mol(mols)
    if not is_counts:
        fps = fingerprints_from_mol(mols, type = 'binary')
    estimated_unc = fit._qbc(fps)
    query_idx = np.argsort(estimated_unc)[::-1][:n] # get the n highest entropies
    return local_idx_to_fulldata_idx(N, selected_feedback, query_idx)

def exploitation(N, n, smiles, fit, selected_feedback, is_counts = True, rng = None, t = None):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fps = fingerprints_from_mol(mols)
    if not is_counts:
        fps = fingerprints_from_mol(mols, type = 'binary')
    score_pred = fit.predict_proba(fps)[:,1]
    query_idx = np.argsort(score_pred)[::-1][:n] # get the n highest
    return local_idx_to_fulldata_idx(N, selected_feedback, query_idx)

def exploitation_reg(N, n, smiles, fit, selected_feedback, is_counts = True, rng = None, t = None):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fps = fingerprints_from_mol(mols)
    if not is_counts:
        fps = fingerprints_from_mol(mols, type = 'binary')
    evaluation_model = logPEvaluationModel()
    values = fit.predict(fps)
    score_pred = [evaluation_model.utility(v, low = 2, high = 4) for v in values]
    query_idx = np.argsort(score_pred)[::-1][:n] # get the n highest
    return local_idx_to_fulldata_idx(N, selected_feedback, query_idx)

def margin_selection(N, n, smiles, fit, selected_feedback, is_counts = True, rng = None, t = None):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fps = fingerprints_from_mol(mols)
    if not is_counts:
        fps = fingerprints_from_mol(mols, type = 'binary')
    predicted_prob = fit.predict_proba(fps)
    rev = np.sort(predicted_prob, axis=1)[:, ::-1]
    values = rev[:, 0] - rev[:, 1]
    query_idx = np.argsort(values)[:n]
    return local_idx_to_fulldata_idx(N, selected_feedback, query_idx)

# def diversity_sampling
# 
# def expected_model_change

def random_selection(N, n, smiles, fit, selected_feedback, rng, t=None):
    selected = rng.choice(N-len(selected_feedback), n, replace=False)
    return local_idx_to_fulldata_idx(N, selected_feedback, selected)

def select_query(N, n, smiles, fit, selected_feedback, acquisition = 'random', rng = None, t = None):
    '''
    Parameters
    ----------
    smiles: array-like object of high-scoring smiles
    n: number of queries to select
    fit: fitted model at round k
    acquisition: acquisition type
    rng: random number generator

    Returns
    -------
    int idx: 
        Index of the query

    '''
    # select acquisition: (TODO: EIG, other strategies)
    if acquisition == 'uncertainty':
        acq = uncertainty_sampling
    elif acquisition == 'qbc':
        acq = vote_entropy_qbc
    elif acquisition == 'greedy':
        acq = exploitation
    elif acquisition == 'greedy_regression':
        acq = exploitation_reg
    elif acquisition == 'random':
        acq = random_selection
    else:
        print("Warning: unknown acquisition criterion. Using random sampling.")
        acq = random_selection
    return acq(N, n, smiles, fit, selected_feedback, rng, t)
