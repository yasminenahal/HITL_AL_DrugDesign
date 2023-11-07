import numpy as np

from rdkit import Chem

from utils import fingerprints_from_mol
from scripts.simulated_expert import logPEvaluationModel


def local_idx_to_fulldata_idx(N, selected_feedback, idx):
    all_idx = np.arange(N)
    mask = np.ones(N, dtype=bool)
    mask[selected_feedback] = False
    pred_idx = all_idx[mask]
    try:
        pred_idx[idx]
        return pred_idx[idx]
    except:
        valid_idx = [i if 0 <= i < len(pred_idx) else len(pred_idx) - 1 for i in idx]
        return pred_idx[valid_idx]

def epig(data, n, smiles, fit, selected_feedback, is_counts = True, rng = None, t = None):
    """
    data: pool of unlabelled molecules
    n: number of queries to select
    smiles: array-like object of high-scoring smiles
    selected_feedback: previously selected in previous feedback rounds
    is_counts: depending on whether the model was fitted on counts (or binary) molecular features
    """
    N = len(data)
    data = data.sample(frac = 0.2).sort_index()
    mols_pool = [Chem.MolFromSmiles(s) for s in data.SMILES]
    mols_target = [Chem.MolFromSmiles(s) for s in smiles[:1000]]
    # calculate fps for the pool molecules
    fps_pool = fingerprints_from_mol(mols_pool)
    if not is_counts:
        fps_pool = fingerprints_from_mol(mols_pool, type = 'binary')
    # calculate fps for the target molecules
    fps_target = fingerprints_from_mol(mols_target)
    if not is_counts:
        fps_target = fingerprints_from_mol(mols_target, type = 'binary')
    probs_pool = fit._get_prob_distribution(fps_pool)
    probs_target = fit._get_prob_distribution(fps_target)
    estimated_epig_scores = fit._estimate_epig(probs_pool, probs_target)
    query_idx = np.argsort(estimated_epig_scores.numpy())[::-1][:n]
    return local_idx_to_fulldata_idx(N, selected_feedback, query_idx)    

def uncertainty_sampling(data, n, smiles, fit, selected_feedback, is_counts = True, rng = None, t = None):
    """
    data: pool of unlabelled molecules
    n: number of queries to select
    smiles: array-like object of high-scoring smiles
    selected_feedback: previously selected in previous feedback rounds
    is_counts: depending on whether the model was fitted on counts (or binary) molecular features
    """
    N = len(data)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fps = fingerprints_from_mol(mols)
    if not is_counts:
        fps = fingerprints_from_mol(mols, type = 'binary')
    estimated_unc = fit._uncertainty(fps)
    print(estimated_unc)
    query_idx = np.argsort(estimated_unc)[::-1][:n]
    return local_idx_to_fulldata_idx(N, selected_feedback, query_idx)

def vote_entropy_qbc(data, n , smiles, fit, selected_feedback, is_counts = True, rng = None, t = None):
    N = len(data)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fps = fingerprints_from_mol(mols)
    if not is_counts:
        fps = fingerprints_from_mol(mols, type = 'binary')
    estimated_unc = fit._qbc(fps)
    query_idx = np.argsort(estimated_unc)[::-1][:n] # get the n highest entropies
    return local_idx_to_fulldata_idx(N, selected_feedback, query_idx)

def exploitation_classification(data, n, smiles, fit, selected_feedback, is_counts = True, rng = None, t = None):
    N = len(data)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fps = fingerprints_from_mol(mols)
    print(fit)
    if not is_counts:
        fps = fingerprints_from_mol(mols, type = 'binary')
    score_pred = fit._predict_proba(fps)[:,1]
    query_idx = np.argsort(score_pred)[::-1][:n] # get the n highest
    return local_idx_to_fulldata_idx(N, selected_feedback, query_idx)

def exploitation_regression(data, n, smiles, fit, selected_feedback, is_counts = True, rng = None, t = None):
    N = len(data)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fps = fingerprints_from_mol(mols)
    if not is_counts:
        fps = fingerprints_from_mol(mols, type = 'binary')
    evaluation_model = logPEvaluationModel()
    values = fit._predict(fps)
    score_pred = [evaluation_model.utility(v, low = 2, high = 4) for v in values]
    query_idx = np.argsort(score_pred)[::-1][:n] # get the n highest
    return local_idx_to_fulldata_idx(N, selected_feedback, query_idx)

def margin_selection(data, n, smiles, fit, selected_feedback, is_counts = True, rng = None, t = None):
    N = len(data)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fps = fingerprints_from_mol(mols)
    if not is_counts:
        fps = fingerprints_from_mol(mols, type = 'binary')
    predicted_prob = fit._predict_proba(fps)
    rev = np.sort(predicted_prob, axis=1)[:, ::-1]
    values = rev[:, 0] - rev[:, 1]
    query_idx = np.argsort(values)[:n]
    return local_idx_to_fulldata_idx(N, selected_feedback, query_idx)

# def diversity_sampling
# 
# def expected_model_change

def random_selection(data, n, smiles, fit, selected_feedback, rng, t=None):
    N = len(data)
    selected = rng.choice(N-len(selected_feedback), n, replace=False)
    return local_idx_to_fulldata_idx(N, selected_feedback, selected)

def select_query(data, n, smiles, fit, selected_feedback, acquisition = 'random', rng = None, t = None):
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
    N = len(data)
    # select acquisition: (TODO: EIG, other strategies)
    if acquisition == 'uncertainty':
        acq = uncertainty_sampling
    elif acquisition == 'qbc':
        acq = vote_entropy_qbc
    elif acquisition == 'greedy_classification':
        acq = exploitation_classification
    elif acquisition == 'greedy_regression':
        acq = exploitation_regression
    elif acquisition == 'random':
        acq = random_selection
    elif acquisition == 'epig':
        acq = epig
    else:
        print("Warning: unknown acquisition criterion. Using random sampling.")
        acq = random_selection
    return acq(data, n, smiles, fit, selected_feedback, rng, t)
