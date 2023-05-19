import pickle
from utils import fingerprints_from_mol
from numpy.random import binomial
from rdkit.Chem.Crippen import MolLogP

class logPEvaluationModel():
    def __init__(self):
        pass

    def calc_logp(self, mol):
        if mol:
            score = MolLogP(mol)
            return float(score)
        return 0.0

    def feedback_score(self, mol, noise):
        if mol:
            score = self.calc_logp(mol) + noise
            return float(score)
        return 0.0

class ActivityEvaluationModel():
    """Scores based on an ECFP classifier for activity."""

    def __init__(self, path):
        with open(path, "rb") as f:
            self.clf = pickle.load(f)

    def predict(self, mol):
        if mol:
            fp = fingerprints_from_mol(mol)
            c = self.clf.predict(fp)
            return int(c)
        else:
            return 0

    def predict_proba(self, mol):
        if mol:
            fp = fingerprints_from_mol(mol)
            score = self.clf.predict_proba(fp)[:, 1]
            return float(score)
        return 0.0

    def old_feedback_score(self, mol, noise):
        "Takes in as input an RDKit mol object and outputs a value between 0 and 1"
        # calculate molecular features
        if mol: # if mol is not a NoneType Object
            fp = fingerprints_from_mol(mol)
            score = self.clf.decision_function(fp) + noise # based on the SVM oracle from Olivecrona et al.
            return float(score)
        return 0.0

    def feedback_bernoulli(self, mol, pi_human):
        if mol:
            c = self.predict(mol) # oracle
            c_human = c * binomial(1, pi_human) + (1-c) * binomial(1, 1 - pi_human)
            return int(c_human)
        else:
            return 0