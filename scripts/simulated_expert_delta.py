import pickle
import numpy as np
from utils import fingerprints_from_mol, double_sigmoid
from tdc import Oracle

class logPEvaluationModel():
    def __init__(self):
        pass

    def oracle_score(self, smi):
        oracle = Oracle(name = 'LogP')
        if smi:
            score = oracle(smi)
            return float(score)
        return 0.0

    def human_score(self, smi, noise):
        if smi:
            human_score = self.oracle_score(smi) + noise
            print("human score :", human_score)
            return float(human_score)
        else:
            return 0.0

    def utility(self, x, low, high, alpha_1 = 10, alpha_2 = 10):
        utility_score = double_sigmoid(x, low, high, alpha_1, alpha_2)
        return utility_score

class ActivityEvaluationModel():
    """Oracle scores based on an ECFP classifier for activity."""

    def __init__(self, path):
        with open(path, "rb") as f:
            self.clf = pickle.load(f)

    def predict_proba(self, mol):
        if mol:
            fp = fingerprints_from_mol(mol)
            score = self.clf.predict_proba(fp)[:, 1]
            return float(score)
        return 0.0

    def feedback_score(self, mol, noise, reward):
        "Takes in as input an RDKit mol object and outputs a value between 0 and 1"
        # calculate molecular features
        if mol: # if mol is not a NoneType Object
            fp = fingerprints_from_mol(mol)
            human_score = self.sigmoid(self.clf.decision_function(fp) + noise) # based on the SVM oracle from Olivecrona et al.
            delta = np.abs(human_score - reward)
            print(human_score, reward, 1-delta)
            return 1-delta
        return 0.0

    def sigmoid(self, score):
        return 1/(1 + np.exp(-score))
