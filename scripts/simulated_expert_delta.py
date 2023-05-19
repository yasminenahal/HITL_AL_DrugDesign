import pickle
import numpy as np
from utils import fingerprints_from_mol
from rdkit.Chem.Crippen import MolLogP

class logPEvaluationModel():
    def __init__(self):
        pass

    def calc_logp(self, mol):
        if mol:
            score = MolLogP(mol)
            return float(score)
        return 0.0

    def feedback_score(self, mol, noise, reward, human_agreement_threshold):
        if mol:
            human_score = self.calc_logp(mol) + noise
            delta = np.abs(human_score - reward)
            print("reward :", reward, "human score :", human_score)
            # if human disagrees
            if delta > human_agreement_threshold:
                return human_score
            else:
                return int(0)

    def feedback_label(self, mol, noise, human_range):
        if mol:
            human_score = self.calc_logp(mol) + noise
            # if human agrees
            if human_score >= human_range[0] and human_score <= human_range[1]:
                return 1
            else:
                return 0

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