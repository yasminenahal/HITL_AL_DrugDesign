import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

def fingerprints_from_mol(mol, type = "counts", size = 2048, radius = 3):
    "and kwargs"

    if type == "binary":
        if isinstance(mol, list):
            fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius, useFeatures=True) for m in mol if m is not None]
            l = len(mol)
        else:
            fps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius, useFeatures=True)]
            l = 1
        nfp = np.zeros((l, size), np.int32)
        for i in range(l):
            for idx,v in enumerate(fps[i]):
                nfp[i, idx] += int(v)

    if type == "counts":
        if isinstance(mol, list):
            fps = [AllChem.GetMorganFingerprint(m, radius, useCounts=True, useFeatures=True) for m in mol if m is not None]
            l = len(mol)
        else:
            fps = [AllChem.GetMorganFingerprint(mol, radius, useCounts=True, useFeatures=True)]
            l = 1
        nfp = np.zeros((l, size), np.int32)
        for i in range(l):
            for idx,v in fps[i].GetNonzeroElements().items():
                nidx = idx%size
                nfp[i, nidx] += int(v)
    
    return nfp

def double_sigmoid(x, low, high, alpha_1, alpha_2):
    return 10**(x*alpha_1)/(10**(x*alpha_1)+10**(low*alpha_1)) - 10**(x*alpha_2)/(10**(x*alpha_2)+10**(high*alpha_2))
