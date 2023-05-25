import pandas as pd
import numpy as np

import functools
from multiprocessing import Pool

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors



#class standardizer

#class cleanser

class MorganECFPBits:
    #add possibility to do feature selection based on colinearity
    def __init__(self, df):
        self.mols = [Chem.MolFromSmiles(s) for s in df.smiles]
        self.smiles = df.smiles.values
        self.activity = df.activity.values

    def mol2fp(self, mol, radius = 3, n_bits = 2048):
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius = radius, nBits = n_bits)
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        print("**** Processing Fingerprint calculation")

        return arr

    def compute_ecfp_bits(self, radius = 3, n_bits = 2048, path_name = "./"):
        bit_headers = ["bit"+str(i) for i in range(n_bits)]
        arr = np.empty((0, n_bits), int).astype(int)

        for mol in self.mols:
            fp = self.mol2fp(mol, radius = radius, n_bits = n_bits)
            arr = np.vstack((arr, fp))

        df = pd.DataFrame(np.asarray(arr).astype(int), columns = bit_headers)

        df.insert(loc = 0, column = "smiles", value = self.smiles)
        df.insert(loc = 1, column = "activity", value = self.activity)

        df.to_csv(path_name+"ECFP_bits.csv", index = False)

        return df

    def compute_ecfp_bits_mp(self, radius = 3, n_bits = 2048, path_name = "./"):
        bit_headers = ["bit"+str(i) for i in range(n_bits)]

        partial_mol2fp = functools.partial(self.mol2fp, radius = radius, n_bits = n_bits)

        fps = []
        with Pool() as pool:
            for fp in list(pool.map(partial_mol2fp, self.mols)):
                fps.append(fp)

        df = pd.DataFrame(np.stack(fps).astype(int), columns = bit_headers)

        df.insert(loc = 0, column = "smiles", value = self.smiles)
        df.insert(loc = 1, column = "activity", value = self.activity)

        df.to_csv(path_name+"ECFP_bits.csv", index = False)

        return df

class MorganECFPCounts:
    def __init__(self, df):
        self.mols = [Chem.MolFromSmiles(s) for s in df.smiles]
        self.smiles = df.smiles.values
        self.activity = df.activity.values

    def compute_ecfp_counts(self, radius = 3, n_bits = 2048, use_features = True, path_name = "./"):
        bit_headers = ["bit"+str(i) for i in range(n_bits)]
        fps = []

        for mol in self.mols:
            fp = AllChem.GetHashedMorganFingerprint(mol, radius, nBits = n_bits, useFeatures = use_features)
            fps.append([self._to_numpy_arr(fp, dtype = np.int32)])

        df = pd.DataFrame(np.vstack(fps), columns = bit_headers)

        df.insert(loc = 0, column = "smiles", value = self.smiles)
        df.insert(loc = 1, column = "activity", value = self.activity)

        df.to_csv(path_name+"ECFP_counts.csv", index = False)

        return df

    def _to_numpy_arr(self, rdkit_fp, dtype = None):
        numpy_fp = np.zeros((0,), dtype = dtype)
        DataStructs.ConvertToNumpyArray(rdkit_fp, numpy_fp)
        return numpy_fp
        

class RDKit2D:
    #TODO
    #add normalized descriptors
    #add possibility to visualize correlations
    #add possibility to do feature selection based on colinearity
    def __init__(self, df):
        self.mols = [Chem.MolFromSmiles(s) for s in df.smiles]
        self.smiles = df.smiles.values
        self.activity = df.activity.values

    def compute_2d_rdkit(self, path_name = "./"):
        rdkit_2d_desc = []
        calc = MoleculeDescriptors.MolecularDescriptorCalculator(x[0] for x in Descriptors._descList)
        header = calc.GetDescriptorNames()

        for i in range(len(self.mols)):
            desc = calc.CalcDescriptors(self.mols[i])
            rdkit_2d_desc.append(desc)

        df = pd.DataFrame(rdkit_2d_desc, columns = header)

        df.insert(loc = 0, column = "smiles", value = self.smiles)
        df.insert(loc = 1, column = "activity", value = self.activity)

        df.to_csv(path_name+"_RDKit_2D.csv", index = False)

        return df