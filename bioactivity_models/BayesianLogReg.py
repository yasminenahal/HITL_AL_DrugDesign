import numpy as np
import pandas as pd

from scipy.stats import bernoulli

class BayesianLogReg():
    def __init__(self, fit):
        """
        fit: fit object (stan_model["fit"])
        """
        la = fit.extract(permuted=True)
        self.trace_beta = pd.DataFrame(la["beta"])
        self.trace_alpha = pd.DataFrame(la["alpha"])

    def _compute_distributions(self, x):
        """
        x: list of molecular feature vectors
        """
        prob_dists = []
        # for each molecule to be predicted, we get a distribution of predicted labels from 
        # the different samples of the model parameters
        for n in range(len(x)):
            preds_per_mol = []
            for i in range(len(self.trace_beta)):
                beta_vec_i = self.trace_beta.iloc[i]
                if len(self.trace_alpha) == len(self.trace_beta):
                    alpha_i = self.trace_alpha.iloc[i].values[0]
                z_i = x[n,:].dot(beta_vec_i) + alpha_i
                p_i = self._inv_logit(z_i)
                label_i = bernoulli.rvs(p_i, size = 1).item()
                preds_per_mol.append(label_i)
            if len(preds_per_mol) == len(self.trace_beta) == len(self.trace_alpha):
                if len(np.unique(preds_per_mol)) > 1: # if the two classes are predicted
                    pos_labels = np.unique(preds_per_mol, return_counts = True)[1][1] # for label == 1
                    neg_labels = np.unique(preds_per_mol, return_counts = True)[1][0] # for label == 0
                    pos_percent = pos_labels / len(preds_per_mol)
                    neg_percent = neg_labels / len(preds_per_mol)
                else:
                    if np.unique(preds_per_mol, return_counts = True)[0].item() == 1:
                        pos_labels = np.unique(preds_per_mol, return_counts = True)[1].item()
                        pos_percent = pos_labels / len(preds_per_mol)
                        neg_percent = 0.0
                    if np.unique(preds_per_mol, return_counts = True)[0].item() == 0:
                        neg_labels = np.unique(preds_per_mol, return_counts = True)[1].item()
                        neg_percent = neg_labels / len(preds_per_mol)
                        pos_percent = 0.0

            prob_dists.append((neg_percent, pos_percent))
        
        return np.array(prob_dists)

    def _uncertainty(self, x):
        prob_dists = self._compute_distributions(x)
        prbslogs = prob_dists * np.log2(prob_dists)
        numerator = 0 - np.sum(prbslogs)
        denominator = np.log2(prob_dists.size)
        entropy = numerator / denominator
        
        return entropy