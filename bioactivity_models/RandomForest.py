import pickle

import numpy as np
from scripts.acquisition import vote_entropy_qbc
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

class RandomForestClf(RandomForestClassifier):
    def __init__(self, fit):
        super(RandomForestClf, self).__init__()
        self.fit = fit

    def _reinitialize_regressor(self, x, y, sample_weight = None, save_to_path = None):
        rf = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
                      max_features='auto', max_leaf_nodes=None,
                      min_impurity_decrease=0.0, min_impurity_split=None,
                      min_samples_leaf=1, min_samples_split=2,
                      min_weight_fraction_leaf=0.0, n_estimators=50,
                      n_jobs=None, oob_score=False, random_state=None,
                      verbose=0, warm_start=False)
        rf.fit(x, y, sample_weight = sample_weight)
        if save_to_path is not None:
            pickle.dump(rf, open(save_to_path, 'wb'))

    def _reinitialize_classifier(self, x, y, sample_weight = None, save_to_path = None):
        rf = RandomForestClassifier(n_estimators = 300, class_weight={0: 1, 1: 2})
        rf.fit(x, y, sample_weight = sample_weight)

        if save_to_path is not None:
            pickle.dump(rf, open(save_to_path, 'wb'))

    def _qbc(self, x):
        prop_votes = self.fit.predict_proba(x)
        vote_entropy = []

        for i in range(len(prop_votes)):
            vote_entropy.append(
                - np.sum(
                    [
                        prop_votes[i,0] * np.log2(prop_votes[i,0]), 
                        prop_votes[i,1] * np.log2(prop_votes[i,1])
                    ]
                )
            )
        return vote_entropy

class RandomForestReg(RandomForestRegressor):
    def __init__(self, fit):
        super(RandomForestReg, self).__init__()
        self.fit = fit

    def _reinitialize_regressor(self, x, y, sample_weight = None, save_to_path = None):
        rf = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
                      max_features='auto', max_leaf_nodes=None,
                      min_impurity_decrease=0.0, min_impurity_split=None,
                      min_samples_leaf=1, min_samples_split=2,
                      min_weight_fraction_leaf=0.0, n_estimators=50,
                      n_jobs=None, oob_score=False, random_state=None,
                      verbose=0, warm_start=False)
        rf.fit(x, y, sample_weight = sample_weight)
        if save_to_path is not None:
            pickle.dump(rf, open(save_to_path, 'wb'))

    def _uncertainty(self, x):
        individual_trees = self.fit.estimators_
        subEstimates = np.array(
            [
                tree.predict(np.stack(x))
                for tree in individual_trees
            ]
        )
        return np.std(subEstimates, axis=0)
