"""Data imputation - KNN Imputation using different distance methods"""
import numpy as np
from scipy.special import comb
from sklearn.impute import KNNImputer


class PenaltyImputer:
    def __init__(self, ratio, data, k):
        # TODO check axis in the apply
        penalties = data.apply(lambda col: PenaltyImputer.__get_null_penalty_per_col(col, ratio)).to_numpy()
        self.imp = KNNImputer(missing_values=np.nan, n_neighbors=k, weights='distance', metric=lambda X, Y, **kwds: PenaltyImputer.__knn_dist(X, Y, penalties=penalties, **kwds))

    @staticmethod
    def __get_null_penalty_per_col(col, ratio):
        n = max(2, round(ratio * col.count()))
        denominator = comb(n, 2)  # n choose 2
        step = 1 / n
        samples = np.array([np.nanquantile(col, min([q, 1])) for q in np.arange(step, 1 + step, step)])
        penalty = sum([sum((i - samples[np.where(samples > i)]) ** 2) for i in samples[:-1]]) / denominator
        return penalty

    @staticmethod
    def __knn_dist(X, Y, missing_values=np.nan, **kwds):
        penalties = kwds["penalties"]
        vec = (X - Y) ** 2
        vec[np.where(np.isnan(vec))] = penalties[np.where(np.isnan(vec))]
        score = np.sqrt(np.sum(vec))
        return score

    def fit(self, X, y=None):
        return self.imp.fit(X)

    def fit_transform(self, X, y=None, **fit_params):
        return self.imp.fit_transform(X,  y, **fit_params)

    def transform(self, X):
        return self.imp.transform(X)


class WeightedImputer:
    def __init__(self, k):
        self.imp = KNNImputer(missing_values=np.nan, n_neighbors=k, weights='distance',
                         metric=lambda X, Y, **kwds: WeightedImputer.__weighted_distance(X, Y, **kwds))

    @staticmethod
    def __weighted_distance(X, Y, missing_values=np.nan, **kwds):
        vec = (X - Y) ** 2
        count = np.count_nonzero(~np.isnan(vec))
        score = np.sqrt(np.nansum(vec)) / count
        return score

    def fit(self, X, y=None):
        return self.imp.fit(X)

    def fit_transform(self, X, y=None, **fit_params):
        return self.imp.fit_transform(X,  y, **fit_params)

    def transform(self, X):
        return self.imp.transform(X)
