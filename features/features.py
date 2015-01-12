__author__ = 'mihaileric'

import os,sys
"""Add root directory path"""
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

from util.DataStreamer import DataStreamer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from scipy import sparse


class IdentityFeatures(BaseEstimator):
    """Class for implementing identity indicator features for those
        provided by default in the data."""
    def __init__(self):
        pass

    def get_feature_names(self):
        return np.array(['ad_width', 'ad_height']) #try silly features for proof-of-concept

    def fit(self, data_points, y=None):
        pass

    def transform(self, data_points):
        features = []
        for d in data_points:
            features.append([float(d['C15']), float(d['C16'])])

        return np.array(features, dtype=np.float32)

    def fit_transform(self, data_points, y=None, **fit_params):
        #May implement this later if necessary
        pass

class IPFeatures(BaseEstimator):
    """Class for implementing features related to device IP values;
    will probably combine this with a broader feature set."""
    def __init__(self):
        pass

    def get_feature_names(self):
        return np.array(['device_ip']) #try silly features for proof-of-concept

    def fit(self, data_points, y=None):
        pass

    def transform(self, data_points):
        features = []
        for d in data_points:
            ip = int(d['device_ip'],16) #Convert hex string to integer
            features.append([ip])

        return np.array(features)

    def fit_transform(self, data_points, y=None, **fit_params):
        #May implement this later if necessary
        pass

class FeatureStacker(BaseEstimator):
    """Class for specifying a set of features from which to make a
        composite feature vector. List of transformer tuples
        (name, estimator) are passed to the constructor."""
    def __init__(self, transformer_list):
        self.transformers = transformer_list

    def get_feature_names(self):
        pass

    def fit(self, data_points, y=None):
        for name, estimator in self.transformers:
            estimator.fit(data_points)
        return self

    def transform(self, data_points):
        features = []
        for name, estimator in self.transformers:
            features.append(estimator.transform(data_points))
        sparse_features = [sparse.issparse(f) for f in features]
        if np.any(sparse_features):
            features = np.hstack(features).tocsr()
        else:
            features = np.hstack(features)
        return features