__author__ = 'mihaileric'

import os,sys
"""Add root directory path"""
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

from util.DataStreamer import DataStreamer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class IdentityFeatures(BaseEstimator, TransformerMixin):
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
        #widths = [d.data['C15'] for d in data_points]
        #heights = [d.data['C16'] for d in data_points]
        for d in data_points:
            features.append([d.data['C15', d.data['C16']]])

        return np.array(features)

    def fit_transform(self, data_points, y=None, **fit_params):
        #May implement this later if necessary
        pass