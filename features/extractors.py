__author__ = 'mihaileric'

import os,sys
"""Add root directory path"""
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

from util.DataStreamer import DataStreamer
from sklearn.base import BaseEstimator, TransformerMixin


class IdentityFeatures(BaseEstimator, TransformerMixin):
    """Class for implementing identity indicator features for those
        provided by default in the data."""
    def __init__(self):
        pass

    def fit(self, data_points, y=None):
        pass

    def transform(self, data_points):
        pass

    def fit_transform(self, data_points, y=None, **fit_params):
        pass