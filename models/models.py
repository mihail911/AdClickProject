__author__ = 'mihaileric'

import os, sys
"""Add root directory path"""
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

from sklearn.pipeline import Pipeline
from features.features import IdentityFeatures


def build_naive_bayes_model(*args):
    """Builds a basic naive bayes model using certain feature set.
    Returns as a pipeline with specified features."""

