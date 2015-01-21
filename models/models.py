__author__ = 'mihaileric'

import os, sys
"""Add root directory path"""
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

from sklearn.pipeline import Pipeline
from features.features import IdentityFeatures, IPFeatures, FeatureStacker
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def build_logistic_regression_model(*args):
    """Builds a logistic regression model using certain feature set, specified
    via 'args'. Returns as a pipeline with specified features."""
    model = LogisticRegression() #TODO: tune hyperparameters for model

    ###Maybe apply other data transforms (e.g. TF-IDF, K-best, PCA, etc.)

    features = FeatureStacker([feat for feat in args]) #make list of features to be applied
    pipeline = Pipeline([('feat', features), ('log_reg', model)])
    return pipeline

def build_svm_model(*args):
    """Builds an svm model using certain feature set, specified via
    'args'. Returns as a pipeline with specified features. Will hypertune the
    choice of kernel parameter."""
    model = SVC()
    features = FeatureStacker([feat for feat in args]) #make list of features to be applied
    pipeline = Pipeline([('feat', features), ('svm', model)])
    return pipeline


