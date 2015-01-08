__author__ = 'mihaileric'

import os,sys
"""Add root directory path"""
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

import bz2
from models import build_logistic_regression_model
from util.DataStreamer import DataPoint, DataStreamer
from features.features import FeatureStacker, IPFeatures, IdentityFeatures

filename = "../util.subsampled_data_train.bz2"

#Script for training models
def load_data(filename):
    """Load data file, which is a bz2 file, of training samples."""
    data_points = [d for d in DataStreamer.load_bz2_file(filename)]
    features = [('identity', IdentityFeatures()), ('ip', IPFeatures())]
    stacker = FeatureStacker(features)
    stacker.fit(data_points)
    feature_vector = stacker.transform(data_points)
    #TODO: write feature_vector to disk