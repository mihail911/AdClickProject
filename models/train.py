__author__ = 'mihaileric'

import os,sys
"""Add root directory path"""
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

import bz2
from models import build_logistic_regression_model
from util.DataStreamer import DataPoint, DataStreamer
from features.features import FeatureStacker, IPFeatures, IdentityFeatures
from util.utilities import  load_sparse_csr, save_sparse_csr
import numpy as np
import logging

data_filename = "../util.subsampled_data_train.bz2"
outfile_name = "feature_vector"

#TODO: Do argument parsing for getting what features user wants in feature vector generation

def generate_feature_vector(data_filename):
    """Load data file, which is a bz2 file, of training samples,
    generate corresponding feature vector, and write to disk."""
    data_points = [d for d in DataStreamer.load_bz2_file(data_filename)]
    labels = generate_labels_vector(data_points)
    features = [('identity', IdentityFeatures()), ('ip', IPFeatures())]
    stacker = FeatureStacker(features)
    stacker.fit(data_points)
    feature_vector = stacker.transform(data_points)

    #Write feature vector and labels vector to disk
    save_sparse_csr(outfile_name, feature_vector)
    save_sparse_csr('labels_vector', labels) #TODO: don't hard-code the name

def generate_labels_vector(data_points):
    """Generates labels for all data samples, provided as argument."""
    labels = []
    for d in data_points:
        labels.append(float(d.data['click']))
    return np.array(labels, dtype=np.float32)

def train_model(feature_vec_filename=outfile_name, model_type='logistic_regression'):
    """Trains model of specified type by loading provided feature vector."""
    model = None
    if model_type == 'logistic_regression':
        model = build_logistic_regression_model()
    #TODO: handle cases for loading other models when I finish writing them
    feature_vector = load_sparse_csr(outfile_name)
    labels_vector = load_sparse_csr('labels_vector')

    logging.info("Training %s model" %(model_type))
    model.fit(feature_vector, labels_vector)

    return model

def test_model():
    """Test model and report statistics."""


