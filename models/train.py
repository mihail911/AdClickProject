__author__ = 'mihaileric'

import os,sys
"""Add root directory path"""
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

import bz2
from models import build_logistic_regression_model, build_svm_model
from util.DataStreamer import DataPoint, DataStreamer
from features.features import FeatureStacker, IPFeatures, IdentityFeatures, SiteIDFeatures, TimeFeatures
from util.utilities import  load_sparse_csr, save_sparse_csr
from sklearn.metrics import f1_score
import numpy as np
import logging
import argparse

train_data_filename = "../util/subsampled_data_train.bz2"
test_data_filename = "../util/subsampled_data_test.bz2"
outfile_name = "feature_vector"

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='arguments for train/test')
parser.add_argument('--train_data_file', type=str, help='Data file to be used for training model')
parser.add_argument('--test_data_file', type=str, help='Data file to be used for testing model')
parser.add_argument('--features', type=str, nargs='+', help='Features to extract from data')
parser.add_argument('--model', type=str, help='Modelling algorithm to use')
args = parser.parse_args()

def generate_feature_vector(data_filename):
    """Load data file, which is a bz2 file, of training samples,
    generate corresponding feature vector, and write to disk."""
    data_points = [d for d in DataStreamer.load_bz2_file(data_filename)]
    labels = generate_labels_vector(data_points)
    features = [('identity', IdentityFeatures()), ('ip', IPFeatures()),
                ('site_id', SiteIDFeatures()), ('time', TimeFeatures())]
    stacker = FeatureStacker(features)
    feature_vector = stacker.fit_transform(data_points)

    #Write feature vector and labels vector to disk
    save_sparse_csr(outfile_name, feature_vector)
    save_sparse_csr('labels_vector', labels) #TODO: don't hard-code the name

def generate_labels_vector(data_points):
    """Generates labels for all data samples, provided as argument."""
    labels = []
    for d in data_points:
        labels.append([float(d['click'])])
    return np.array(labels, dtype=np.float32)

def train_model(train_data_filename, model_type=None):
    """Trains model of specified type by loading provided feature vector."""
    model = None
    if model_type == 'logistic_regression':
        model = build_logistic_regression_model(('identity', IdentityFeatures()),
                                         ('ip', IPFeatures()))
    elif model_type == 'svm':
        model = build_logistic_regression_model(('identity', IdentityFeatures()),
                                         ('ip', IPFeatures()))

    #TODO: handle cases for loading other models when I finish writing them
    #feature_vector = load_sparse_csr(outfile_name) #DO I REALLY NEED TO WRITE VECTORS TO DISK?
    #labels_vector = load_sparse_csr('labels_vector')
    data_points = [d for d in DataStreamer.load_bz2_file(train_data_filename)]
    labels = generate_labels_vector(data_points)

    logging.info("Training %s model" %(model_type))
    model.fit(data_points, labels)

    return model

def test_model(model, train_filename=None, test_filename=None):
    """Test model and report statistics."""
    data_points = [d for d in DataStreamer.load_bz2_file(train_filename)]
    true_output = np.array([float(d['click']) for d in data_points])
    #features = [('identity', IdentityFeatures()), ('ip', IPFeatures())]
    #feature_stacker = FeatureStacker(features)
    #transformed_features = feature_stacker.transform(data_points)
    logging.info("Testing model on %s containing %d data points." %(test_filename, len(data_points)))
    prediction = model.predict(data_points) #Prediction always returns vector of 1s
    f1 = f1_score(prediction, true_output)
    logging.info("Calculated f1 score for model: %f" %f1)


#generate_feature_vector(train_data_filename)
model = train_model(train_data_filename, 'logistic_regression') #Getting same f1 score wtf??
test_model(model, train_data_filename, test_data_filename)

