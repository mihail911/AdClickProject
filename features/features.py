__author__ = 'mihaileric'

import os,sys
"""Add root directory path"""
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

from util.DataStreamer import DataStreamer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
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
            if d['click']: #is an ad
                pass
                #print 'features for click: ', d['C15'], ' ', d['C16']

        return np.array(features, dtype=np.float32)

    def fit_transform(self, data_points, y=None, **fit_params):
        #May implement this later if necessary
        pass

class SiteIDFeatures(CountVectorizer):
    """Class that learns a vocabulary on all encountered site IDs."""
    def __init__(self):
        pass

    def get_feature_names(self):
        return np.array(['site_id'])

    def convert_features(self, data_points=None):
        features = []
        for d in data_points:
            features.append([float(d['site_id'])])
        return features

    def fit(self, data_points, y=None):
        features = []
        for d in data_points:
            features.append([float(d['site_id'])])
        super(SiteIDFeatures, self).fit(features, y)


    def transform(self, data_points):
        features = SiteIDFeatures.convert_features(data_points)
        return super(SiteIDFeatures, self).transform(features)

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
            ip = int(d['device_ip'],16) % 1000 #Convert hex string to integer
            features.append([ip])

        return np.array(features)

    def fit_transform(self, data_points, y=None, **fit_params):
        #May implement this later if necessary
        pass

class TimeFeatures(BaseEstimator):
    """Features making use of the timestamp of the ad. Proper processing is done
    in order to convert timestamp to useful format."""
    def __init__(self):
        pass

    def get_feature_names(self):
        return np.array(['timestamp_info'])

    @staticmethod
    def get_day_hour(timestamp):
        """Returns day and hour of timestamp as an array."""
        return np.array([float(timestamp[4:6]), float(timestamp[6:8])])

    def fit(self, data_points, y=None):
        pass

    def transform(self, data_points):
        features = []
        for d in data_points:
            features.append(TimeFeatures.get_day_hour(d['hour']))
        return np.array(features)

    def fit_transform(self, data_points, y=None, **fit_params):
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