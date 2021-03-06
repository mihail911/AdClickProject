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

default_params = {
    'input':'content', 'encoding':'utf-8', 'charset':None, 'decode_error':'strict', 'charset_error':None, 'strip_accents':None, 'lowercase':True, 'preprocessor':None, 'tokenizer':None, 'stop_words':None, 'token_pattern':'(?u)\b\w\w+\b', 'ngram_range':(1, 1), 'analyzer':'word', 'max_df':1.0, 'min_df':1, 'max_features':None, 'vocabulary':None, 'binary':False, 'dtype':np.int64
}

def update_default(new_values):
    """Updates dict of init parameters."""
    original = default_params.copy()
    for key in default_params:
        if key in new_values:
            original[key] = new_values[key]
    return original

class IdentityFeatures(BaseEstimator):
    """Class for implementing identity indicator features for those
        provided by default in the data."""
    def __init__(self):
        pass

    def get_feature_names(self):
        return np.array(['ad_width', 'ad_height']) #try silly features for proof-of-concept

    def fit(self, data_points, y=None):
        pass

    def docs_from_data_points(self, data_points=None):
        docs = []
        for d in data_points:
            docs.append([float(d['C15']), float(d['C16'])])
        return docs

    def transform(self, data_points):
        docs = self.docs_from_data_points(data_points)
        return np.array(docs, dtype=np.float32)

    def fit_transform(self, data_points, y=None, **fit_params):
        docs = self.docs_from_data_points(data_points)
        return np.array(docs, dtype=np.float32)

class AbstractDocsVectorizer(CountVectorizer):
    def __init__(self, **init_params):
        super(AbstractDocsVectorizer, self).__init__(**init_params)

    def fit(self, raw_documents, y=None, **init_params):
        return super(AbstractDocsVectorizer, self).fit(self.docs_from_data_points(raw_documents), y, **init_params)

    def transform(self, raw_documents, y=None, **init_params):
        return super(AbstractDocsVectorizer, self).transform(self.docs_from_data_points(raw_documents), y, **init_params)

    def fit_transform(self, raw_documents, y=None, **init_params):
        return super(AbstractDocsVectorizer, self).fit_transform(self.docs_from_data_points(raw_documents), y, **init_params)

class SiteIDFeatures(AbstractDocsVectorizer):
    """Class that learns a vocabulary on all encountered site IDs."""
    def __init__(self,
                 input=u'content',
                 encoding=u'utf-8',
                 charset=None,
                 decode_error=u'strict',
                 charset_error=None,
                 strip_accents=None,
                 lowercase=True,
                 preprocessor=None,
                 tokenizer=None,
                 stop_words=None,
                 ngram_range=(1, 1),
                 analyzer=u'word',
                 max_df=1.0,
                 min_df=0,
                 token_pattern=u'(?u)\\b\\w\\w+\\b',
                 max_features=None,
                 vocabulary=None,
                 binary=False,
                 dtype=np.int64):
        super(SiteIDFeatures, self).__init__(**update_default(locals())) #TODO: Write get_params

    def get_feature_names(self):
        return np.array(['site_id'])

    def docs_from_data_points(self, data_points=None):
        docs = []
        for d in data_points:
            docs.append(d['site_id']) #Adding single 'entity' instead of ['entity']
        return docs

    # def fit(self, data_points, y=None):
    #     features = self.docs_from_data_points(data_points)
    #     super(SiteIDFeatures, self).fit(features, y)


    # def transform(self, data_points):
    #     features = self.docs_from_data_points(data_points)
    #     return super(SiteIDFeatures, self).transform(features)

    # def fit_transform(self, data_points, y=None, **fit_params):
    #     if type(data_points) == list:
    #         print 'hi'
    #     else:
    #         print 'no'
    #     features = self.docs_from_data_points(data_points)
    #     return super(SiteIDFeatures, self).fit_transform(features, y, **fit_params)


class IPFeatures(BaseEstimator):
    """Class for implementing features related to device IP values;
    will probably combine this with a broader feature set."""
    def __init__(self):
        pass

    def get_feature_names(self):
        return np.array(['device_ip']) #try silly features for proof-of-concept

    def docs_from_data_points(self, data_points=None):
        docs = []
        for d in data_points:
            ip = int(d['device_ip'],16) % 1000
            docs.append([ip])
        return docs

    def fit(self, data_points, y=None):
        pass

    def transform(self, data_points):
        docs = self.docs_from_data_points(data_points)
        return np.array(docs)

    def fit_transform(self, data_points, y=None, **fit_params):
        docs = self.docs_from_data_points(data_points)
        return np.array(docs)

class TimeFeatures(BaseEstimator):
    """Features making use of the timestamp of the ad. Proper processing is done
    in order to convert timestamp to useful format."""
    def __init__(self):
        pass

    def get_feature_names(self):
        return np.array(['timestamp_info'])

    @classmethod
    def get_day_hour(cls, timestamp):
        """Returns day and hour of timestamp as an array."""
        return np.array([float(timestamp[4:6]), float(timestamp[6:8])])

    def docs_from_data_points(self, data_points=None):
        docs = []
        for d in data_points:
            docs.append(TimeFeatures.get_day_hour(d['hour']))
        return docs

    def fit(self, data_points, y=None):
        pass

    def transform(self, data_points):
        docs = self.docs_from_data_points(data_points)
        return np.array(docs)

    def fit_transform(self, data_points, y=None, **fit_params):
        docs = self.docs_from_data_points(data_points)
        return np.array(docs)

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

    def fit_transform(self, data_points, y=None, **fit_params):
        features = []
        for name, estimator in self.transformers:
            features.append(estimator.fit_transform(data_points))
        sparse_features = [sparse.issparse(f) for f in features]
        if np.any(sparse_features):
            features = sparse.hstack(features).tocsr()
        else:
            features = np.hstack(features)
        return features