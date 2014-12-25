__author__ = 'mihaileric'
import json
from numpy import mean, std, min, max
import cPickle as pickle
from collections import Counter

import os,sys
"""Add root directory path"""
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

all_counters = ('C1_counts', 'C15_counts',
'C16_counts', 'C17_counts', 'C18_counts', 'C19_counts', 'C20_counts',
'C21_counts', 'site_id_counts', 'site_domain_counts', 'site_category_counts',
'app_id_counts', 'app_domain_counts', 'app_category_counts', 'device_id_counts',
'device_model_counts', 'device_type_counts', 'device_conn_type_counts')

def generate_counter_mapping():
    """Generates a super counter that is mapping from
    feature description -> counter for that feature."""
    counter_mapping = {}
    for counter_name in all_counters:
        with open('featureCounts/' + counter_name, 'r') as f:
            counter_mapping[counter_name] = json.load(f)

    #Dump counter mapping to disk
    with open('counter_mapping.json', 'w') as f:
        json.dump(counter_mapping, f)
    return counter_mapping

def dump_counter_names():
    """Dump all the counter names as a pickle file."""
    with open('counter_names.pkl', 'w') as f:
        pickle.dump(all_counters,f)

def feature_count_stats():
    try:
        f = open('counter_mapping.json', 'r')
        mapping = json.load(f)
    except:
        print 'Failed to open \"counter_mapping\" '
    for feature_counter in all_counters:
        feat_count = mapping[feature_counter]
        print 'Feature %s statistics: Mean: %s, Std: %s, Min: %s, Max: %s, Num Elems: %s' \
        %(feature_counter, mean(feat_count.values()), std(feat_count.values()),
        min(feat_count.values()), max(feat_count.values()), len(feat_count.values()))


