__author__ = 'mihaileric'

import os,sys
"""Add root directory path"""
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

import json
from numpy import mean, std, min, max
import cPickle as pickle
from collections import Counter
import logging
import scipy.sparse
import bz2
import numpy as np
from scipy.sparse import csr_matrix

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

def save_sparse_csr(filename, array):
    """
    Saves a sparse scipy matrix with custom representation in bz2 format.
    """
    outfile = bz2.BZ2File('../output/'+filename+'.me.bz2', mode='wb', compresslevel=9)
    outfile.write(str(array.shape[0]) + "," + str(array.shape[1]) + '\n') #Store dimensions of matrix

    #Store all non-zero entries along with their location (row,col)
    row_vals, col_vals = array.nonzero()
    logging.info("Saving sparse matrix of dimensions (%s, %s) to %s" %(array.shape[0], array.shape[1], filename+'.me.bz2'))
    for row, col in zip(row_vals.tolist(), col_vals.tolist()):
        outfile.write(str(row) + "," + str(col) + "," + str(array[row,col]))
    outfile.close()

def load_sparse_csr(filename):
    """Loads a sparse matrix stored in custom representation in bz2 compressed format."""
    infile = bz2.BZ2File('../output/'+filename+'.me.bz2', mode='rb')
    #Store matrix values in separate data array, row index, and column index array
    data = []
    row_ind = []
    col_ind = []
    row_size, col_size = infile.read().strip().split(',')

    for line in infile:
        row, col, val = line.strip().split(',')
        data += [int(val)]
        row_ind += [int(row)]
        col_ind += [int(col)]
    infile.close()

    logging.info('Loading sparse matrix of shape (%s, %s) from $s' %(row_size, col_size, filename+'.me.bz2'))
    data = np.array(data)
    row_ind = np.array(data)
    col_ind = np.array(data)

    return csr_matrix(data, (row_ind, col_ind), shape=(int(row_size), int(col_size)))




