#!/usr/bin/env python
__author__ = 'mihaileric'

import sys, os
"""Add root directory path"""
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

import bz2, json, logging
import argparse
from DataStreamer import DataStreamer
from collections import Counter

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='splits data bz2 into train/test files with a '
                                             'specified number of examples')
parser.add_argument('data_file', help='data file to be split')
parser.add_argument('out_file', help='prefix for out files where data to be stored')
parser.add_argument('--num_examples', type=int, help='number of examples desired for training', default=100000)
parser.add_argument('--test_frac', type=float, help='fraction of data to include for testing', default=0.2)
args = parser.parse_args()


num_train_needed = args.num_examples
num_test_needed = args.num_examples * args.test_frac

train_out = bz2.BZ2File(args.out_file+'_train.bz2', mode='wb', compresslevel=9)
test_out = bz2.BZ2File(args.out_file+'_test.bz2', mode='wb', compresslevel=9)

initial_count = dict(zip((0,1), [0]*2))
train_counts = Counter(initial_count)
test_counts = Counter(initial_count)

print 'len dicts', len(train_counts), train_counts, train_counts.values()
train_so_far = 0
test_so_far = 0
count = 0

for data_point in DataStreamer.load_data_file(args.data_file):
    if count % 10000 == 0:
        logging.info('Processed %d examples', count)

    #Ensure equal number of positive and negative clicks
    click_result = data_point.data['click']
    more_train_needed = [c for c in train_counts.values() if c < num_train_needed]
    can_use_for_train = train_counts[int(click_result)] < num_train_needed

    more_test_needed = [c for c in test_counts.values() if c < num_test_needed]
    can_use_for_test = test_counts[int(click_result)] < num_test_needed

    #Don't need any more examples for train/test
    if not (more_test_needed or more_train_needed):
        break

    if more_train_needed and can_use_for_train:
        train_counts[int(click_result)] += 1
        train_out.write(data_point.to_json() + '\n')
        train_so_far += 1
        #print 'train_so_far %d' %train_so_far

    if more_test_needed and can_use_for_test:
        test_counts[int(click_result)] += 1
        test_out.write(data_point.to_json() + '\n')
        test_so_far += 1
        #print 'test_so_far %d' %test_so_far

    count += 1

train_out.close()
test_out.close()

logging.info('Dumped %d training examples, %d test examples, %d in total' %(train_so_far, test_so_far, train_so_far + test_so_far))
