#!/usr/bin/env python
__author__ = 'mihaileric'
"""Module I will use to try techniques and lines of code out."""

import os,sys
"""Add root directory path"""
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

from features.features import IdentityFeatures, FeatureStacker
from util.DataStreamer import DataPoint

#custom assert function
def assert_equals(expected, got):
    assert expected == got, 'expected %s but got %s' %(expected, got)

#Some artificial examples to test features
exampleA = DataPoint({'C21': '79', 'device_ip': 'ddd2926e', 'site_id': '1fbe01fe', 'app_id': 'ecad2386', 'C19': '35', 'C18': '0', 'device_type': '1', 'id': '1000009418151094273', 'C17': '1722', 'click': '0', 'C15': '500', 'C14': '15706', 'C16': '503', 'device_conn_type': '2', 'C1': '1005', 'app_category': '07d7df22', 'site_category': '28905ebd', 'app_domain': '7801e8d9', 'site_domain': 'f3845767', 'banner_pos': '0', 'device_id': 'a99f214a', 'C20': '-1', 'hour': '14102100', 'device_model': '44956a24'})
exampleB = DataPoint({'C21': '79', 'device_ip': 'ddd2926e', 'site_id': '1fbe01fe', 'app_id': 'ecad2386', 'C19': '35', 'C18': '0', 'device_type': '1', 'id': '1000009418151094273', 'C17': '1722', 'click': '0', 'C15': '249', 'C14': '15706', 'C16': '501', 'device_conn_type': '2', 'C1': '1005', 'app_category': '07d7df22', 'site_category': '28905ebd', 'app_domain': '7801e8d9', 'site_domain': 'f3845767', 'banner_pos': '0', 'device_id': 'a99f214a', 'C20': '-1', 'hour': '14102100', 'device_model': '44956a24'})
500, 249,
503, 501
examples = [exampleA, exampleB] #concatenated features

ident_feat_extractor = IdentityFeatures()
ident_feat_extractor.fit(examples)
features = ident_feat_extractor.transform(examples)

assert_equals(str(500), features[0,0])
assert_equals(str(503), features[0,1])
assert_equals(str(249), features[1,0])
assert_equals(str(501), features[1,1])

print 'passed Identity Features test'
