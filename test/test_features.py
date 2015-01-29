#!/usr/bin/env python
__author__ = 'mihaileric'
"""Module to test whether feature extractors work properly."""

import os,sys
"""Add root directory path"""
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

from features.features import IdentityFeatures, IPFeatures, FeatureStacker, TimeFeatures, SiteIDFeatures
from util.DataStreamer import DataPoint

#custom assert function
def assert_equals(expected, got):
    assert expected == got, 'expected %s but got %s' %(expected, got)

#Some artificial examples to test features
exampleA = {'C21': '79', 'device_ip': '10', 'site_id': '1fbe01fe', 'app_id': 'ecad2386', 'C19': '35', 'C18': '0', 'device_type': '1', 'id': '1000009418151094273', 'C17': '1722', 'click': '0', 'C15': '500', 'C14': '15706', 'C16': '503', 'device_conn_type': '2', 'C1': '1005', 'app_category': '07d7df22', 'site_category': '28905ebd', 'app_domain': '7801e8d9', 'site_domain': 'f3845767', 'banner_pos': '0', 'device_id': 'a99f214a', 'C20': '-1', 'hour': '14102100', 'device_model': '44956a24'}
exampleB = {'C21': '79', 'device_ip': '12', 'site_id': '1fbe01fe', 'app_id': 'ecad2386', 'C19': '35', 'C18': '0', 'device_type': '1', 'id': '1000009418151094273', 'C17': '1722', 'click': '0', 'C15': '249', 'C14': '15706', 'C16': '501', 'device_conn_type': '2', 'C1': '1005', 'app_category': '07d7df22', 'site_category': '28905ebd', 'app_domain': '7801e8d9', 'site_domain': 'f3845767', 'banner_pos': '0', 'device_id': 'a99f214a', 'C20': '-1', 'hour': '14101340', 'device_model': '44956a24'}

examples = [exampleA, exampleB] #concatenated features

ident_feat_extractor = IdentityFeatures()
ident_feat_extractor.fit(examples)
features = ident_feat_extractor.transform(examples)

assert_equals(500.0, features[0,0])
assert_equals(503.0, features[0,1])
assert_equals(249.0, features[1,0])
assert_equals(501.0, features[1,1])

print 'passed Identity Features test'

time_feat_extractor = TimeFeatures()
time_feat_extractor.fit(examples)
features = time_feat_extractor.transform(examples)

assert_equals(21.0, features[0,0])
assert_equals(0.0, features[0,1])
assert_equals(13.0, features[1,0])
assert_equals(40.0, features[1,1])

print 'passed Time Features test'

site_id_feat_extractor = SiteIDFeatures()
site_id_feat_extractor.fit(examples)
features = site_id_feat_extractor.transform(examples)

print 'Site ID Features', features
print 'passed Site ID Features test'

ip_feat_extractor = IPFeatures()
stacked_feat_extractor = FeatureStacker([('identity', ident_feat_extractor),
                                         ('ip', ip_feat_extractor)])
stacked_feat_extractor.fit(examples)
features = stacked_feat_extractor.transform(examples)

assert_equals(500.0, features[0,0])
assert_equals(503.0, features[0,1])
assert_equals(249.0, features[1,0])
assert_equals(501.0, features[1,1])
assert_equals(16.0, features[0,2])
assert_equals(18.0, features[1,2])

print 'passed Feature Stacker test'

