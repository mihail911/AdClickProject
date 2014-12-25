__author__ = 'mihaileric'
import os, sys
import bz2, csv, logging, json

class DataPoint(object):
    """Object for representing a single data point."""
    def __init__(self, data):
        self.data = data

    @classmethod
    def extract_data(cls, line, fields):
        data = {}
        for index, field in enumerate(fields):
            data[field] = line[index]
        return DataPoint(data)

    def to_json(self):
        return json.dumps(self.data)

class DataStreamer(object):
    """Class for streaming data."""

    @classmethod
    def extract_feature_counts(cls, file_name):
        count = 0
        file = bz2.BZ2File(file_name, 'rb')
        reader = csv.reader(file)
        fields = reader.next()
        for line in reader:
            if count % 10000 == 0:
                logging.info("Processed %d examples", count)
            data_point = DataPoint.extract_data(line, fields)
            count += 1

    @classmethod
    def load_data_file(cls, file_name):
        file = bz2.BZ2File(file_name, 'rb')
        reader = csv.reader(file)
        fields = reader.next()
        for line in reader:
            data_point = DataPoint.extract_data(line, fields)
            yield data_point


