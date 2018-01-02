import os
import pandas as pd
import numpy as np

def load_train():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dataset = pd.read_json(dir_path + '/train.json')
    return dataset

def load_test():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dataset = pd.read_json(dir_path + '/test.json')
    return dataset

def normalize_image(img):
    data = np.array(img)
    min_val, max_val = np.min(data), np.max(data)
    return (data - min_val ) / (max_val - min_val)

def convert_to_normalized_array(band_series):
    return np.concatenate(band_series.apply(normalize_image)).reshape(-1, 75, 75)
