import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pickle
import torch
import pandas as pd
import numpy as np

def load_data(data_dir = '../../data/splits.pkl', as_tensors = True):
    """
    Loads X_train, X_val, X_test, y_test from pickle files.
    Returns torch tensors.
    """
    with open(data_dir, 'rb') as f:
        splits = pickle.load(f)

    x_train = splits['X_train']
    x_val = splits['X_val']
    x_test = splits['X_test']
    y_test = splits['Y_test']

    if as_tensors:
        x_train = torch.tensor(x_train.values).float()
        x_val = torch.tensor(x_val.values).float()
        x_test = torch.tensor(x_test.values).float()
        y_test = torch.tensor(np.array(y_test)).long()

    return x_train, x_val, x_test, y_test
