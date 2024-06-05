import pickle
import numpy as np
from scipy.stats import entropy
from tqdm import tqdm
import os
import json
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
import random
import pandas as pd
import csv
import joblib
from sklearn.metrics import roc_auc_score
file_path = 'result/m_hal/vlguard_test_adjust_layer_9_multi_ne_1.3.bin'
with open(file_path, "rb") as f:
    responses = pickle.load(f)


for idx, response in responses.items():
    id = response['id']
    image_file = response['image_file']
    prompt = response['prompt']
    response_from_dataset = response['response_from_dataset']
    response_adjust_model = response['response_adjust_model']


