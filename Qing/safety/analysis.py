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
file_path_1 = 'result/vlguard/vlguard_test_adjust_layer_9_multi_ne_1.3.bin'
with open(file_path_1, "rb") as f:
    responses_1 = pickle.load(f)


for idx, response in responses_1.items():
    id = response['id']
    image_file = response['image_file']
    prompt = response['prompt']
    response_from_dataset = response['response_from_dataset']
    response_adjust_model = response['response_adjust_model']


file_path_2 = 'result/vlguard/vlguard_test_origin.bin'
with open(file_path_2, "rb") as f:
    responses_2 = pickle.load(f)


for idx, response in responses_2.items():
    id = response['id']
    image_file = response['image_file']
    prompt = response['prompt']
    response_from_dataset = response['response_from_dataset']
    response_adjust_model = response['response_adjust_model']


file_path_3 = 'result/vlguard/vlguard_test_adjust_layer_9_multi_ne_0.5.bin'
with open(file_path_3, "rb") as f:
    responses_3 = pickle.load(f)


for idx, response in responses_3.items():
    id = response['id']
    image_file = response['image_file']
    prompt = response['prompt']
    response_from_dataset = response['response_from_dataset']
    response_adjust_model = response['response_adjust_model']



file_path_4 = 'result/safebench/safebench_layer_9_multi_ne_0.5.bin'
with open(file_path_4, "rb") as f:
    responses_4 = pickle.load(f)


for idx, response in responses_4.items():
    id = response['id']
    image_file = response['image_file']
    prompt = response['prompt']





file_path_5 = 'result/safebench/safebench_origian.bin'
with open(file_path_5, "rb") as f:
    responses_5 = pickle.load(f)


for idx, response in responses_5.items():
    id = response['id']
    image_file = response['image_file']
    prompt = response['prompt']
