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
from googletrans import Translator, LANGUAGES
# translator = Translator()
import time

ad_activations_file = "result/vlguard/vlguard_test_adjust_activations_layer_9_multi_ne_0.5.json"
res = {}
idx1 = 0
with open(ad_activations_file, 'r') as file:
    for line in file:
        line_info = json.loads(line)
        image_file = line_info['image_file']
        response = line_info['response']
        prompt = line_info['prompt']
        translator = Translator()
        prompt_translate = translator.translate(prompt, src='en', dest='zh-cn').text
        answer_translate = translator.translate(response, src='en', dest='zh-cn').text
        res[image_file] = {"prompt": prompt, "trans_prompt": prompt_translate, "adjust_activation_response": response, "trans_adjust_activation_response": answer_translate}
        idx1 += 1
        if idx1 >= 50:
            break
        time.sleep(1)


origin_file = "result/vlguard/vlguard_test_origin_model.json"
idx2 = 0
with open(origin_file, 'r') as file:
    for line in file:
        line_info = json.loads(line)
        image_file = line_info['image_file']
        response = line_info['response']
        prompt = line_info['prompt']
        answer_translate = translator.translate(response, src='en', dest='zh-cn').text
        res[image_file].update({"origin_model_response": response, "trans_origin_model_response":answer_translate})
        idx2 += 1
        if idx2 >= 50:
            break
        time.sleep(1)


file_path = "result/vlguard/vlguard_combine.json"
with open(file_path, "w") as json_file:
    json.dump(res, json_file, indent=4)

qingli = 3








# file_path_1 = 'result/vlguard/vlguard_test_adjust_layer_9_multi_ne_1.3.bin'
# with open(file_path_1, "rb") as f:
#     responses_1 = pickle.load(f)
#
#
# for idx, response in responses_1.items():
#     id = response['id']
#     image_file = response['image_file']
#     prompt = response['prompt']
#     response_from_dataset = response['response_from_dataset']
#     response_adjust_model = response['response_adjust_model']
#
#
# file_path_2 = 'result/vlguard/vlguard_test_origin.bin'
# with open(file_path_2, "rb") as f:
#     responses_2 = pickle.load(f)
#
#
# for idx, response in responses_2.items():
#     id = response['id']
#     image_file = response['image_file']
#     prompt = response['prompt']
#     response_from_dataset = response['response_from_dataset']
#     response_adjust_model = response['response_adjust_model']
#
#
# file_path_3 = 'result/vlguard/vlguard_test_adjust_layer_9_multi_ne_0.5.bin'
# with open(file_path_3, "rb") as f:
#     responses_3 = pickle.load(f)
#
#
# for idx, response in responses_3.items():
#     id = response['id']
#     image_file = response['image_file']
#     prompt = response['prompt']
#     response_from_dataset = response['response_from_dataset']
#     response_adjust_model = response['response_adjust_model']
#
#
#
# file_path_4 = 'result/safebench/safebench_layer_9_multi_ne_0.5.bin'
# with open(file_path_4, "rb") as f:
#     responses_4 = pickle.load(f)
#
#
# for idx, response in responses_4.items():
#     id = response['id']
#     image_file = response['image_file']
#     prompt = response['prompt']
#
#
#
#
#
# file_path_5 = 'result/safebench/safebench_origian.bin'
# with open(file_path_5, "rb") as f:
#     responses_5 = pickle.load(f)
#
#
# for idx, response in responses_5.items():
#     id = response['id']
#     image_file = response['image_file']
#     prompt = response['prompt']
