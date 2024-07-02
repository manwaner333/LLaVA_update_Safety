from torch.utils.data import Dataset
import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
import requests
from PIL import Image
import math
import random
import numpy as np
import pickle
from io import BytesIO
from scipy.stats import entropy
# import spacy
from transformers import TextStreamer
import re
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

np.random.seed(42)
torch.manual_seed(42)


def extract_block_heads_activations(layers, activations_file="aa.json"):
    res = {}
    for i in range(0, 32):
        res[f"layer_{i}"] = []
    for layer in layers:
        print(f"layer: {layer}")
        layer_res = []
        with open(activations_file, "r") as f:
            for line in f:
                item = json.loads(line)
                id = item["id"]
                match = re.match(r'layer_(\d+)_head_(\d+)', id)
                if match:
                    layer_num = int(match.group(1))
                    head_num = int(match.group(2))
                    print(f'Layer: {layer_num}, Head: {head_num}')
                    if layer_num == layer:
                        layer_res.extend(item['block_self_attn_heads_activations'][0])
        res[f"layer_{layer}"].append(layer_res)
    return res

def generate_data(file1=None, file2=None):
    res = {}
    for i in range(0, 32):
        for j in range(0, 32):
            res[f"layer_{i}_head_{j}"] = {'activations': [], 'labels': []}

    if file1 is not None:
        with open(file1, "r") as file_1:  # train_file
            for line in file_1:
                item = json.loads(line)
                safe = item["safe"]
                block_activations = item['block_self_attn_heads_activations']
                for layer_num in range(0, 32):
                    layer_activations = block_activations[str(layer_num)][0]
                    for head_num in range(len(layer_activations)):
                        res[f"layer_{layer_num}_head_{head_num}"]['activations'].append(layer_activations[head_num])
                        res[f"layer_{layer_num}_head_{head_num}"]['labels'].append(1.0)

    if file2 is not None:
        with open(file2, "r") as file_2:  # train_file
            for line in file_2:
                item = json.loads(line)
                safe = item["safe"]
                block_activations = item['block_self_attn_heads_activations']
                for layer_num in range(0, 32):
                    layer_activations = block_activations[str(layer_num)][0]
                    for head_num in range(len(layer_activations)):
                        res[f"layer_{layer_num}_head_{head_num}"]['activations'].append(layer_activations[head_num])
                        res[f"layer_{layer_num}_head_{head_num}"]['labels'].append(0.0)
    return res

def extract_block_activations(layers, activations_file="aa.json"):
    res = {}
    for i in range(0, 32):
        res[f"layer_{i}"] = []

    for layer in layers:
        print(f"layer: {layer}")
        with open(activations_file, "r") as f:
            for line in f:
                item = json.loads(line)
                idx = item['idx']
                # print(idx)
                id = item['id']
                image_file = item["image_file"]
                safe = item["safe"]
                harmful_category = item['harmful_category']
                harmful_subcategory = item['harmful_subcategory']
                prompt = item["prompt"]
                block_self_attn_heads_activations = item['activations'][str(layer)][0]
                res[f"layer_{layer}"].append(block_self_attn_heads_activations)
    return res


if __name__ == "__main__":
    # step1: generate weight direction
    # pos_cases_path = 'playground/data/vlguard/answer_llava_v1.5_7b_vlguard_confrontational_cases_filter_pos.json'
    # neg_cases_path = 'playground/data/vlguard/answer_llava_v1.5_7b_vlguard_confrontational_cases_filter_neg.json'
    # save_path = "llava_v1.5_7b_pos_neg_cases"
    # start_layer = 0
    # end_layer = 31
    # layers = list(range(start_layer, end_layer + 1))
    # res = generate_data(pos_cases_path, neg_cases_path)
    #
    # out_file = "result/vlguard/llava_v1.5_7b_probe_layer_head_on_pos_nes_cases.json"
    # out_file_path = os.path.expanduser(out_file)
    # if os.path.exists(out_file_path):
    #     os.remove(out_file_path)
    #
    # with open(out_file_path, 'w') as out_file:
    #     for layer_num in range(0, 32):  # 0, 32
    #         for head_num in range(0, 32):  # 0, 32
    #             layer_head_data = res[f"layer_{layer_num}_head_{head_num}"]
    #             activations = layer_head_data['activations']
    #             labels = layer_head_data['labels']
    #             train_data, test_data, train_labels, test_labels = train_test_split(activations, labels, test_size=0.2,
    #                                                                                 random_state=42)
    #             X_train = np.array(train_data)
    #             Y_train = np.array(train_labels)
    #             X_test = np.array(test_data)
    #             Y_test = np.array(test_labels)
    #             seed = 42
    #             clf = LogisticRegression(random_state=seed, max_iter=5).fit(X_train, Y_train)
    #             Y_pred = clf.predict(X_train)
    #             Y_test_pred = clf.predict(X_test)
    #             acc = accuracy_score(Y_test, Y_test_pred)
    #             print(f"acc: {acc}")
    #             direction = clf.coef_
    #             direction = direction / np.linalg.norm(direction)
    #             proj_vals = activations @ direction.T
    #             proj_val_std = np.std(proj_vals)
    #             head_vector = proj_val_std * direction
    #
    #             json.dump({"id": f"layer_{layer_num}_head_{head_num}", "best_accuracy": acc,
    #                        'block_self_attn_heads_activations': head_vector.tolist()}, out_file)
    #             out_file.write('\n')

    # generate vectors
    save_path = "llava_v1.5_7b_vlguard_train_head_level_from_probe_weight_direction"
    if not os.path.exists(f"vectors/{save_path}"):
        os.mkdir(f"vectors/{save_path}")

    if not os.path.exists(f"clustering/{save_path}"):
        os.mkdir(f"clustering/{save_path}")

    start_layer = 0
    end_layer = 31
    layers = list(range(start_layer, end_layer + 1))

    start_head = 0
    end_head = 31
    heads = list(range(start_head, end_head + 1))



    answers_vlguard_pos_neg_file = "result/vlguard/llava_v1.5_7b_probe_layer_head_on_pos_nes_cases.json"
    res_heads_pos_neg = extract_block_heads_activations(layers, activations_file=answers_vlguard_pos_neg_file)


    for layer in layers:
        key = f"layer_{layer}"
        res_pos_neg_sub = res_heads_pos_neg[key]
        vec = torch.tensor(np.array(res_pos_neg_sub))
        torch.save(vec, f"vectors/{save_path}/vec_layer_{layer}.pt")



