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

np.random.seed(42)
torch.manual_seed(42)


# def extract_block_heads_activations(layers, heads, activations_file="aa.json"):
#     res = {}
#     for i in range(0, 32):
#         for j in range(0, 32):
#             res[f"layer_{i}_head_{j}"] = []
#
#     with open(activations_file, "r") as f:
#         for line in f:
#             item = json.loads(line)
#             idx = item['idx']
#             print(idx)
#             id = item['id']
#             image_file = item["image_file"]
#             safe = item["safe"]
#             harmful_category = item['harmful_category']
#             harmful_subcategory = item['harmful_subcategory']
#             prompt = item["prompt"]
#             for layer in layers:
#                 block_self_attn_heads_activations = item["block_activations"][str(layer)][0]
#                 for head in heads:
#                     res[f"layer_{layer}_head_{head}"].append(block_self_attn_heads_activations[head])
#     return res


def extract_block_heads_activations(layers, activations_file="aa.json"):
    res = {}
    for i in range(0, 32):
        res[f"layer_{i}"] = []
    for layer in layers:
        print(f"layer: {layer}")
        with open(activations_file, "r") as f:
            for line in f:
                item = json.loads(line)
                idx = item['idx']
                id = item['id']
                image_file = item["image_file"]
                safe = item["safe"]
                harmful_category = item['harmful_category']
                harmful_subcategory = item['harmful_subcategory']
                prompt = item["prompt"]
                block_self_attn_heads_activations = item['block_self_attn_heads_activations'][str(layer)][0]
                layer_res = []
                for head in block_self_attn_heads_activations:
                    layer_res.extend(head)
                res[f"layer_{layer}"].append(layer_res)
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
    save_path = "llava_v1.5_13b_vlguard_train_layer_level"
    # save_path = "llava_v1.5_7b_vlguard_train_head_level"

    if not os.path.exists(f"vectors/{save_path}"):
        os.mkdir(f"vectors/{save_path}")

    if not os.path.exists(f"clustering/{save_path}"):
        os.mkdir(f"clustering/{save_path}")

    start_layer = 0
    end_layer = 2
    layers = list(range(start_layer, end_layer + 1))

    start_head = 0
    end_head = 31
    heads = list(range(start_head, end_head + 1))

    # answers_vlguard_pos_file = "playground/data/vlguard/answer_llava_v1.5_7b_vlguard_confrontational_cases_filter_pos.json"
    # answers_vlguard_neg_file = "playground/data/vlguard/answer_llava_v1.5_7b_vlguard_confrontational_cases_filter_neg.json"
    answers_vlguard_pos_file = "playground/data/vlguard/answer_llava_v1.5_13b_vlguard_confrontational_cases_filter_pos.json"
    answers_vlguard_neg_file = "playground/data/vlguard/answer_llava_v1.5_13b_vlguard_confrontational_cases_filter_neg.json"
    # layer level
    res_pos = extract_block_activations(layers, activations_file=answers_vlguard_pos_file)
    res_neg = extract_block_activations(layers, activations_file=answers_vlguard_neg_file)
    # head level
    # res_heads_pos = extract_block_heads_activations(layers, activations_file=answers_vlguard_pos_file)
    # res_heads_neg = extract_block_heads_activations(layers, activations_file=answers_vlguard_neg_file)

    for layer in layers:
        key = f"layer_{layer}"
        # layer level
        res_pos_sub = res_pos[key]
        res_neg_sub = res_neg[key]
        # head level
        # res_pos_sub = res_heads_pos[key]
        # res_neg_sub = res_heads_neg[key]
        positive = torch.tensor(np.array(res_pos_sub))
        negative = torch.tensor(np.array(res_neg_sub))
        vec = (positive - negative).mean(dim=0)
        torch.save(vec, f"vectors/{save_path}/vec_layer_{layer}.pt")


    # for layer in layers:
    #     for head in heads:
    #         key = f"layer_{layer}_head_{head}"
    #         res_pos_sub = res_pos[key]
    #         res_neg_sub = res_neg[key]
    #         positive = torch.tensor(np.array(res_pos_sub))
    #         negative = 0.5 * torch.tensor(np.array(res_neg_sub))
    #         vec = (positive - negative).mean(dim=0)
    #         torch.save(vec, f"vectors/{save_path}/vec_layer_{layer}_head_{head}.pt")



