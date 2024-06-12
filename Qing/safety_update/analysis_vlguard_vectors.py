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


def save_activation_projection_tsne(
    activations1,
    activations2,
    activations3,
    activations4,
    activations5,
    fname,
    title,
    label1="Unsafe",
    label2="Safe_unsafe",
    label3="Safe_safe",
    label4="safebench_with",
    label5="safebench_without"
):
    """
    activations1: n_samples x vector dim tensor
    activations2: n_samples x vector dim tensor

    projects to n_samples x 2 dim tensor using t-SNE (over the full dataset of both activations 1 and 2) and saves visualization.
    Colors projected activations1 as blue and projected activations2 as red.
    """
    plt.clf()
    activations_np = np.concatenate((np.array(activations1), np.array(activations2), np.array(activations3), np.array(activations4), np.array(activations5)), axis=0).astype(np.float16)

    # t-SNE transformation
    tsne = TSNE(n_components=2)
    projected_activations = tsne.fit_transform(np.array(activations_np))
    len1 = np.array(activations1).shape[0]
    len2 = np.array(activations2).shape[0]
    len3 = np.array(activations3).shape[0]
    len4 = np.array(activations4).shape[0]
    len5 = np.array(activations5).shape[0]
    # Splitting back into activations1 and activations2
    activations1_projected = projected_activations[0:len1, :]
    activations2_projected = projected_activations[len1:len1+len2, :]
    activations3_projected = projected_activations[len1+len2:len1+len2+len3, :]
    activations4_projected = projected_activations[len1+len2+len3:len1+len2+len3+len4, :]
    activations5_projected = projected_activations[len1 + len2 + len3 + len4:, :]

    # Visualization
    for x, y in activations1_projected:
        plt.scatter(x, y, color="blue", marker="o", alpha=0.4)

    for x, y in activations2_projected:
        plt.scatter(x, y, color="red", marker="o", alpha=0.4)

    for x, y in activations3_projected:
        plt.scatter(x, y, color="green", marker="o", alpha=0.4)

    for x, y in activations4_projected:
        plt.scatter(x, y, color="orange", marker="o", alpha=0.4)

    for x, y in activations5_projected:
        plt.scatter(x, y, color="pink", marker="o", alpha=0.4)

    # Adding the legend
    scatter1 = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="blue",
        markersize=10,
        label=label1,
    )
    scatter2 = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="red",
        markersize=10,
        label=label2,
    )

    scatter3 = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="green",
        markersize=10,
        label=label3,
    )

    scatter4 = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="orange",
        markersize=10,
        label=label4,
    )

    scatter5 = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="pink",
        markersize=10,
        label=label5,
    )

    # plt.legend(handles=[scatter1, scatter2, scatter3, scatter4, scatter5], loc='upper center', bbox_to_anchor=(0.4, 1.18), ncol=3)
    plt.legend(handles=[scatter1, scatter2, scatter3, scatter4, scatter5], loc='upper center', bbox_to_anchor=(0.4, 1.18), ncol=5)
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.show()
    # plt.savefig(fname)


def save_activation_projection_tsne_for_vlguard(
    activations1,
    activations2,
    activations3,
    fname,
    title,
    label1="Unsafe",
    label2="Safe_unsafe",
    label3="Safe_safe",
):
    """
    activations1: n_samples x vector dim tensor
    activations2: n_samples x vector dim tensor

    projects to n_samples x 2 dim tensor using t-SNE (over the full dataset of both activations 1 and 2) and saves visualization.
    Colors projected activations1 as blue and projected activations2 as red.
    """
    plt.clf()
    activations_np = np.concatenate((np.array(activations1), np.array(activations2), np.array(activations3)), axis=0).astype(np.float16)

    # t-SNE transformation
    tsne = TSNE(n_components=2)
    projected_activations = tsne.fit_transform(np.array(activations_np))
    len1 = np.array(activations1).shape[0]
    len2 = np.array(activations2).shape[0]
    len3 = np.array(activations3).shape[0]
    # Splitting back into activations1 and activations2
    activations1_projected = projected_activations[0:len1, :]
    activations2_projected = projected_activations[len1:len1+len2, :]
    activations3_projected = projected_activations[len1+len2:len1+len2+len3, :]

    # Visualization
    for x, y in activations1_projected:
        plt.scatter(x, y, color="blue", marker="o", alpha=0.4)

    for x, y in activations2_projected:
        plt.scatter(x, y, color="red", marker="o", alpha=0.4)

    for x, y in activations3_projected:
        plt.scatter(x, y, color="green", marker="o", alpha=0.4)


    # Adding the legend
    scatter1 = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="blue",
        markersize=10,
        label=label1,
    )
    scatter2 = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="red",
        markersize=10,
        label=label2,
    )

    scatter3 = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="green",
        markersize=10,
        label=label3,
    )

    plt.legend(handles=[scatter1, scatter2, scatter3], loc='upper center', bbox_to_anchor=(0.47, 1.16), ncol=3)
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    # plt.show()
    plt.savefig(fname)

def extract_activations_for_vlguard(layer, activations_file="aa.json"):
    unsafe_activations = []
    safe_safe_activations = []
    safe_unsafe_activations = []

    with open(activations_file, "r") as f:
        for line in f:
            item = json.loads(line)
            idx = item['idx']
            print(idx)
            id = item['id']
            image_file = item["image_file"]
            safe = item["safe"]
            harmful_category = item['harmful_category']
            harmful_subcategory = item['harmful_subcategory']
            prompt = item["prompt"]
            activations_layer = item["activations"][str(layer)][0]
            if safe == "unsafe":
                unsafe_activations.append(activations_layer)
            elif safe == "safe_unsafe":
                safe_unsafe_activations.append(activations_layer)
            elif safe == "safe_safe":
                safe_safe_activations.append(activations_layer)

    return unsafe_activations, safe_unsafe_activations, safe_safe_activations


def extract_activations_for_safebench(layer, activations_file="aa.json"):
    res = []

    with open(activations_file, "r") as f:
        for line in f:
            item = json.loads(line)
            idx = item['idx']
            print(idx)
            id = item['id']
            image_file = item["image_file"]
            safe = item["safe"]
            harmful_category = item['harmful_category']
            harmful_subcategory = item['harmful_subcategory']
            prompt = item["prompt"]
            activations_layer = item["activations"][str(layer)][0]
            res.append(activations_layer)

    qingli = 3
    return res





if __name__ == "__main__":
    # analysis activations
    save_path = "llava_v1.6_vicuna_7b_vlguard_train_instructions"
    if not os.path.exists(f"clustering/{save_path}"):
        os.mkdir(f"clustering/{save_path}")

    start_layer = 8
    end_layer = 9
    layers = list(range(start_layer, end_layer + 1))

    for layer in layers:
        # vlguard
        answers_vlguard_file = "playground/data/vlguard/llava_v1.6_vicuna_7b_train_filter_activations.json"
        unsafe_activations, safe_unsafe_activations, safe_safe_activations = extract_activations_for_vlguard(layer, activations_file=answers_vlguard_file)

        fname = f"clustering/{save_path}/activations_layer_{layer}.png"
        title = f"t-SNE projected activations layer {layer}"

        # save vlguard
        save_activation_projection_tsne_for_vlguard(unsafe_activations, safe_unsafe_activations, safe_safe_activations,
                                                    fname=fname, title=title)
