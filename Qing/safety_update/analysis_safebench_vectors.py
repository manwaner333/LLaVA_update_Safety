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




def save_activation_projection_tsne_for_safebench(
    activations1,
    activations2,
    fname,
    title,
    label1="with_image",
    label2="without_image",
):
    """
    activations1: n_samples x vector dim tensor
    activations2: n_samples x vector dim tensor

    projects to n_samples x 2 dim tensor using t-SNE (over the full dataset of both activations 1 and 2) and saves visualization.
    Colors projected activations1 as blue and projected activations2 as red.
    """
    plt.clf()
    activations_np = np.concatenate((np.array(activations1), np.array(activations2)), axis=0).astype(np.float16)

    # t-SNE transformation
    tsne = TSNE(n_components=2)
    projected_activations = tsne.fit_transform(np.array(activations_np))
    len1 = np.array(activations1).shape[0]
    len2 = np.array(activations2).shape[0]
    # Splitting back into activations1 and activations2
    activations1_projected = projected_activations[0:len1, :]
    activations2_projected = projected_activations[len1:len1+len2, :]

    # Visualization
    for x, y in activations1_projected:
        plt.scatter(x, y, color="blue", marker="o", alpha=0.4)

    for x, y in activations2_projected:
        plt.scatter(x, y, color="red", marker="o", alpha=0.4)

    # for i, txt in enumerate(range(len1)):
    #     plt.annotate(txt, (projected_activations[i, 0], projected_activations[i, 1]), fontsize=8)
    # for i, txt in enumerate(range(len1, len1+len2)):
    #     plt.annotate(txt, (projected_activations[i, 0], projected_activations[i, 1]), fontsize=8)


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


    plt.legend(handles=[scatter1, scatter2], loc='upper center', bbox_to_anchor=(0.47, 1.16), ncol=3)
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.show()
    # plt.savefig(fname)


def save_activation_projection_tsne_for_safebench_only_oneset(
    activations1,
    fname,
    title,
    label1="with_image",
    label2="without_image",
):
    """
    activations1: n_samples x vector dim tensor
    activations2: n_samples x vector dim tensor

    projects to n_samples x 2 dim tensor using t-SNE (over the full dataset of both activations 1 and 2) and saves visualization.
    Colors projected activations1 as blue and projected activations2 as red.
    """
    plt.clf()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    activations_np = np.array(activations1).astype(np.float16)
    # t-SNE transformation
    tsne = TSNE(n_components=2)
    projected_activations = tsne.fit_transform(np.array(activations_np))
    len1 = np.array(activations1).shape[0]
    # Splitting back into activations1 and activations2
    activations1_projected = projected_activations[0:len1, :]


    # Visualization
    for x, y in activations1_projected:
        axes[0].scatter(x, y, color="blue", marker="o", alpha=0.4)

    flags = [True, False, False, False, True, True, False, False, False, True, False, True, True, False, True, True, False,
     True, True, False, True, True, True, True, True, False, True, True, False, True, True, True, True, False, False,
     True, True, False, False, False, True, True, False, True, True, False, False, False, False, False, False, False,
     False, False, False, True, False, False, False, False, True, True, False, False, False, True, False, False, True,
     False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False,
     True, False, False, True, True, False, True, False, True, True, True, False, False, False, True, True, True, True,
     True, True, True, False, True, False, False, True, True, False, True, True, True, True, False, False, False, True,
     False, True, True, False, True, False, True, True, True, True, False, False, False, True, False, False, True, True,
     False, False, True, True, True, False, True, True, False, False, True, False, False, False, False, False, False,
     True, True, False, False, False, True, False, False, True, False, True, False, False, False, False, True, True,
     False, False, False, True, False, True, True, True, True, False, False, True, True, False, False, False, False,
     False, False, False, True, False, False, False, False, False, False, False, True, False, True, False, False, False,
     False, False, True, False, True, False, False, True, True, False, True, True, False, False, False, False, True,
     True, False, False, False, True, True, True, False, True, False, False, True, True, False, True, True, False, True,
     True, False, False, False, False, False, False, False, False, True, False, True, True, True, True, False, True,
     False, True, True, True, True, True, False, True, True, False, True, True, True, True, True, False, True, True,
     True, False, True, True, True, True, True, False, False, True, True, True, True, False, True, True, True, False,
     False, True, True, True, False, False, False, False, False, True, False, False, False, False, True, True, True,
     False, False, False, False, False, True, False, True, False, False, True, True, False, False, True, True, False,
     False, True, True, False, False, True, False, False, True, False, False, True, False, False, False, False, True,
     True, True, True, False, False, False, True, True, False, True, True, False, False, True, True, True, True, True,
     False, True, True, True, False, True, True, True, False, False, True, True, True, False, True, True, True, True,
     True, True, True, True, True, False, True, False, False, True, True, True, True, True, True, True, True, True,
     True, True, True, False, True, True, False, True, True, True, True, True, True, True, True, False, True, False,
     True, False, False, True, False, True, True, True, False, True, False, True, True, False, True, True, True, True,
     True, True, True, True, True, True, True, True, True, True, True, False, True, True, False, False, False, True,
     True, False, True, True, False, False, True, False, True, True, True, True, True, False, True, True, False, True,
     False, True, False, True, True, False, True, False, True, True, False, True, True, False, False, True, False, True,
     True, True, True, True, False, True, False, True, False, True]


    idx = 0
    for x, y in activations1_projected:
        if flags[idx] == True:
            axes[1].scatter(x, y, color="red", marker="o", alpha=0.4)
        else:
            axes[1].scatter(x, y, color="blue", marker="o", alpha=0.4)
        idx += 1


    # Adding the legend
    # scatter1 = plt.Line2D(
    #     [0],
    #     [0],
    #     marker="o",
    #     color="w",
    #     markerfacecolor="blue",
    #     markersize=10,
    #     label=label1,
    # )
    # scatter2 = plt.Line2D(
    #     [0],
    #     [0],
    #     marker="o",
    #     color="w",
    #     markerfacecolor="red",
    #     markersize=10,
    #     label=label2,
    # )


    # plt.legend(handles=[scatter1], loc='upper center', bbox_to_anchor=(0.47, 1.16), ncol=3)
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
    save_path = "llava_v1.5_7b_safebench_instructions"
    if not os.path.exists(f"clustering/{save_path}"):
        os.mkdir(f"clustering/{save_path}")

    start_layer = 8
    end_layer = 10
    layers = list(range(start_layer, end_layer + 1))

    for layer in layers:

        answers_safebench_with_image = "playground/data/safebench/llava-v1.5-7b_safebench_activations.json"
        # answers_safebench_with_image = "playground/data/safebench/safebench_activations.json"
        safebench_with_images = extract_activations_for_safebench(layer, activations_file=answers_safebench_with_image)

        answers_safebench_without_image = "playground/data/safebench/llava-v1.5-7b_safebench_activations_without_image.json"
        # answers_safebench_without_image = "playground/data/safebench/safebench_activations_without_image.json"
        safebench_without_images = extract_activations_for_safebench(layer, activations_file=answers_safebench_without_image)


        fname = f"clustering/{save_path}/activations_layer_{layer}.png"
        title = f"t-SNE projected activations layer {layer}"
        # save_activation_projection_tsne_for_safebench(safebench_with_images, safebench_without_images, fname=fname, title=title)
        save_activation_projection_tsne_for_safebench_only_oneset(safebench_without_images, fname=fname, title=title)
