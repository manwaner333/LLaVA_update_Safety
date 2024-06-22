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
import itertools

np.random.seed(42)
torch.manual_seed(42)




def extract_activations(layer, activations_file="aa.json"):
    activations = []
    labels = []

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
            activations_head = item["block_activations"][str(layer)][0]
            activations_layer = list(itertools.chain(*activations_head))
            activations.append(activations_layer)
            labels.append(safe)
    return activations, labels




def save_activation_projection_tsne(
    activations1,
    labels1,
    activations2,
    labels2,
    activations3,
    labels3,
    activations4,
    labels4,
    fname,
    title
):

    plt.clf()
    plt.figure(figsize=(8, 7))
    activations_np = np.concatenate((np.array(activations1), np.array(activations2), np.array(activations3), np.array(activations4)), axis=0).astype(np.float16)

    # t-SNE transformation
    tsne = TSNE(n_components=2)
    projected_activations = tsne.fit_transform(np.array(activations_np))
    len1 = np.array(activations1).shape[0]
    len2 = np.array(activations2).shape[0]
    len3 = np.array(activations3).shape[0]
    len4 = np.array(activations4).shape[0]

    # Splitting back into activations1 and activations2
    activations1_projected = projected_activations[0:len1, :]
    activations2_projected = projected_activations[len1:len1+len2, :]
    activations3_projected = projected_activations[len1+len2:len1+len2+len3, :]
    activations4_projected = projected_activations[len1+len2+len3:len1+len2+len3+len4, :]


    plt.scatter(activations1_projected[:, 0], activations1_projected[:, 1], color="blue", marker="o", alpha=0.4, label="lmalpaca")
    plt.scatter(activations2_projected[:, 0], activations2_projected[:, 1], color="red", marker="o", alpha=0.4, label="xstest")
    plt.scatter(activations3_projected[:, 0], activations3_projected[:, 1], color="green", marker="o", alpha=0.4, label="safebench_alpaca")
    plt.scatter(activations4_projected[:, 0], activations4_projected[:, 1], color="orange", marker="o", alpha=0.4, label="safebench_vlguard")


    # legend = plt.legend(handles=[scatter1, scatter2, scatter3, scatter4], loc='upper center', bbox_to_anchor=(0.45, 1.16), ncol=3)
    plt.legend(loc='upper center', ncol=4)
    # legend.get_frame().set_alpha(0.3)
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    # plt.show()
    plt.savefig(fname)


def save_activation_projection_tsne_detail(
    activations1,
    labels1,
    activations2,
    labels2,
    activations3,
    labels3,
    activations4,
    labels4,
    fname,
    title
):

    plt.clf()
    plt.figure(figsize=(8, 7))
    activations_np = np.concatenate((np.array(activations1), np.array(activations2), np.array(activations3), np.array(activations4)), axis=0).astype(np.float16)

    # t-SNE transformation
    tsne = TSNE(n_components=2)
    projected_activations = tsne.fit_transform(np.array(activations_np))
    len1 = np.array(activations1).shape[0]
    len2 = np.array(activations2).shape[0]
    len3 = np.array(activations3).shape[0]
    len4 = np.array(activations4).shape[0]

    # Splitting back into activations1 and activations2
    activations1_projected = projected_activations[0:len1, :]
    activations2_projected = projected_activations[len1:len1+len2, :]
    activations3_projected = projected_activations[len1+len2:len1+len2+len3, :]
    activations4_projected = projected_activations[len1+len2+len3:len1+len2+len3+len4, :]

    # lm
    lm_index = [item for item in range(len(labels1)) if labels1[item] == 1]
    lm_activations = activations1_projected[lm_index]

    # alpaca_train
    alpaca_train_index = [item for item in range(len(labels1)) if labels1[item] == 0]
    alpaca_train_activations = activations1_projected[alpaca_train_index]

    # xstest
    # xstest_index = [item for item in range(len(labels2)) if labels2[item] == 1]
    # xstest_activations = activations2_projected[xstest_index]
    xstest_activations = activations2_projected

    # safebench_text
    safebench_index = [item for item in range(len(labels3)) if labels3[item] == 1]
    safebench_activations = activations3_projected[safebench_index]

    # alpaca_test
    alpaca_test_index = [item for item in range(len(labels3)) if labels3[item] == 0]
    alpaca_test_activations = activations3_projected[alpaca_test_index]

    # safebench_image
    safebench_image_index = [item for item in range(len(labels4)) if labels4[item] == 1]
    safebench_image_activations = activations4_projected[safebench_image_index]

    # vlguard
    vlguard_index = [item for item in range(len(labels4)) if labels4[item] == 0]
    vlguard_activations = activations4_projected[vlguard_index]



    # lm
    plt.scatter(lm_activations[:, 0], lm_activations[:, 1], color="blue", marker="o", alpha=0.4, label="lm(-)")

    # alpaca_train and alpaca_test
    plt.scatter(alpaca_train_activations[:, 0], alpaca_train_activations[:, 1], color="red", marker="o", alpha=0.4, label="alpaca(+)")
    plt.scatter(alpaca_test_activations[:, 0], alpaca_test_activations[:, 1], color="red", marker="o", alpha=0.4)

    # xstest
    # plt.scatter(xstest_activations[:, 0], xstest_activations[:, 1], color="green", marker="o", alpha=0.4, label="xstest")

    # safebench_text
    plt.scatter(safebench_activations[:, 0], safebench_activations[:, 1], color="orange", marker="o", alpha=0.4, label="safebench_text(-)")

    # safeben_image
    plt.scatter(safebench_image_activations[:, 0], safebench_image_activations[:, 1], color="cyan", marker="o", alpha=0.4, label="safeben_image(-)")

    # vlguard
    plt.scatter(vlguard_activations[:, 0], vlguard_activations[:, 1], color="purple", marker="o", alpha=0.4, label="vlguard(+)")


    # legend = plt.legend(handles=[scatter1, scatter2, scatter3, scatter4], loc='upper center', bbox_to_anchor=(0.45, 1.16), ncol=3)
    plt.legend(loc='upper center',  bbox_to_anchor=(0.45, 1.16), ncol=3)
    # legend.get_frame().set_alpha(0.3)
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    # plt.show()
    plt.savefig(fname)


if __name__ == "__main__":
    # analysis activations
    save_path = "llava_v1.5_7b_instructions"
    if not os.path.exists(f"clustering/{save_path}"):
        os.mkdir(f"clustering/{save_path}")

    start_layer = 0
    end_layer = 31
    layers = list(range(start_layer, end_layer + 1))

    for layer in layers:
        # lmalpaca
        lmalpaca_file = "playground/data/lmalpaca/llava-v1.5-7b_lmalpaca_head_activations_without_image_new.json"
        lmalpaca_activations, lmalpaca_labels = extract_activations(layer, activations_file=lmalpaca_file)

        # xstest
        xstest = "playground/data/xstest/llava-v1.5-7b_xstest_head_activations_without_image.json"
        xstest_activations, xstest_labels = extract_activations(layer, activations_file=xstest)

        # safebench_alpaca
        safebench_alpaca = "playground/data/safebench_alpaca/llava-v1.5-7b_safebench_alpaca_head_activations_without_image_new.json"
        safebench_alpaca_activations, safebench_alpaca_labels = extract_activations(layer, activations_file=safebench_alpaca)

        # safebench_vlguard
        safebench_vlguard = "playground/data/safebench_vlguard/llava-v1.5-7b_safebench_vlguard_head_activations_with_image.json"
        safebench_vlguard_activations, safebench_vlguard_labels = extract_activations(layer, activations_file=safebench_vlguard)


        fname = f"clustering/{save_path}/activations_layer_{layer}.png"
        title = f"t-SNE projected activations layer {layer}"
        save_activation_projection_tsne_detail(lmalpaca_activations, lmalpaca_labels, xstest_activations, xstest_labels,
                                        safebench_alpaca_activations, safebench_alpaca_labels,
                                        safebench_vlguard_activations, safebench_vlguard_labels,
                                        fname=fname, title=title)
