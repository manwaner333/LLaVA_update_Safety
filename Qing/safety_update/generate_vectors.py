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

def load_image(image_file, noise_figure):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')

    if noise_figure:
        blurred_image = image.filter(ImageFilter.GaussianBlur(radius=10))
    else:
        blurred_image = image
    return blurred_image




def add_vector_after_position(matrix, vector, position_ids, after=None):
    after_id = after
    if after_id is None:
        after_id = position_ids.min().item() - 1
    mask = position_ids > after_id
    mask = mask.unsqueeze(-1)
    matrix += mask.float() * vector
    return matrix


class BlockOutputWrapper(torch.nn.Module):
    def __init__(self, block, unembed_matrix, norm, tokenizer):
        super().__init__()
        self.block = block
        self.unembed_matrix = unembed_matrix
        self.norm = norm
        self.tokenizer = tokenizer

        self.block.self_attn = AttnWrapper(self.block.self_attn)
        self.post_attention_layernorm = self.block.post_attention_layernorm

        self.attn_out_unembedded = None
        self.intermediate_resid_unembedded = None
        self.mlp_out_unembedded = None
        self.block_out_unembedded = None

        self.activations = None
        self.add_activations = None
        self.after_position = None

        self.save_internal_decodings = False

        self.calc_dot_product_with = None
        self.dot_products = []

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        # print(output[0])
        # print(output[0].shape)
        self.activations = output[0]
        if self.calc_dot_product_with is not None:
            last_token_activations = self.activations[0, -1, :]
            decoded_activations = self.unembed_matrix(self.norm(last_token_activations))
            top_token_id = torch.topk(decoded_activations, 1)[1][0]
            top_token = self.tokenizer.decode(top_token_id)
            dot_product = torch.dot(last_token_activations, self.calc_dot_product_with)
            self.dot_products.append((top_token, dot_product.cpu().item()))
        if self.add_activations is not None:
            augmented_output = add_vector_after_position(
                matrix=output[0],
                vector=self.add_activations,
                position_ids=kwargs["position_ids"],
                after=self.after_position,
            )
            output = (augmented_output + self.add_activations,) + output[1:]

        if not self.save_internal_decodings:
            return output

        # Whole block unembedded
        self.block_output_unembedded = self.unembed_matrix(self.norm(output[0]))

        # Self-attention unembedded
        attn_output = self.block.self_attn.activations
        self.attn_out_unembedded = self.unembed_matrix(self.norm(attn_output))

        # Intermediate residual unembedded
        attn_output += args[0]
        self.intermediate_resid_unembedded = self.unembed_matrix(self.norm(attn_output))

        # MLP unembedded
        mlp_output = self.block.mlp(self.post_attention_layernorm(attn_output))
        self.mlp_out_unembedded = self.unembed_matrix(self.norm(mlp_output))

        return output

    def add(self, activations):
        self.add_activations = activations

    def reset(self):
        self.add_activations = None
        self.activations = None
        self.block.self_attn.activations = None
        self.after_position = None
        self.calc_dot_product_with = None
        self.dot_products = []


class AttnWrapper(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.activations = None

    def forward(self, *args, **kwargs):
        output = self.attn(*args, **kwargs)
        self.activations = output[0]
        return output

class ModelHelper:
    def __init__(self, args):
        self.device = args.device
        disable_torch_init()
        model_path = os.path.expanduser(args.model_path)
        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, args.model_base,
                                                                                                   self.model_name, args.load_8bit,
                                                                                                   args.load_4bit, device=self.device)
        self.END_STR = torch.tensor(self.tokenizer.encode("ASSISTANT:")[1:]).to(
            self.device
        )
        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i] = BlockOutputWrapper(
                layer, self.model.lm_head, self.model.model.norm, self.tokenizer
            )

    def set_save_internal_decodings(self, value):
        for layer in self.model.model.layers:
            layer.save_internal_decodings = value

    def set_after_positions(self, pos):
        for layer in self.model.model.layers:
            layer.after_position = pos


    def get_logits(self, tokens, images, image_sizes):
        with torch.no_grad():
            outputs = self.model(tokens, images=images, image_sizes=[image_sizes])
            logits = outputs['logits']
            return logits

    def get_last_activations(self, layer):
        return self.model.model.layers[layer].activations

    def set_add_activations(self, layer, activations):
        self.model.model.layers[layer].add(activations)

    def set_calc_dot_product_with(self, layer, vector):
        self.model.model.layers[layer].calc_dot_product_with = vector

    def get_dot_products(self, layer):
        return self.model.model.layers[layer].dot_products

    def reset_all(self):
        for layer in self.model.model.layers:
            layer.reset()

    def print_decoded_activations(self, decoded_activations, label, topk=10):
        data = self.get_activation_data(decoded_activations, topk)[0]
        print(label, data)

    def get_activation_data(self, decoded_activations, topk=10):
        softmaxed = torch.nn.functional.softmax(decoded_activations[0][-1], dim=-1)
        values, indices = torch.topk(softmaxed, topk)
        probs_percent = [int(v * 100) for v in values.tolist()]
        tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))
        return list(zip(tokens, probs_percent)), list(zip(tokens, values.tolist()))



def save_activation_projection_tsne(
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
    activations3_projected = projected_activations[len1+len2:, :]

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

    plt.legend(handles=[scatter1, scatter2, scatter3])
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    # plt.show()
    plt.savefig(fname)

def plot_all_activations(layers, activations_file="aa.json", save_path="b"):
    unsafe_activations = []
    safe_safe_activations = []
    safe_unsafe_activations = []
    if not os.path.exists(f"clustering/{save_path}"):
        os.mkdir(f"clustering/{save_path}")



    for layer in layers:
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

        save_activation_projection_tsne(unsafe_activations, safe_unsafe_activations, safe_safe_activations
                                        , fname=f"clustering/{save_path}/activations_layer_{layer}.png"
                                        , title=f"t-SNE projected activations layer {layer}"
                                        )



def generate_and_save_steering_vectors(model_helper, dataset, start_layer=0, end_layer=32):
    model_name = model_helper.model_name
    model_helper.set_save_internal_decodings(False)
    model_helper.reset_all()
    answers_file = os.path.expanduser(args.answers_file)
    if os.path.exists(answers_file):
        os.remove(answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    with open(answers_file, 'w') as file:
        for item in dataset:
            token_idx = -2
            layers = list(range(start_layer, end_layer + 1))
            activations = dict([(layer, []) for layer in layers])
            idx = item['idx']
            print(idx)
            id = item['id']
            image_file = item["image"]
            safe = item["safe"]
            harmful_category = item['harmful_category']
            harmful_subcategory = item['harmful_subcategory']
            prompt = item["prompt"]

            noise_figure = False
            image = load_image(os.path.join(args.image_folder, image_file), noise_figure)
            image_size = image.size
            image_tensor = process_images([image], model_helper.image_processor, model_helper.model.config)

            if type(image_tensor) is list:
                image_tensor = [image.to(model_helper.model.device, dtype=torch.float16) for image in image_tensor]
            else:
                image_tensor = image_tensor.to(model_helper.model.device, dtype=torch.float16)

            # load conv
            if "llama-2" in model_name.lower():
                conv_mode = "llava_llama_2"
            elif "mistral" in model_name.lower():
                conv_mode = "mistral_instruct"
            elif "v1.6-34b" in model_name.lower():
                conv_mode = "chatml_direct"
            elif "v1" in model_name.lower():
                conv_mode = "llava_v1"
            elif "mpt" in model_name.lower():
                conv_mode = "mpt"
            else:
                conv_mode = "llava_v0"

            if args.conv_mode is not None and conv_mode != args.conv_mode:
                print(
                    '[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(
                        conv_mode, args.conv_mode, args.conv_mode))
            else:
                args.conv_mode = conv_mode

            conv = conv_templates[args.conv_mode].copy()
            if "mpt" in model_name.lower():
                roles = ('user', 'assistant')
            else:
                roles = conv.roles

            if image is not None:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
                image = None

            conv.append_message(conv.roles[0], inp)
            in_prompt = conv.get_prompt()
            input_ids = (tokenizer_image_token(in_prompt, model_helper.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model_helper.model.device))
            # print(input_ids)
            model_helper.reset_all()
            model_helper.get_logits(input_ids, images=image_tensor, image_sizes=image_size)
            for layer in layers:
                activations_all = model_helper.get_last_activations(layer)
                activations_sub = activations_all[0, token_idx, :].detach().cpu().numpy().tolist()
                activations[layer].append(activations_sub)

            # save activations
            output = {"idx": idx,
                      'id': id,
                      "safe": safe,
                      "harmful_category": harmful_category,
                      "harmful_subcategory": harmful_subcategory,
                      "image_file": image_file,
                      "prompt": prompt,
                      "activations": activations,
                      }
            json.dump(output, file)
            file.write('\n')
            # if idx > 30:
            #     break






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="playground/data/vlguard/train_filter.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=None)  # 0.2
    parser.add_argument("--top-p", type=float, default=None)  # 0.99
    parser.add_argument("--top-k", type=int, default=None)  # 5 # there is no top-k before
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--noise-figure", type=bool, default=False)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--add-activations", type=bool, default=False)
    parser.add_argument("--add-dot-products", type=bool, default=False)
    parser.add_argument("--adj-layer", type=int, default=8)
    parser.add_argument("--multiplier", type=float, default=0.5)
    parser.add_argument("--figure-sizes-file", type=str, default="figure_sizes.jsonl")
    parser.add_argument("--start-layer", type=int, default=8)
    parser.add_argument("--end-layer", type=int, default=10)

    args = parser.parse_args()

    model_helper = ModelHelper(args)

    question_file = os.path.expanduser(args.question_file)
    data = []
    with open(question_file, "r") as f:
        for line in f:
            data.append(json.loads(line))

    start_layer = args.start_layer
    end_layer = args.end_layer
    # generate activations
    generate_and_save_steering_vectors(model_helper, data, start_layer=start_layer, end_layer=end_layer)
    # analysis activations
    save_path = args.model_path.split("/")[-1] + "_" + args.question_file.split("/")[-1].split(".")[0]
    plot_all_activations(list(range(start_layer, end_layer + 1)), activations_file=args.answers_file, save_path=save_path)









