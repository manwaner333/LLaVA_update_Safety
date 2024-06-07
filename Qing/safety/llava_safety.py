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

np.random.seed(42)
torch.manual_seed(42)

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def get_vec(layer, path):
    return torch.load(f"vectors/{path}/vec_layer_{layer}.pt")



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
    if (position_ids > after_id).float().sum() > 1:
        print("There are some problems about insert position!!!")
    matrix += mask.float() * vector
    return matrix

def find_subtensor_position(tensor, sub_tensor):
    n, m = tensor.size(0), sub_tensor.size(0)
    if m > n:
        return -1
    for i in range(n - m + 1):
        if torch.equal(tensor[i : i + m], sub_tensor):
            return i
    return -1
def find_instruction_end_postion(tokens, end_str):
    end_pos = find_subtensor_position(tokens, end_str)
    return end_pos + len(end_str) - 1

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


    def get_logits(self, tokens):
        with torch.no_grad():
            logits = self.model(tokens).logits
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

    def decode_all_layers(
        self,
        tokens,
        topk=10,
        print_attn_mech=True,
        print_intermediate_res=True,
        print_mlp=True,
        print_block=True,
    ):
        tokens = tokens.to(self.device)
        self.get_logits(tokens)
        for i, layer in enumerate(self.model.model.layers):
            print(f"Layer {i}: Decoded intermediate outputs")
            if print_attn_mech:
                self.print_decoded_activations(
                    layer.attn_out_unembedded, "Attention mechanism", topk=topk
                )
            if print_intermediate_res:
                self.print_decoded_activations(
                    layer.intermediate_resid_unembedded,
                    "Intermediate residual stream",
                    topk=topk,
                )
            if print_mlp:
                self.print_decoded_activations(
                    layer.mlp_out_unembedded, "MLP output", topk=topk
                )
            if print_block:
                self.print_decoded_activations(
                    layer.block_output_unembedded, "Block output", topk=topk
                )

    def plot_decoded_activations_for_layer(self, layer_number, tokens, topk=10):
        tokens = tokens.to(self.device)
        self.get_logits(tokens)
        layer = self.model.model.layers[layer_number]

        data = {}
        data["Attention mechanism"] = self.get_activation_data(
            layer.attn_out_unembedded, topk
        )[1]
        data["Intermediate residual stream"] = self.get_activation_data(
            layer.intermediate_resid_unembedded, topk
        )[1]
        data["MLP output"] = self.get_activation_data(layer.mlp_out_unembedded, topk)[1]
        data["Block output"] = self.get_activation_data(
            layer.block_output_unembedded, topk
        )[1]

        # Plotting
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
        fig.suptitle(f"Layer {layer_number}: Decoded Intermediate Outputs", fontsize=21)

        for ax, (mechanism, values) in zip(axes.flatten(), data.items()):
            tokens, scores = zip(*values)
            ax.barh(tokens, scores, color="skyblue")
            ax.set_title(mechanism)
            ax.set_xlabel("Value")
            ax.set_ylabel("Token")

            # Set scientific notation for x-axis labels when numbers are small
            ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="x")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def get_activation_data(self, decoded_activations, topk=10):
        softmaxed = torch.nn.functional.softmax(decoded_activations[0][-1], dim=-1)
        values, indices = torch.topk(softmaxed, topk)
        probs_percent = [int(v * 100) for v in values.tolist()]
        tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))
        return list(zip(tokens, probs_percent)), list(zip(tokens, values.tolist()))


    def generate_text(self, image_file, prompt, figure_size, max_new_tokens=50):
        noise_figure = False
        image = load_image(os.path.join(args.image_folder, image_file), noise_figure)
        image_size = image.size
        image_tensor = process_images([image], self.image_processor, self.model.config)

        if type(image_tensor) is list:
            image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        # load conv
        if "llama-2" in self.model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in self.model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in self.model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in self.model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(
                    conv_mode, args.conv_mode, args.conv_mode))
        else:
            args.conv_mode = conv_mode

        conv = conv_templates[args.conv_mode].copy()
        if "mpt" in self.model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = conv.roles

        if image is not None:
            if self.model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n'
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt

            conv.append_message(conv.roles[0], inp)
            image = None

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX,
                                          return_tensors='pt').unsqueeze(0).to(self.model.device)

        instr_pos = find_instruction_end_postion(input_ids[0], self.END_STR) + figure_size - 1
        self.set_after_positions(instr_pos)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        # with torch.inference_mode():
        model_outputs, image_features_size = self.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            max_length=max_new_tokens
        )

        res = self.tokenizer.decode(model_outputs[0], skip_special_tokens=True)

        return res



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
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
    parser.add_argument("--vectors-path", type=str, default=None)

    args = parser.parse_args()

    model_helper = ModelHelper(args)

    print("model-path: {}; question-file: {}; image-folder: {}; adj-layer: {}; multiplier: {}; max-new-tokens: {}; "
          "add-activations: {}; add-dot-products: {}; answers-file: {}"
          .format(args.model_path, args.question_file, args.image_folder, args.adj_layer, args.multiplier
                  , args.max_new_tokens, args.add_activations, args.add_dot_products, args.answers_file))

    layer = args.adj_layer
    multiplier = args.multiplier
    max_new_tokens = args.max_new_tokens
    vectors_path = args.vectors_path

    model_helper.set_save_internal_decodings(False)

    if args.add_activations:
        print("adjust activations.")
        model_helper.reset_all()
        vec = get_vec(layer, vectors_path)
        model_helper.set_add_activations(layer, multiplier * vec.cuda())
    elif args.add_dot_products:
        print("adjust dot_products.")
        model_helper.reset_all()
        vec = get_vec(layer, vectors_path)
        model_helper.set_calc_dot_product_with(layer, vec.cuda())


    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    if os.path.exists(answers_file):
        os.remove(answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    figure_sizes_file = os.path.expanduser(args.figure_sizes_file)
    responses = {}
    figure_size_info = {}

    with open(figure_sizes_file, 'r') as file:
        for line in file:
            line_info = json.loads(line)
            figure_size_info[line_info['image_file']] = line_info['image_features_size']

    count = 0
    with open(answers_file, 'w') as file:
        for line in tqdm(questions):
            idx = line['id']
            image_file = line['image']
            prompt = line['prompt']
            response_from_dataset = line['response_from_dataset']
            figure_size = figure_size_info[image_file][0]
            res = model_helper.generate_text(image_file, prompt, figure_size, max_new_tokens=10)
            output = {"id": idx,
                      "image_file": image_file,
                      "prompt": prompt,
                      "response": res,
                      "response_from_dataset": response_from_dataset,
                      }
            json.dump(output,  file)
            file.write('\n')
            count += 1
    print("Final count is {}".format(count))





