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
import spacy
from transformers import TextStreamer
import re

# np.random.seed(42)
torch.manual_seed(42)

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    with open(os.path.expanduser(args.question_file), "rb") as f:
        questions = pickle.load(f)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    responses = {}

    # 读取数据
    if "pope_adversarial_new_prompt" in args.question_file.lower():
        data_flag = "self_data"
    elif "m_haldetect" in args.question_file.lower():
        data_flag = "m_hal"

    if data_flag == 'm_hal':
        filter_file = 'result/m_hal/m_hal_val.json'
    elif data_flag == 'self_data':
        filter_file = 'result/self_data/self_data_val.json'

    with open(filter_file, 'r') as f:
        keys_val = json.load(f)

    if "v1.5" in model_name.lower():
        model_flag = "v1.5"
    elif "v1.6-vicuna" in model_name.lower():
        model_flag = "v1.6-vicuna"
    elif "v1.6-mistral" in model_name.lower():
        model_flag = "v1.6-mistral"
    print(f"model_flag: {model_flag}; data_flag: {data_flag}")

    for key, line in tqdm(questions.items()):
        if key not in keys_val:
            continue

        # idx = line["question_id"]
        image_file = line["image_file"]
        label = line['labels']
        sentences = line['sentences']
        res = re.sub(r'\s+', ' ', line["text"].replace('\n', ''))
        qs = line["prompts"]
        cur_prompt = qs


        # load image
        image = load_image(os.path.join(args.image_folder, image_file))
        image_size = image.size
        image_tensor = process_images([image], image_processor, model.config)

        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)

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
        # print(args.conv_mode)

        conv = conv_templates[args.conv_mode].copy()
        if "mpt" in model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = conv.roles

        if image is not None:
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + qs
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            conv.append_message(conv.roles[0], qs)

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        # streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image_size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                num_return_sequences=5,
                # streamer=streamer,
                # use_cache=True
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        conv.messages[-1][-1] = outputs

        response_value = {"question_id": key,
                          'image_id': image_file,
                          "prompts": cur_prompt,
                          "old_res": res,  # 原来的response
                          "label": label,  # 原来每个句子的label
                          "sentences": sentences,  # 根据原来的response 切分成的句子
                          "text": outputs,  # 现在生成的多个response
                          "model_id": model_name
        }
        responses[key] = response_value

    with open(answers_file, 'wb') as file:
        pickle.dump(responses, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.bin")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.5)  # 0.2
    parser.add_argument("--top_p", type=float, default=None)  # 0.99
    parser.add_argument("--top_k", type=int, default=None)  # 5 # there is no top-k before
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1)
    parser.add_argument("--with_role", type=bool, default=True)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    eval_model(args)
