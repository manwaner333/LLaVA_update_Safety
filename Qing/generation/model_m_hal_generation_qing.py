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
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math
import random
import numpy as np
import pickle
from scipy.stats import entropy
import spacy
import re

# np.random.seed(42)
# torch.manual_seed(42)

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, load_4bit=True)

    # questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    with open(os.path.expanduser(args.question_file), "rb") as f:
        questions = pickle.load(f)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    responses = {}

    # 读取数据
    with open('result/m_hal/m_hal_val.json', 'r') as f:
        keys_val = json.load(f)

    count = 0
    for key, line in tqdm(questions.items()):

        # if count > 5:
        #     break

        if key not in keys_val:
            continue

        image_file = line["image_file"]
        label = line["labels"]
        sentences = line['sentences']
        res = re.sub(r'\s+', ' ', line["text"].replace('\n', ''))
        qs = line["prompts"]
        cur_prompt = qs

        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            model_outputs = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                num_return_sequences=5,
                use_cache=True
            )

        output_ids = model_outputs
        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        # filter_outputs = []
        # for output in outputs:
        #     output = output.strip()
        #     if output.endswith(stop_str):
        #         output = output[:-len(stop_str)]
        #     output = output.strip()
        #     filter_outputs.append(output)

        response_value = {"question_id": key,
                          'image_id': image_file,
                          "prompts": cur_prompt,
                          "old_res": res,   # 原来的response
                          "label": label,   # 原来每个句子的label
                          "sentences": sentences,  #根据原来的response 切分成的句子
                          "prompt_tokens": input_token_len,
                          "text": outputs,  # 现在生成的多个response
                          "model_id": model_name
                          }
        responses[key] = response_value
        count += 1

    with open(answers_file, 'wb') as file:
        pickle.dump(responses, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.bin")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.5)  # 0.2
    parser.add_argument("--top_p", type=float, default=None)  # 0.99
    parser.add_argument("--top_k", type=int, default=None)  # 5 # there is no top-k before
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
