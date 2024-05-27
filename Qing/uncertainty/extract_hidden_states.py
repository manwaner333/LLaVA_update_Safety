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
from PIL import Image, ImageFilter

np.random.seed(42)
torch.manual_seed(42)

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


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


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    responses = {}
    with_role = args.with_role
    hidden_layer = args.hidden_layer
    noise_figure = args.noise_figure

    if "v1.5" in model_name.lower():
        model_flag = "v1.5"
    elif "v1.6-vicuna" in model_name.lower():
        model_flag = "v1.6-vicuna"
    elif "v1.6-mistral" in model_name.lower():
        model_flag = "v1.6-mistral"

    if "pope_adversarial_new_prompt" in args.question_file.lower():
        data_flag = "self_data"
    elif "m_haldetect" in args.question_file.lower():
        data_flag = "m_hal"
    elif "gqa" in args.question_file.lower():
        data_flag = "gqa"
    elif "pope" in args.question_file.lower():
        data_flag = "pope"

    if data_flag == "gqa":
        filter_file = "result/gqa/gqa_val.json"
    elif data_flag == "pope":
        filter_file = "result/coco2014_val/pope_val.json"
    elif data_flag == "m_hal":
        filter_file = "result/m_hal/m_hal_val.json"
    elif data_flag == "self_data":
        filter_file = "result/self_data/self_data_val.json"

    with open(filter_file, 'r') as f:
        keys_val = json.load(f)

    # keys_val = []
    # for i in range(3):
    #     keys_val.append(i)

    print(f"model_flag: {model_flag}; data_flag: {data_flag}")
    image_tokens_lengths = []
    for line in tqdm(questions):
        idx = line["question_id"]

        if idx not in keys_val:
            continue

        image_file = line["image"]

        if data_flag == "gqa":
            image_file = line["image"] + ".jpg"

        if data_flag == "m_hal":
            labels = line["new_labels"]
        else:
            labels = line["labels"]

        sentences = line['sentences']
        qs = line["question"]
        res = re.sub(r'\s+', ' ', line["response"].replace('\n', ''))

        # load image
        image = load_image(os.path.join(args.image_folder, image_file), noise_figure)
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
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                if with_role:
                    inp = DEFAULT_IMAGE_TOKEN + '\n' + qs + " " + conv.roles[1] + ": " + res
                else:
                    inp = DEFAULT_IMAGE_TOKEN + '\n' + qs + " " + res
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            conv.append_message(conv.roles[0], inp)

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            model_outputs, image_features_size = model(
                input_ids,
                images=image_tensor,
                image_sizes=[image_size],
                output_hidden_states=True,
                # output_attentions=True,
                # streamer=streamer,
            )

        # hidden states
        hidden_states = model_outputs['hidden_states'][hidden_layer][0]

        # logit
        logits = model_outputs['logits']

        # attention
        # attentions = model_outputs['attentions']
        # attentions_c = attentions[31][0, 0, :, 0:1]
        # ques_start_idx  and res_end_idx 在此处是相对于 input_ids的
        images_idx = torch.where(input_ids == IMAGE_TOKEN_INDEX)[1].detach().cpu().numpy().tolist()[0]
        # print(images_idx)
        ques_start_idx = images_idx + 2 + 1  # 其中2对应 """ and '\n'
        input_len = input_ids.shape[1]
        if model_flag in ["v1.5", "v1.6-vicuna"]:
            res_end_idx = input_len - 5 - 1   # 其中6对应： "ASSISTANT:"
        elif model_flag in ["v1.6-mistral"]:
            res_end_idx = input_len - 4 - 1  # 其中6对应： [, /, INST, ]
        ques_res_len = res_end_idx - ques_start_idx + 1
        shifted_input_ids = input_ids[:, ques_start_idx + 1:res_end_idx + 1]

        # 下面是相对于整个hidden states 里面的元素的
        if model_flag in ["v1.5", "v1.6-vicuna"]:
            before_image_tokens_length = images_idx
        elif model_flag in ["v1.6-mistral"]:
            before_image_tokens_length = images_idx

        image_tokens_length = image_features_size[0]
        if image_tokens_length not in image_tokens_lengths:
            image_tokens_lengths.append(image_tokens_length)
        # print(f"image_tokens_length that we use: {image_tokens_length}")

        ques_start_for_total_idx = before_image_tokens_length + image_tokens_length + 2         # -6 - ques_res_len + 1
        res_end_for_total_idx = ques_start_for_total_idx + ques_res_len - 1     # -6
        shifted_logits = logits[:, ques_start_for_total_idx:res_end_for_total_idx, :]

        # Convert logits to probabilities
        log_probs = torch.nn.functional.log_softmax(shifted_logits, dim=-1)
        gathered_log_probs = torch.gather(log_probs, 2, shifted_input_ids.unsqueeze(-1)).squeeze(-1)
        gathered_log_probs = gathered_log_probs.detach().cpu().numpy()

        # Convert logits to entropies
        probs = torch.softmax(shifted_logits, dim=-1)[0]
        probs = probs.detach().cpu().numpy()
        entropies = 2 ** (entropy(probs, base=2, axis=-1))

        tokens = []
        token_logprobs = []
        token_entropies = []
        tokens_idx = []
        token_and_logprobs = []
        for t in range(shifted_input_ids.shape[1]):
            gen_tok_id = shifted_input_ids[:, t]
            gen_tok = tokenizer.decode(gen_tok_id)
            lp = gathered_log_probs[:, t][0]
            entro = entropies[t]

            tokens.append(gen_tok)
            token_logprobs.append(lp)
            token_entropies.append(entro)
            token_and_logprobs.append([gen_tok, lp, entro])
            tokens_idx.append(gen_tok_id.detach().cpu().numpy().tolist())

        combined_attentions = {}
        combined_hidden_states = {}
        combined_token_logprobs = {}
        combined_token_entropies = {}

        # ques_tokens = tokenizer(line["question"]).input_ids
        # ques_tokens_len = len(ques_tokens) - 1

        total_tokens = []
        for t in range(input_ids[:, images_idx+1:].shape[1]):
            total_gen_tok_id = input_ids[:, images_idx+1 + t]
            total_gen_tok = tokenizer.decode(total_gen_tok_id)
            total_tokens.append(total_gen_tok)
        question_tf = "".join(line["question"].split(" "))
        xarr = [i for i in range(len(total_tokens))]
        for i1 in xarr:
            mystring = "".join(total_tokens[i1:])
            if question_tf not in mystring:
                break
        i1 = i1 - 1
        for i2 in xarr[::-1]:
            mystring = "".join(total_tokens[i1:i2 + 1])
            if question_tf not in mystring:
                break
        i2 = i2 + 1
        ques_tokens_len = i2 - i1 + 1

        ques_end_for_total_idx = ques_start_for_total_idx + ques_tokens_len - 1
        # ques_hidden_states = hidden_states[ques_end_for_total_idx:ques_end_for_total_idx + 1, :]
        ques_hidden_states = hidden_states[ques_start_for_total_idx:ques_end_for_total_idx + 1, :]
        combined_hidden_states["ques"] = ques_hidden_states.detach().cpu().numpy().tolist()

        # add figure information
        figure_end_for_total_idx = ques_start_for_total_idx - 2
        figure_hidden_state = hidden_states[figure_end_for_total_idx:figure_end_for_total_idx+1]
        combined_hidden_states["fig"] = figure_hidden_state.detach().cpu().numpy().tolist()


        sentences_end = []
        sentences_end.append(ques_end_for_total_idx)
        record = []
        if with_role:
            if model_flag in ["v1.5", "v1.6-vicuna"]:
                start_idx = ques_end_for_total_idx + 5 + 1
            elif model_flag in ["v1.6-mistral"]:
                start_idx = ques_end_for_total_idx + 4 + 1
        else:
            start_idx = ques_end_for_total_idx + 1
        record1 = []
        for sent_i, sentence in enumerate(sentences):
            # sentence exist in the passage, so we need to find where it is [i1, i2]
            sentence_tf = "".join(sentence.split(" "))
            xarr = [i for i in range(len(tokens))]
            for i1 in xarr:
                mystring = "".join(tokens[i1:])
                if sentence_tf not in mystring:
                    break
            i1 = i1 - 1
            for i2 in xarr[::-1]:
                mystring = "".join(tokens[i1:i2 + 1])
                if sentence_tf not in mystring:
                    break
            i2 = i2 + 1

            sentence_len = i2 - i1 + 1
            sentence_end = start_idx + sentence_len - 1
            hidden_state = hidden_states[start_idx:sentence_end + 1, :]
            # attention = attentions[31][0, -1, sentence_end, start_idx:sentence_end + 1]
            # attention = attentions[31][0, :, start_idx:sentence_end + 1, start_idx:sentence_end + 1].detach().cpu().numpy().tolist()
            combined_hidden_states[sent_i] = hidden_state.detach().cpu().numpy().tolist()
            combined_token_logprobs[sent_i] = token_logprobs[i1:i2+1]
            combined_token_entropies[sent_i] = token_entropies[i1:i2+1]
            # combined_attentions[sent_i] = attention
            sentences_end.append(sentence_end)
            record.append([start_idx, sentence_end])
            record1.append([i1, i2])
            start_idx = sentence_end + 1

        if model_flag in ["v1.5", "v1.6-vicuna"]:
            if sentence_end != logits.shape[1] - 1 - 5:
                print("There are some mistake between the length of hidden states and the length of sentence.")
                print(idx)
        elif model_flag in ["v1.6-mistral"]:
            if sentence_end != logits.shape[1] - 1 - 4:
                print("There are some mistake between the length of hidden states and the length of sentence.")
                print(idx)
        else:
            print("There are some mistake between the length of hidden states and the length of sentence.")
            print(idx)

        log_probs = {
            "tokens": tokens,
            "token_logprobs": token_logprobs,
            "token_entropies": token_entropies,
            "tokens_idx": tokens_idx,
            "combined_hidden_states": combined_hidden_states,
            "combined_token_logprobs": combined_token_logprobs,
            "combined_token_entropies": combined_token_entropies,
            "combined_attentions": combined_attentions,
            "token_and_logprobs": token_and_logprobs
        }

        output = {"question_id": idx,
                  "image_file": line["image"],
                  "prompts": line["question"],
                  "text": res,
                  "sentences": sentences,
                  "logprobs": log_probs,
                  "labels": labels
                  }
        responses[idx] = output

    print(image_tokens_lengths)
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
    parser.add_argument("--noise_figure", type=bool, default=False)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--hidden_layer", type=int, default=32)
    args = parser.parse_args()

    eval_model(args)
