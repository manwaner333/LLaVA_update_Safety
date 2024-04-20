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


# Adversarial
print("Adversarial")
question_file = "playground/data/coco2014_val_qa_eval/coco_pope_adversarial_extract_info.json"
questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]
questions_number = 0
images = set()
for line in tqdm(questions):
    imge = line["image"]
    images.add(imge)
    questions_number += 1

print(questions_number)
print(len(images))


# Popular
print("Popular")
question_file = "playground/data/coco2014_val_qa_eval/coco_pope_popular_extract_info.json"
questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]
questions_number = 0
images = set()
for line in tqdm(questions):
    imge = line["image"]
    images.add(imge)
    questions_number += 1

print(questions_number)
print(len(images))


# Random
print("Random")
question_file = "playground/data/coco2014_val_qa_eval/coco_pope_random_extract_info.json"
questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]
questions_number = 0
images = set()
for line in tqdm(questions):
    imge = line["image"]
    images.add(imge)
    questions_number += 1

print(questions_number)
print(len(images))


# GQA
print("GQA")
question_file = "playground/data/gqa/gqa_testdev_balanced_questions_yes_no.json"
questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]
questions_number = 0
images = set()
for line in tqdm(questions):
    imge = line["image"]
    images.add(imge)
    questions_number += 1

print(questions_number)
print(len(images))

# M-Hal
print("M-Hal")
question_file = "playground/data/m_hal/synthetic_val_data_from_M_HalDetect.json"
questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]
questions_number = 0
images = set()
for line in tqdm(questions):
    imge = line["image"]
    images.add(imge)
    questions_number += 1

print(questions_number)
print(len(images))



# self-data
print("self-data")
question_file = "playground/data/self_data/llava_v15_7b_pope_adversarial_new_prompt_responses_denoted.json"
questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]
questions_number = 0
images = set()
for line in tqdm(questions):
    imge = line["image"]
    images.add(imge)
    questions_number += 1

print(questions_number)
print(len(images))
