import json
import pickle
import os
import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
clear_figure = True

if clear_figure:
    pope_adv_file = "result/coco2014_val/llava_v15_7b_answer_coco_pope_adversarial_extract_info_update_with_role.bin"
else:
    pope_adv_file = "result/coco2014_val/llava_v15_7b_answer_coco_pope_adversarial_extract_info_update_no_clear_with_role.bin"

if clear_figure:
    pope_ran_file = "result/coco2014_val/llava_v15_7b_answer_coco_pope_random_extract_info_update_with_role.bin"
else:
    pope_ran_file = "result/coco2014_val/llava_v15_7b_answer_coco_pope_random_extract_info_update_no_clear_with_role.bin"

if clear_figure:
    pope_pop_file = "result/coco2014_val/llava_v15_7b_answer_coco_pope_popular_extract_info_update_with_role.bin"
else:
    pope_pop_file = "result/coco2014_val/llava_v15_7b_answer_coco_pope_popular_extract_info_update_no_clear_with_role.bin"

if clear_figure:
    pope_adv_llava16 = "result/coco2014_val/llava_v16_7b_answer_coco_pope_adversarial_extract_info_update_with_role.bin"
else:
    pope_adv_llava16 = "result/coco2014_val/llava_v16_7b_answer_coco_pope_adversarial_extract_info_update_no_clear_with_role.bin"

if clear_figure:
    pope_ran_llava16 = "result/coco2014_val/llava_v16_7b_answer_coco_pope_random_extract_info_update_with_role.bin"
else:
    pope_ran_llava16 = "result/coco2014_val/llava_v16_7b_answer_coco_pope_random_extract_info_update_no_clear_with_role.bin"

if clear_figure:
    pope_pop_llava16 = "result/coco2014_val/llava_v16_7b_answer_coco_pope_popular_extract_info_update_with_role.bin"
else:
    pope_pop_llava16 = "result/coco2014_val/llava_v16_7b_answer_coco_pope_popular_extract_info_update_no_clear_with_role.bin"


if clear_figure:
    pope_adv_llavamoe = "result/coco2014_val/llava_v16_mistral_7b_answer_coco_pope_adversarial_extract_info_update_with_role.bin"
else:
    pope_adv_llavamoe = "result/coco2014_val/llava_v16_mistral_7b_answer_coco_pope_adversarial_extract_info_update_no_clear_with_role.bin"

if clear_figure:
    pope_ran_llavamoe = "result/coco2014_val/llava_v16_mistral_7b_answer_coco_pope_random_extract_info_update_with_role.bin"
else:
    pope_ran_llavamoe = "result/coco2014_val/llava_v16_mistral_7b_answer_coco_pope_random_extract_info_update_no_clear_with_role.bin"

if clear_figure:
    pope_pop_llavamoe = "result/coco2014_val/llava_v16_mistral_7b_answer_coco_pope_popular_extract_info_update_with_role.bin"
else:
    pope_pop_llavamoe = "result/coco2014_val/llava_v16_mistral_7b_answer_coco_pope_popular_extract_info_update_no_clear_with_role.bin"



truthful_qa_llama15 = "result/truthful_qa/answer_truthful_qa.bin"

gqa_llava15 = "result/gqa/llava_v15_7b_answer_gqa_testdev_balanced_questions_yes_no_update_with_role.bin"
gqa_llava16 = "result/gqa/llava_v16_7b_answer_gqa_testdev_balanced_questions_yes_no_update_with_role.bin"
gqa_llavamoe = "result/gqa/llava_v16_mistral_7b_answer_gqa_testdev_balanced_questions_yes_no_update_with_role.bin"


mhal_val_file = "result/m_hal/llava_v15_7b_answer_synthetic_val_data_from_M_HalDetect_update_with_role.bin"
mhal_val_llava16_7b = "result/m_hal/llava_v16_7b_answer_synthetic_val_data_from_M_HalDetect_update_with_role.bin"
mhal_val_llava16_moe = "result/m_hal/llava_v16_mistral_7b_answer_synthetic_val_data_from_M_HalDetect_update_with_role.bin"

self_data_llava15_7b = "result/self_data/llava_v15_7b_answer_pope_adversarial_new_prompt_responses_denoted_update_with_role.bin"
self_data_llava16_7b = "result/self_data/llava_v16_7b_answer_pope_adversarial_new_prompt_responses_denoted_update_with_role.bin"
self_data_llava16_moe = "result/self_data/llava_v16_mistral_7b_answer_pope_adversarial_new_prompt_responses_denoted_update_with_role.bin"




label_map = {
    "ACCURATE": 0,
    "ANALYSIS": 0,
    "INACCURATE": 1,
    "UNSURE": 0,
}


def save_bin(file_path, content):
    with open(file_path, "wb") as file:
        pickle.dump(content, file)


def prepare_data(data, model, split=None):

    if data == "self_data":
        if model == "llava15_7b":
            file_name = self_data_llava15_7b
        elif model == "llava16_7b":
            file_name = self_data_llava16_7b
        elif model == "llava16_moe":
            file_name = self_data_llava16_moe

    elif data == "pope_adv":
        if model == "llava15_7b":
            file_name = pope_adv_file
        elif model == "llava16_7b":
            file_name = pope_adv_llava16
        elif model == "llava16_moe":
            file_name = pope_adv_llavamoe

    elif data == "pope_ran":
        if model == "llava15_7b":
            file_name = pope_ran_file
        elif model == "llava16_7b":
            file_name = pope_ran_llava16
        elif model == "llava16_moe":
            file_name = pope_ran_llavamoe

    elif data == "pope_pop":
        if model == "llava15_7b":
            file_name = pope_pop_file
        elif model == "llava16_7b":
            file_name = pope_pop_llava16
        elif model == "llava16_moe":
            file_name = pope_pop_llavamoe

    elif data == "gqa":
        if model == "llava15_7b":
            file_name = gqa_llava15
        elif model == "llava16_7b":
            file_name = gqa_llava16
        elif model == "llava16_moe":
            file_name = gqa_llavamoe

    elif data == "mhal":
        if model == "llava15_7b":
            file_name = mhal_val_file
        elif model == "llava16_7b":
            file_name = mhal_val_llava16_7b
        elif model == "llava16_moe":
            file_name = mhal_val_llava16_moe

    elif data == "truthful_qa":
        if model == "llama15_7b":
            file_name = truthful_qa_llama15


    sentence_train_file = os.path.join("Qing/data/", "_".join([data, model, "train"]) + "_sentence.bin")
    sentence_val_file = os.path.join("Qing/data/", "_".join([data, model, "val"]) + "_sentence.bin")
    question_train_file = os.path.join("Qing/data/", "_".join([data, model, "train"]) + "_question.bin")
    question_val_file = os.path.join("Qing/data/", "_".join([data, model, "val"]) + "_question.bin")
    response_train_file = os.path.join("Qing/data/", "_".join([data, model, "train"]) + "_response.bin")
    response_val_file = os.path.join("Qing/data/", "_".join([data, model, "val"]) + "_response.bin")


    empty_cases = 0
    question_pairs = []
    response_pairs = []
    sentence_pairs = []

    with open(file_name, 'rb') as f:
        responses = pickle.load(f)


    if data == "pope_adv":
        if split == "train":
            filter_file = 'result/coco2014_val/pope_train.json'
        elif split == "val":
            filter_file = 'result/coco2014_val/pope_val.json'
    elif data == "pope_pop":
        if split == "train":
            filter_file = 'result/coco2014_val/pope_train.json'
        elif split == "val":
            filter_file = 'result/coco2014_val/pope_val.json'
    elif data == "pope_ran":
        if split == "train":
            filter_file = 'result/coco2014_val/pope_train.json'
        elif split == "val":
            filter_file = 'result/coco2014_val/pope_val.json'
    elif data == "gqa":
        if split == "train":
            filter_file = 'result/gqa/gqa_train.json'
        elif split == "val":
            filter_file = 'result/gqa/gqa_val.json'
    elif data == "mhal":
        if split == "train":
            filter_file = 'result/m_hal/m_hal_train.json'
        elif split == "val":
            filter_file = 'result/m_hal/m_hal_val.json'
    elif data == "self_data":
        if split == "train":
            filter_file = 'result/self_data/self_data_train.json'
        elif split == "val":
            filter_file = 'result/self_data/self_data_val.json'
    elif data == "truthful_qa":
        if split == "train":
            filter_file = 'result/truthful_qa/truthful_qa_train.json'
        elif split == "val":
            filter_file = 'result/truthful_qa/truthful_qa_val.json'


    with open(filter_file, 'r') as f:
        keys_val = json.load(f)

    for idx, content in responses.items():

        if idx not in keys_val:
            continue

        cur_data = content

        labels = [label_map[label] for label in cur_data["labels"]]

        hidden_states = cur_data["logprobs"]["combined_hidden_states"]

        if hidden_states["ques"]:
            try:
                ques_pair = {"features": hidden_states["ques"][0], "label": max(labels)}
                question_pairs.append(ques_pair)
            except:
                print("empty labels:", idx)
                empty_cases += 1

        else:
            print("empty ques:", idx)
            empty_cases += 1

        for i in range(len(labels)):
            if data.startswith("mhal") or data == "self_data":
                if hidden_states[i]:
                    sent_pair = {"features": hidden_states[i][0], "label": labels[i]}
                    sentence_pairs.append(sent_pair)
                    if i == len(labels) - 1:
                        resp_pair = {"features": hidden_states[i][0], "label": max(labels)}
                        response_pairs.append(resp_pair)
                else:
                    print("empty sent:", idx)
                    empty_cases += 1

            else:
                if hidden_states[i]:
                    resp_pair = {"features": hidden_states[i][0], "label": labels[i]}
                    response_pairs.append(resp_pair)
                else:
                    print("empty sent:", idx)
                    empty_cases += 1

    if split == 'train':
        save_bin(question_train_file, question_pairs)
        save_bin(response_train_file, response_pairs)
        save_bin(sentence_train_file, sentence_pairs)
        print(len(question_pairs), len(response_pairs), len(sentence_pairs))
    else:
        save_bin(question_val_file, question_pairs)
        save_bin(response_val_file, response_pairs)
        save_bin(sentence_val_file, sentence_pairs)
        print(len(question_pairs), len(response_pairs), len(sentence_pairs))




# pope_adv
# prepare_data("pope_adv", model="llava15_7b", split="train")
# prepare_data("pope_adv", model="llava15_7b", split="val")
# prepare_data("pope_adv", model="llava16_7b", split="train")
# prepare_data("pope_adv", model="llava16_7b", split="val")
# prepare_data("pope_adv", model="llava16_moe", split="train")
# prepare_data("pope_adv", model="llava16_moe", split="val")

# pope_ran
# prepare_data("pope_ran", model="llava15_7b", split="train")
# prepare_data("pope_ran", model="llava15_7b", split="val")
# prepare_data("pope_ran", model="llava16_7b", split="train")
# prepare_data("pope_ran", model="llava16_7b", split="val")
# prepare_data("pope_ran", model="llava16_moe", split="train")
# prepare_data("pope_ran", model="llava16_moe", split="val")

# pope_pop
prepare_data("pope_pop", model="llava15_7b", split="train")
prepare_data("pope_pop", model="llava15_7b", split="val")
prepare_data("pope_pop", model="llava16_7b", split="train")
prepare_data("pope_pop", model="llava16_7b", split="val")
prepare_data("pope_pop", model="llava16_moe", split="train")
prepare_data("pope_pop", model="llava16_moe", split="val")


# gqa
# prepare_data("gqa", model="llava15_7b", split='train')
# prepare_data("gqa", model="llava15_7b", split='val')
# prepare_data("gqa", model="llava16_7b", split='train')
# prepare_data("gqa", model="llava16_7b", split='val')
# prepare_data("gqa", model="llava16_moe", split='train')
# prepare_data("gqa", model="llava16_moe", split='val')


# self_data
# prepare_data("self_data", model="llava15_7b", split="train")
# prepare_data("self_data", model="llava15_7b", split="val")
# prepare_data("self_data", model="llava16_7b", split='train')
# prepare_data("self_data", model="llava16_7b", split='val')
# prepare_data("self_data", model="llava16_moe", split='train')
# prepare_data("self_data", model="llava16_moe", split='val')


# m_hal
# prepare_data("mhal", model="llava15_7b", split="train")
# prepare_data("mhal", model="llava15_7b", split="val")
# prepare_data("mhal", model="llava16_7b", split="train")
# prepare_data("mhal", model="llava16_7b", split="val")
# prepare_data("mhal", model="llava16_moe", split="train")
# prepare_data("mhal", model="llava16_moe", split="val")





