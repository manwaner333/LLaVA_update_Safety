import json
import pickle
import os
import tqdm
import numpy as np
from sklearn.model_selection import train_test_split

pope_adv_file = "result/coco2014_val/llava_v15_7b_answer_coco_pope_adversarial_extract_info.bin"
pope_ran_file = "result/coco2014_val/llava_v15_7b_answer_coco_pope_random_extract_info.bin"
pope_pop_file = "result/coco2014_val/llava_v15_7b_answer_coco_pope_popular_extract_info.bin"

pope_adv_llava16 = "result/coco2014_val/llava_v16_7b_answer_coco_pope_adversarial_extract_info.bin"
pope_ran_llava16 = "result/coco2014_val/llava_v16_7b_answer_coco_pope_random_extract_info.bin"
pope_pop_llava16 = "result/coco2014_val/llava_v16_7b_answer_coco_pope_popular_extract_info.bin"

pope_adv_llavamoe = "result/coco2014_val/llava_v16_mistral_7b_answer_coco_pope_adversarial_extract_info.bin"
pope_ran_llavamoe = "result/coco2014_val/llava_v16_mistral_7b_answer_coco_pope_random_extract_info.bin"
pope_pop_llavamoe = "result/coco2014_val/llava_v16_mistral_7b_answer_coco_pope_popular_extract_info.bin"


gqa_llava15 = "result/gqa/llava_v15_7b_answer_gqa_testdev_balanced_questions_yes_no.bin"
gqa_llava16 = "result/gqa/llava_v16_7b_answer_gqa_testdev_balanced_questions_yes_no.bin"
gqa_llavamoe = "result/gqa/llava_v16_mistral_7b_answer_gqa_testdev_balanced_questions_yes_no.bin"

mhal_train_file = "result/m_hal/llava_v15_7b_answer_synthetic_train_data_from_M_HalDetect.bin"
mhal_val_file = "result/m_hal/llava_v15_7b_answer_synthetic_val_data_from_M_HalDetect.bin"

mhal_train_llava16_7b = "result/m_hal/llava_v16_7b_answer_synthetic_train_data_from_M_HalDetect.bin"
mhal_val_llava16_7b = "result/m_hal/llava_v16_7b_answer_synthetic_val_data_from_M_HalDetect.bin"

mhal_train_llava16_moe = "result/m_hal/llava_v16_mistral_7b_answer_synthetic_train_data_from_M_HalDetect.bin"
mhal_val_llava16_moe = "result/m_hal/llava_v16_mistral_7b_answer_synthetic_val_data_from_M_HalDetect.bin"

self_data_llava15_7b = "result/self_data/llava_v15_7b_answer_pope_adversarial_new_prompt_responses_denoted.bin"
self_data_llava16_7b = "result/self_data/llava_v16_7b_answer_pope_adversarial_new_prompt_responses_denoted.bin"
self_data_llava16_moe = "result/self_data/llava_v16_mistral_7b_answer_pope_adversarial_new_prompt_responses_denoted.bin"


label_map = {
    "ACCURATE": 0,
    "ANALYSIS": 0,
    "INACCURATE": 1,
    "UNSURE": 0,
}


def save_bin(file_path, content):
    with open(file_path, "wb") as file:
        pickle.dump(content, file)


# with open(pope_file, 'rb') as f:
#     content = pickle.load(f)
#
# for item in content:
#     question_pairs = []

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
            if split == "train":
                file_name = mhal_train_file
            elif split == "val":
                file_name = mhal_val_file
        elif model == "llava16_7b":
            if split == "train":
                file_name = mhal_train_llava16_7b
            elif split == "val":
                file_name = mhal_val_llava16_7b
        elif model == "llava16_moe":
            if split == "train":
                file_name = mhal_train_llava16_moe
            elif split == "val":
                file_name = mhal_val_llava16_moe


    with open(file_name, 'rb') as f:
        content = pickle.load(f)

    if data.startswith("mhal"):
        sentence_file = os.path.join("Qing/data/", "_".join([data, model, split]) + "_sentence.bin")
        question_file = os.path.join("Qing/data/", "_".join([data, model, split]) + "_question.bin")
        response_file = os.path.join("Qing/data/", "_".join([data, model, split]) + "_response.bin")

    elif data == "self_data":
        sentence_train_file = os.path.join("Qing/data/", "_".join([data, model, "train"]) + "_sentence.bin")
        sentence_val_file = os.path.join("Qing/data/", "_".join([data, model, "val"]) + "_sentence.bin")
        question_train_file = os.path.join("Qing/data/", "_".join([data, model, "train"]) + "_question.bin")
        question_val_file = os.path.join("Qing/data/", "_".join([data, model, "val"]) + "_question.bin")
        response_train_file = os.path.join("Qing/data/", "_".join([data, model, "train"]) + "_response.bin")
        response_val_file = os.path.join("Qing/data/", "_".join([data, model, "val"]) + "_response.bin")

    else:
        question_train_file = os.path.join("Qing/data/", "_".join([data, model, "train"]) + "_question.bin")
        question_val_file = os.path.join("Qing/data/", "_".join([data, model, "val"]) + "_question.bin")
        response_train_file = os.path.join("Qing/data/", "_".join([data, model, "train"]) + "_response.bin")
        response_val_file = os.path.join("Qing/data/", "_".join([data, model, "val"]) + "_response.bin")

    empty_cases = 0
    question_pairs = []
    response_pairs = []

    if data.startswith("mhal") or data == "self_data":
        sentence_pairs = []

    for item in content:
        cur_data = content[item]

        labels = [label_map[label] for label in cur_data["labels"]]

        hidden_states = cur_data["logprobs"]["combined_hidden_states"]

        if hidden_states["ques"]:
            try:
                ques_pair = {"features": hidden_states["ques"][0], "label": max(labels)}
                question_pairs.append(ques_pair)
            except:
                print("empty labels:", item)
                empty_cases += 1

        else:
            print("empty ques:", item)
            empty_cases += 1

        for idx in range(len(labels)):
            if data.startswith("mhal") or data == "self_data":
                if hidden_states[idx]:
                    sent_pair = {"features": hidden_states[idx][0], "label": labels[idx]}
                    sentence_pairs.append(sent_pair)
                    if idx == len(labels) - 1:
                        resp_pair = {"features": hidden_states[idx][0], "label": max(labels)}
                        response_pairs.append(resp_pair)
                else:
                    print("empty sent:", item)
                    empty_cases += 1

            else:
                if hidden_states[idx]:
                    resp_pair = {"features": hidden_states[idx][0], "label": labels[idx]}
                    response_pairs.append(resp_pair)
                else:
                    print("empty sent:", item)
                    empty_cases += 1

    if data.startswith("mhal"):
        save_bin(question_file, question_pairs)
        save_bin(response_file, response_pairs)
        save_bin(sentence_file, sentence_pairs)
        print(len(question_pairs), len(response_pairs), len(sentence_pairs))

    elif data == "self_data":
        idx_ques_file = f"{data}_ques_idx.json"
        idx_sent_file = f"{data}_sent_idx.json"
        idx_resp_file = f"{data}_resp_idx.json"

        idx_ques_path = os.path.join("Qing/data/", idx_ques_file)
        idx_sent_path = os.path.join("Qing/data/", idx_sent_file)
        idx_resp_path = os.path.join("Qing/data/", idx_resp_file)

        pairs = [question_pairs, response_pairs, sentence_pairs]
        paths = [idx_ques_path, idx_resp_path, idx_sent_path]
        targets = [[question_train_file, question_val_file], [response_train_file, response_val_file], [sentence_train_file, sentence_val_file]]

        for idx in range(len(pairs)):
            if not os.path.exists(paths[idx]):
                data_idx = list(range(len(pairs[idx])))
                train, val, idx_train, idx_val = train_test_split(pairs[idx], data_idx)
                idx_json = {"train": idx_train, "val": idx_val}
                json.dump(idx_json, open(paths[idx], "w"))
                print("save idx file {}".format(paths[idx]))

            else:
                idx_json = json.load(open(paths[idx]))
                print("load idx file {}".format(paths[idx]))
                train = [pairs[idx][i] for i in idx_json["train"]]
                val = [pairs[idx][i] for i in idx_json["val"]]
            save_bin(targets[idx][0], train)
            save_bin(targets[idx][1], val)
    else:
        idx_ques_file = f"{data}_ques_idx.json"
        idx_resp_file = f"{data}_resp_idx.json"
        idx_ques_path = os.path.join("Qing/data/", idx_ques_file)
        idx_resp_path = os.path.join("Qing/data/", idx_resp_file)
        if not os.path.exists(idx_ques_path):
            idx_ques = list(range(len(question_pairs)))
            ques_train, ques_val, idx_train, idx_val = train_test_split(question_pairs, idx_ques)
            ques_json = {"train": idx_train, "val": idx_val}
            json.dump(ques_json, open(idx_ques_path, "w"))
            print("save idx file {}".format(idx_ques_path))

        else:
            json_ques = json.load(open(idx_ques_path))
            print("load idx file {}".format(idx_ques_path))
            ques_train = [question_pairs[idx] for idx in json_ques["train"]]
            ques_val = [question_pairs[idx] for idx in json_ques["val"]]

        save_bin(question_train_file, ques_train)
        save_bin(question_val_file, ques_val)

        if not os.path.exists(idx_resp_path):
            idx_resp = list(range(len(response_pairs)))
            resp_train, resp_val, idx_train, idx_val = train_test_split(response_pairs, idx_resp)
            resp_json = {"train": idx_train, "val": idx_val}
            json.dump(resp_json, open(idx_resp_path, "w"))
            print("save resp file {}".format(idx_resp_path))

        else:
            json_resp = json.load(open(idx_resp_path))
            resp_train = [response_pairs[idx] for idx in json_resp["train"]]
            resp_val = [response_pairs[idx] for idx in json_resp["val"]]
            print("load resp file {}".format(idx_resp_path))

        # resp_train, resp_val = train_test_split(response_pairs)
        save_bin(response_train_file, resp_train)
        save_bin(response_val_file, resp_val)
        print(len(ques_train), len(ques_val), len(resp_train), len(resp_val))


# prepare_data("pope_adv", model="llava15_7b")
# prepare_data("pope_ran", model="llava15_7b")
# prepare_data("pope_pop", model="llava15_7b")
#
# prepare_data("pope_adv", model="llava16_7b")
# prepare_data("pope_ran", model="llava16_7b")
# prepare_data("pope_pop", model="llava16_7b")
#
# prepare_data("pope_adv", model="llava16_moe")
# prepare_data("pope_ran", model="llava16_moe")
# prepare_data("pope_pop", model="llava16_moe")

# prepare_data("gqa", model="llava15_7b")
# prepare_data("gqa", model="llava16_7b")
# prepare_data("gqa", model="llava16_moe")

# prepare_data("self_data", model="llava15_7b")
# prepare_data("self_data", model="llava16_7b")
# prepare_data("self_data", model="llava16_moe")

prepare_data("mhal", model="llava15_7b", split="train")
prepare_data("mhal", model="llava15_7b", split="val")

prepare_data("mhal", model="llava16_7b", split="train")
prepare_data("mhal", model="llava16_7b", split="val")

prepare_data("mhal", model="llava16_moe", split="train")
prepare_data("mhal", model="llava16_moe", split="val")

# sents = pickle.load(open("data/mhal_llava16_7b_train_sentence.bin"))

# prepare_data("gqa", model="llava15_7b")



