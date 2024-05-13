import json
import pickle
import os
import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import torch
clear_figure = True

truthful_qa_llama15 = "result/truthful_qa/answer_truthful_qa_300.bin"


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

    if data == "truthful_qa":
        if model == "llama15_7b":
            file_name = truthful_qa_llama15


    response_train_file = os.path.join("Qing/data/", "_".join([data, model, "train"]) + "_response.bin")
    response_val_file = os.path.join("Qing/data/", "_".join([data, model, "val"]) + "_response.bin")
    response_test_file = os.path.join("Qing/data/", "_".join([data, model, "test"]) + "_response.bin")


    response_pairs = []

    with open(file_name, 'rb') as f:
        responses = pickle.load(f)

    if data == "truthful_qa":
        if split == "train":
            filter_file = 'result/truthful_qa/train_question_ids.pt'
        elif split == "val":
            filter_file = 'result/truthful_qa/val_question_ids.pt'
        elif split == "test":
            filter_file = 'result/truthful_qa/test_question_ids.pt'

    keys_val = torch.load(filter_file).numpy()

    for idx, content in responses.items():

        question_id = content["question_id"]
        if question_id not in keys_val:
            continue
        label = content["label"]
        if label == 'ACCURATE':
            flag = 0
        else:
            flag = 1

        hidden_states = content["hidden_states"]['answer'][-1]
        resp_pair = {"features": hidden_states, "label": flag}
        response_pairs.append(resp_pair)


    if split == 'train':
        save_bin(response_train_file, response_pairs)
        print(len(response_pairs))
    elif split == 'val':
        save_bin(response_val_file, response_pairs)
        print(len(response_pairs))
    elif split == 'test':
        save_bin(response_test_file, response_pairs)
        print(len(response_pairs))


prepare_data("truthful_qa", model="llama15_7b", split="train")
prepare_data("truthful_qa", model="llama15_7b", split="val")
prepare_data("truthful_qa", model="llama15_7b", split="test")






