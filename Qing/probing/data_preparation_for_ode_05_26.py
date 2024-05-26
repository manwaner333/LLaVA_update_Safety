import json
import pickle
import os
import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import torch
clear_figure = True
from sklearn.decomposition import PCA

company_llama15 = "result/company/answer_company.bin"
neg_company_llama15 = "result/neg_company/answer_neg_company.bin"
fact_llama15 = "result/fact/answer_fact.bin"
neg_fact_llama15 = "result/neg_fact/answer_neg_fact.bin"
animal_llama15 = "result/animal/answer_animal.bin"
city_llama15 = "result/city/answer_city.bin"
neg_city_llama15 = "result/neg_city/answer_neg_city.bin"
element_llama15 = "result/element/answer_element.bin"
invention_llama15 = "result/invention/answer_invention.bin"


label_map = {
    "ACCURATE": 0,
    "ANALYSIS": 0,
    "INACCURATE": 1,
    "UNSURE": 0,
}


def save_bin(file_path, content):
    with open(file_path, "wb") as file:
        pickle.dump(content, file)


def prepare_data(data, model, split, one_flag):
    if data == "company":
        if model == "llama15_7b":
            file_name = company_llama15
    elif data == "neg_company":
        if model == "llama15_7b":
            file_name = neg_company_llama15
    elif data == "fact":
        if model == "llama15_7b":
            file_name = fact_llama15
    elif data == "neg_fact":
        if model == "llama15_7b":
            file_name = neg_fact_llama15
    elif data == "animal":
        if model == "llama15_7b":
            file_name = animal_llama15
    elif data == "city":
        if model == "llama15_7b":
            file_name = city_llama15
    elif data == "neg_city":
        if model == "llama15_7b":
            file_name = neg_city_llama15
    elif data == "element":
        if model == "llama15_7b":
            file_name = element_llama15
    elif data == "invention":
        if model == "llama15_7b":
            file_name = invention_llama15

    if one_flag:
        response_train_file = os.path.join("Qing/data/", "_".join([data, model, "train"]) + "_response_one.bin")
        response_val_file = os.path.join("Qing/data/", "_".join([data, model, "val"]) + "_response_one.bin")
        response_test_file = os.path.join("Qing/data/", "_".join([data, model, "test"]) + "_response_one.bin")
    else:
        response_train_file = os.path.join("Qing/data/", "_".join([data, model, "train"]) + "_response_avg.bin")
        response_val_file = os.path.join("Qing/data/", "_".join([data, model, "val"]) + "_response_avg.bin")
        response_test_file = os.path.join("Qing/data/", "_".join([data, model, "test"]) + "_response_avg.bin")


    response_pairs = []

    with open(file_name, 'rb') as f:
        responses = pickle.load(f)

    if split == "train":
        filter_file = 'result/' + data + '/train_question_ids.pt'
    elif split == "val":
        filter_file = 'result/' + data + '/val_question_ids.pt'
    elif split == "test":
        filter_file = 'result/' + data + '/test_question_ids.pt'

    keys_val = torch.load(filter_file).numpy()

    for idx, content in responses.items():

        question_id = content["question_id"]
        sub_hidden_states = []
        if question_id not in keys_val:
            continue
        label = content["label"]
        if label == 1:
            flag = 0
        else:
            flag = 1
        if one_flag:
            hidden_states = content["hidden_states"]['ques'][-1]
        else:
            question_hidden_states = content["hidden_states"]['ques']
            sub_hidden_states.extend(question_hidden_states)
            hidden_states = np.mean(np.array(sub_hidden_states), axis=0).tolist()
        resp_pair = {"question_id": question_id, "features": hidden_states, "label": flag}
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

if __name__ == '__main__':
    oneflag = False
    data_name = "neg_city"
    prepare_data(data_name, model="llama15_7b", split="train", one_flag=oneflag)
    prepare_data(data_name, model="llama15_7b", split="val", one_flag=oneflag)
    prepare_data(data_name, model="llama15_7b", split="test", one_flag=oneflag)
