import functools
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os
import tqdm
import pandas as pd
import glob
from sklearn.metrics import precision_recall_curve, auc
import pickle

def loadFile(name):
    # load standard json file
    if os.path.isfile(name):
        with open(name) as file:
            data = json.load(file)
    # load file chunks if too big
    elif os.path.isdir(name.split(".")[0]):
        data = {}
        chunks = glob.glob('{dir}/{dir}_*.{ext}'.format(dir=name.split(".")[0], ext=name.split(".")[1]))
        for chunk in chunks:
            with open(chunk) as file:
                data.update(json.load(file))
    else:
        raise Exception("Can't find {}".format(name))
    return data


#  get the question_ids about the binary questions and the answer is yes or no
def get_binary_questions(ori_ques_file, fil_ques_file):
    questions = loadFile(ori_ques_file)
    binary_dict = {}
    for qid, question in questions.items():
        if question["isBalanced"]:
            answerType = "open" if question["types"]["structural"] == "query" else "binary"
            if answerType == 'binary' and question['answer'].lower() in ["yes", "no"]:
                binary_dict[qid] = question

    fil_ques = json.dumps(binary_dict)
    with open(fil_ques_file, 'w') as output_file:
        output_file.write(fil_ques)
    output_file.close()


def analysis_hidden_states(file_name, figure_name, layer):
    # Generate a synthetic high-dimensional dataset
    hidden_states = []
    labels = []
    with open(file_name, "r") as f:
        for line in f.readlines():
            dic = json.loads(line)
            hidden_states.append(dic[layer])
            # (1) text == label, then 1; else: 0
            if dic['text'].lower() == dic['label'].lower():
                labels.append(1)
            else:
                labels.append(0)

    # Applying PCA to reduce dimensions to 2
    pca = PCA(n_components=3)
    X_pca_3d = pca.fit_transform(hidden_states)

    # Applying K-Means clustering
    kmeans = KMeans(n_clusters=2, random_state=0)
    clusters = kmeans.fit_predict(X_pca_3d)

    # Plotting in 3D
    fig = plt.figure(figsize=(14, 8))

    # Plotting the PCA-reduced data colored by true labels
    ax1 = fig.add_subplot(121, projection='3d')
    for i in range(len(X_pca_3d)):
        if labels[i] == 1:
            s1 = ax1.scatter(X_pca_3d[i, 0], X_pca_3d[i, 1], X_pca_3d[i, 2], color='r', alpha=0.7)
        else:
            s2 = ax1.scatter(X_pca_3d[i, 0], X_pca_3d[i, 1], X_pca_3d[i, 2], color='y', alpha=0.7)
    ax1.set_title('True Labels in 3D PCA-reduced Space')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_zlabel('PC3')
    ax1.legend((s1, s2), ("True", "False"), loc='best')

    # Plotting the PCA-reduced data colored by cluster labels
    ax2 = fig.add_subplot(122, projection='3d')
    for i in range(len(X_pca_3d)):
        if clusters[i] == 1:
            o1 = ax2.scatter(X_pca_3d[i, 0], X_pca_3d[i, 1], X_pca_3d[i, 2], color='r', alpha=0.7)
        else:
            o2 = ax2.scatter(X_pca_3d[i, 0], X_pca_3d[i, 1], X_pca_3d[i, 2], color='y', alpha=0.7)
    ax2.set_title('K-Means Clusters in 3D PCA-reduced Space')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_zlabel('PC3')
    ax2.legend((o1, o2), ("Class 1", "Class 2"), loc='best')
    plt.tight_layout()
    path = 'gqa/' + figure_name + '_' + layer + '.png'
    # os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150, format='png')
    plt.show()


def print_AUC(P, R):
    print("AUC: {:.2f}".format(auc(R, P)*100))


def get_questions_answers_merge(que_file_name, ans_file_name, merge_file):
    questions = loadFile(que_file_name)
    binary_dict = {}
    for qid, question in questions.items():
        binary_dict[qid] = question['answer']
    mer_file = open(merge_file, "w")
    with open(ans_file_name, 'r') as predictions:
        for line in predictions:
            prediction = json.loads(line)
            if prediction['question_id'] in binary_dict.keys():
                idx = prediction['question_id']
                text = prediction['text']
                cur_prompt = prediction['prompt']
                layer_0 = prediction['layer_0']
                layer_1 = prediction['layer_1']
                layer_2 = prediction['layer_2']
                layer_3 = prediction['layer_3']
                label = binary_dict[idx]

                mer_file.write(json.dumps({"question_id": idx,
                                           "prompt": cur_prompt,
                                           "text": text,
                                           "label": label,
                                           "layer_0": layer_0,
                                           "layer_1": layer_1,
                                           "layer_2": layer_2,
                                           "layer_3": layer_3,
                                           "metadata": {}}) + "\n")

    merge_file.close()

def get_prob_entropy_infor(que_file_name, ans_file_name):
    questions = loadFile(que_file_name)
    binary_dict = {}
    for qid, question in questions.items():
        binary_dict[qid] = question['answer']
    labels_true = []
    labels_false = []
    logprobs = []
    entropys = []
    with open(ans_file_name, 'r') as predictions:
        for line in predictions:
            prediction = json.loads(line)
            if prediction['question_id'] in binary_dict.keys():
                idx = prediction['question_id']
                text = prediction['text']
                cur_prompt = prediction['prompt']
                label = binary_dict[idx]
                logprob = float(prediction['logprob'])
                entropy = float(prediction['entropy'])
                if label.lower() == text.lower():
                    labels_true.append(1.0)
                    labels_false.append(0.0)
                else:
                    labels_true.append(0.0)
                    labels_false.append(1.0)
                logprobs.append(logprob)
                entropys.append(entropy)

    return labels_true, labels_false, logprobs, entropys

def get_precision_recall_curve(labels, scores, pos_label=1, oneminus_pred=False):
    if oneminus_pred:
        scores = [1.0 - x for x in scores]
    P, R, thre = precision_recall_curve(labels, scores, pos_label=pos_label)
    return P, R, thre

def plot_auc_pr(labels, Rb1, Pb1, Rb2, Pb2, flag):

    random_baseline = np.mean(labels)

    plt.figure(figsize=(5.5, 4.5))
    plt.hlines(y=random_baseline, xmin=0, xmax=1.0, color='grey', linestyles='dotted', label='Random')
    plt.plot(Rb1, Pb1, '-', label='logP')
    plt.plot(Rb2, Pb2, '-', label='H')
    plt.legend()
    # plt.ylim(0.7, 1.02)
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    path = 'gqa/image_' + flag + '.png'
    plt.savefig(path, dpi=150, format='png')
    plt.show()

def generate_gqa_filter_new_prompts(ori_file_name, new_file_name):
    new_file = open(new_file_name, "w")
    images = []
    with open(ori_file_name, 'r') as f:
        for line in f.readlines():
            dic = json.loads(line)
            image = dic['image']
            if image not in images:
                images.append(image)
                question_id = dic['question_id']
                text = "Provide a concise description of the given image."
                new_file.write(json.dumps({"question_id": question_id,
                                           "image": image,
                                           "text": text,
                                           "label": ""
                                           }) + "\n")
                new_file.flush()
    new_file.close()


def generate_gqa_answers_sentences(prompt_file, answers_file, dataset_file):

    ques_images_dic = {}
    with open(prompt_file, 'r') as f:
        for line in f.readlines():
            dic = json.loads(line)
            ques_id = dic['question_id']
            image = dic['image']
            ques_images_dic[ques_id] = image

    with open(answers_file, "rb") as f:
        responses = pickle.load(f)

    res = []
    for idx, response in responses.items():
        # from the cache
        prompt = response["prompts"]
        text = response['text']
        sentences = response['sentences']
        image = ques_images_dic[idx]
        model_id = response['model_id']
        for ele in sentences:
            res.append([image, prompt, text, ele, model_id])

    df = pd.DataFrame(data=res, columns=['image_id', 'prompt', 'passage', 'sentence', "model_id"])
    df.to_csv(dataset_file)




if __name__ == "__main__":

    ############# get binary questions
    ori_ques_file = '../data/gqa/questions1.2/testdev_balanced_questions.json'
    fil_ques_file = '../data/gqa/questions1.2/testdev_balanced_questions_yes_no.json'
    get_binary_questions(ori_ques_file, fil_ques_file)

    ############ 分析gqa生成的hidden states
    # # 1、将生成的答案和原来的问题进行合并
    # que_file_name = '../data/gqa/questions1.2/testdev_balanced_questions_filter.json'
    # ans_file_name = 'gqa/llava_gqa_testdev_balanced_predictions_filter_1.json'
    # merge_file = 'gqa/llava_questions_predictions_merge.json'
    # get_questions_answers_merge(que_file_name, ans_file_name, merge_file)
    # ## 2、分析产生的hidden states
    # file_name = 'gqa/llava_questions_predictions_merge.json'
    # figure_name = "hidden_states"
    # layer = 'layer_0'
    # analysis_hidden_states(file_name, figure_name, layer)


    # ######### 利用selfcheck 里面的一些方法来研究GQA上面的二分类数据上的结果
    # que_file_name = '../data/gqa/questions1.2/testdev_balanced_questions_filter.json'
    # ans_file_name = 'gqa/llava_gqa_testdev_balanced_predictions_filter.json'
    # labels_true, labels_false, logprobs, entropys = get_prob_entropy_infor(que_file_name, ans_file_name)
    # Pb1, Rb1, thre1 = get_precision_recall_curve(labels_true, logprobs, oneminus_pred=False)
    # Pb2, Rb2, thre2  = get_precision_recall_curve(labels_true, entropys, oneminus_pred=True)
    #
    # print("-----------------------")
    # print("Baseline1: logP")
    # print_AUC(Pb1, Rb1)
    # print("-----------------------")
    # print("Baseline2: H")
    # print_AUC(Pb2, Rb2)
    #
    # plot_auc_pr(labels_true, Rb1, Pb1, Rb2, Pb2, flag="true")
    #
    # Pb1, Rb1, thre1 = get_precision_recall_curve(labels_false, logprobs, oneminus_pred=True)
    # Pb2, Rb2, thre2 = get_precision_recall_curve(labels_false, entropys, oneminus_pred=False)
    #
    # print("-----------------------")
    # print("Baseline1: logP")
    # print_AUC(Pb1, Rb1)
    # print("-----------------------")
    # print("Baseline2: H")
    # print_AUC(Pb2, Rb2)
    #
    # plot_auc_pr(labels_false, Rb1, Pb1, Rb2, Pb2, flag="false")


    # 对GQA数据进行summary标注
    # ori_file_name = '../playground/data/gqa/llava_gqa_testdev_balanced_filter.json'
    # new_file_name = '../playground/data/gqa/llava_gqa_testdev_balanced_filter_prompt.json'
    # generate_gqa_filter_new_prompts(ori_file_name, new_file_name)


    # 从模型生成的response中提取出每一句话进行标注， 结果生成csv文件
    # prompt_file = "../playground/data/gqa/llava_gqa_testdev_balanced_filter_prompt.json"
    # answers_file = f"../result/gqa/answer_gqa_testdev_balanced_filter.bin"
    # dataset_file = '../result/gqa/answer_gqa_testdev_balanced_filter.csv'
    # generate_gqa_answers_sentences(prompt_file, answers_file, dataset_file)





    # def generate_coco_pope_adversarial_answers_sentences(file_name, dataset_file):
    #     res = []
    #     with open(file_name, 'r') as f:
    #         for line in f.readlines():
    #             dic = json.loads(line)
    #             image = dic['image']
    #             prompt = dic['prompt']
    #             model_id = dic['model_id']
    #             text = dic['text']
    #             nlp = spacy.load("en_core_web_sm")
    #             text_seqs = []
    #             for item in nlp(text).sents:
    #                 ele = item.text.strip()
    #                 res.append([image, prompt, text, ele, model_id])
    #
    #         df = pd.DataFrame(data=res, columns=['image_id', 'prompt', 'passage', 'sentence', "model_id"])
    #         df.to_csv(dataset_file)



















