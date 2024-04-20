import pickle
import numpy as np
from scipy.stats import entropy
from tqdm import tqdm
import os
import json
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
import random
import pandas as pd
import csv
import joblib
from sklearn.metrics import roc_auc_score


def unroll_pred(scores, indices):
    unrolled = []
    for idx in indices:
        unrolled.extend(scores[idx])
    return unrolled


# def get_PR_with_human_labels(preds, human_labels, pos_label=1, oneminus_pred=False):
#     indices = [k for k in human_labels.keys()]
#     unroll_preds = unroll_pred(preds, indices)
#     if oneminus_pred:
#         unroll_preds = [1.0-x for x in unroll_preds]
#     unroll_labels = unroll_pred(human_labels, indices)
#     assert len(unroll_preds) == len(unroll_labels)
#     print("len:", len(unroll_preds))
#     P, R, thre = precision_recall_curve(unroll_labels, unroll_preds, pos_label=pos_label)
#     # auroc = roc_auc_score(unroll_labels, unroll_preds)
#     return P, R

def get_PR_with_human_labels(preds, human_labels, pos_label=1, oneminus_pred=False):
    unroll_preds = preds
    if oneminus_pred:
        unroll_preds = [1.0 - x for x in unroll_preds]
    unroll_labels = human_labels
    assert len(unroll_preds) == len(unroll_labels)
    print("len:", len(unroll_preds))
    P, R, thre = precision_recall_curve(unroll_labels, unroll_preds, pos_label=pos_label)
    # auroc = roc_auc_score(unroll_labels, unroll_preds)
    return P, R


def print_AUC(P, R):
    print("AUC: {:.2f}".format(auc(R, P) * 100))


def detect_hidden_states(combined_hidden_states):
    for k, v in combined_hidden_states.items():
        if len(v) == 0:
            return False
    return True


def tfidf_encode(vectorizer, sent):
    tfidf_matrix = vectorizer.transform([sent])

    # Convert the TF-IDF matrix for the sentence to a dense format
    dense_tfidf = tfidf_matrix.todense()

    # Get the feature names (vocabulary) from the vectorizer
    feature_names = vectorizer.get_feature_names_out()

    # Tokenize the sentence
    tokenized_sentence = sent.split()

    token_weights = []

    # For each token in the input sentence, get its weight from the TF-IDF model
    for token in tokenized_sentence:
        # Check if the token is in the TF-IDF model's vocabulary
        if token in feature_names:
            # Find the index of the token in the feature names
            token_index = list(feature_names).index(token)
            # Append the weight of the token to the list
            token_weights.append(dense_tfidf[0, token_index])
        else:
            # If the token is not found in the model's vocabulary, assign a weight of 0
            token_weights.append(0)

    return token_weights


def extract_info_from_answers(file_path, data_flag, use_tfidf_weight=False, use_attention_weight=False):

    with open(file_path, "rb") as f:
        responses = pickle.load(f)

    human_label_detect_False = {}
    human_label_detect_True = {}
    average_logprob_scores = {}  # average_logprob
    average_entropy_scores = {}  # lowest_logprob
    lowest_logprob_scores = {}  # average_entropy5
    highest_entropy_scores = {}
    sentences_info = {}
    images_info = {}
    sentences_idx_info = {}
    token_and_logprobs_info = {}
    labels_info = {}
    idx_info = {}
    logprob_avg_response_scores = {}
    logprob_min_response_scores = {}
    entropy_avg_response_scores = {}
    entropy_max_response_scores = {}
    label_True_response = {}
    label_False_response = {}
    tfidf_weight_scores = {}
    attention_weight_scores = {}
    new_vectorizer = joblib.load('Qing/tfidf_model.joblib')

    if data_flag == "self_data":
        filter_file = "result/self_data/self_data_val.json"
    elif data_flag == "m_hal":
        filter_file = "result/m_hal/m_hal_val.json"

    with open(filter_file, 'r') as f:
        keys_val = json.load(f)

    for idx, response in responses.items():

        if idx not in keys_val:
            continue

        question_id = response["question_id"]
        log_probs = response["logprobs"]
        combined_token_logprobs = log_probs["combined_token_logprobs"]
        combined_token_entropies = log_probs["combined_token_entropies"]
        labels = response["labels"]
        sentences_len = len(response["sentences"])
        tokens = response['logprobs']['tokens']
        attentions = response['logprobs']['combined_attentions']


        average_logprob_sent_level = [None for _ in range(sentences_len)]  # [None for _ in range(sentences_len)]
        lowest_logprob_sent_level = [None for _ in range(sentences_len)]
        average_entropy_sent_level = [None for _ in range(sentences_len)]
        highest_entropy_sent_level = [None for _ in range(sentences_len)]
        label_True_sent_level = [None for _ in range(sentences_len)]
        label_False_sent_level = [None for _ in range(sentences_len)]
        sentence_sent_level = [None for _ in range(sentences_len)]
        image_sent_level = [None for _ in range(sentences_len)]
        sentence_idx_sent_level = [None for _ in range(sentences_len)]
        token_and_logprob_sent_level = [None for _ in range(sentences_len)]
        label_sent_level = [None for _ in range(sentences_len)]
        idx_sent_level = [None for _ in range(sentences_len)]
        tfidf_weight_sent_level = [None for _ in range(sentences_len)]
        attention_weight_sent_level = [None for _ in range(sentences_len)]

        response_log_probs = []
        response_entropies = []
        for i in range(sentences_len):
            sentence = response["sentences"][i]
            sentence_log_probs = [item for item in combined_token_logprobs[i]]  # combined_token_logprobs[i]
            sentence_entropies = combined_token_entropies[i]
            label = labels[i]

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

            tfidf_weights = tfidf_encode(new_vectorizer, " ".join(response['logprobs']['tokens'][i1:i2 + 1]))
            sentence_log_probs_weight = [a * b for a, b in zip(sentence_log_probs, tfidf_weights)]
            sentence_entropies_weight = [a * b for a, b in zip(sentence_entropies, tfidf_weights)]
            if use_tfidf_weight:
                sentence_log_probs = sentence_log_probs_weight
                sentence_entropies = sentence_entropies_weight


            attention_weights = attentions    # [i][-1][-1]
            sentence_log_probs_weight = [a * b for a, b in zip(sentence_log_probs, attention_weights)]
            sentence_entropies_weight = [a * b for a, b in zip(sentence_entropies, attention_weights)]
            if use_attention_weight:
                sentence_log_probs = sentence_log_probs_weight
                sentence_entropies = sentence_entropies_weight


            # if label in ['ACCURATE', 'INACCURATE']:
            average_logprob = np.mean(sentence_log_probs)
            lowest_logprob = np.min(sentence_log_probs)
            average_entropy = np.mean(sentence_entropies)
            highest_entropy = np.max(sentence_entropies)

            average_logprob_sent_level[i] = average_logprob
            lowest_logprob_sent_level[i] = lowest_logprob
            average_entropy_sent_level[i] = average_entropy
            highest_entropy_sent_level[i] = highest_entropy
            sentence_sent_level[i] = response["sentences"][i]
            image_sent_level[i] = response["image_file"]
            token_and_logprob_sent_level[i] = log_probs['token_and_logprobs']
            label_sent_level[i] = label
            sentence_idx_sent_level[i] = i
            idx_sent_level[i] = question_id
            tfidf_weight_sent_level[i] = [ele for ele in zip(" ".join(response['logprobs']['tokens'][i1:i2 + 1]).split(), tfidf_weights)]
            attention_weight_sent_level[i] = [ele for ele in zip(" ".join(response['logprobs']['tokens'][i1:i2 + 1]).split(), attention_weights)]

            if label == 'ACCURATE' or label == 'ANALYSIS':
                true_score = 1.0
                false_score = 0.0
            elif label == 'INACCURATE':
                true_score = 0.0
                false_score = 1.0

            label_True_sent_level[i] = true_score
            label_False_sent_level[i] = false_score

            # response
            response_log_probs.extend(sentence_log_probs)
            response_entropies.extend(sentence_entropies)

        # sentence level
        average_logprob_scores[question_id] = average_logprob_sent_level
        lowest_logprob_scores[question_id] = lowest_logprob_sent_level
        average_entropy_scores[question_id] = average_entropy_sent_level
        highest_entropy_scores[question_id] = highest_entropy_sent_level
        human_label_detect_True[question_id] = label_True_sent_level
        human_label_detect_False[question_id] = label_False_sent_level
        sentences_info[question_id] = sentence_sent_level
        images_info[question_id] = image_sent_level
        sentences_idx_info[question_id] = sentence_idx_sent_level
        token_and_logprobs_info[question_id] = token_and_logprob_sent_level
        labels_info[question_id] = label_sent_level
        idx_info[question_id] = idx_sent_level
        tfidf_weight_scores[question_id] = tfidf_weight_sent_level
        attention_weight_scores[question_id] = attention_weight_sent_level

        # response level
        response_avg_logprob = np.mean(response_log_probs)
        response_min_logprob = np.min(response_log_probs)
        response_avg_entropy = np.mean(response_entropies)
        response_max_entropy = np.max(response_entropies)
        logprob_avg_response_scores[question_id] = [response_avg_logprob]
        logprob_min_response_scores[question_id] = [response_min_logprob]
        entropy_avg_response_scores[question_id] = [response_avg_entropy]
        entropy_max_response_scores[question_id] = [response_max_entropy]

        if 0.0 in label_True_sent_level:
            label_True_response[question_id] = [0.0]
            label_False_response[question_id] = [1.0]
        else:
            label_True_response[question_id] = [1.0]
            label_False_response[question_id] = [0.0]

    return (average_logprob_scores, lowest_logprob_scores, average_entropy_scores, highest_entropy_scores
            , human_label_detect_True, human_label_detect_False, sentences_info, images_info, sentences_idx_info
            , token_and_logprobs_info, labels_info, idx_info, logprob_avg_response_scores, logprob_min_response_scores
            , entropy_avg_response_scores, entropy_max_response_scores, label_True_response, label_False_response
            , tfidf_weight_scores, attention_weight_scores)



def form_dataframe_from_extract_info(average_logprob_scores, lowest_logprob_scores, average_entropy_scores, highest_entropy_scores, human_label_detect_True, human_label_detect_False, sentences_info, images_info
                                     , sentences_idx_info, token_and_logprobs_info, labels_info, idx_info, tfidf_weight_scores, attention_weight_scores):

    average_logprob_pd = []
    lowest_logprob_pd = []
    average_entropy_pd = []
    highest_entropy_pd = []
    human_label_detect_True_pd = []
    human_label_detect_False_pd = []
    sentences_pd = []
    images_pd = []
    sentences_idx_pd = []
    token_and_logprobs_pd = []
    labels_pd = []
    idxs_pd = []
    label_True_response_pd = []
    label_False_response_pd = []
    tfidf_weight_pd = []
    attention_weight_pd = []

    for dic_idx in list(average_logprob_scores.keys()):

        average_logprob_pd.extend(average_logprob_scores[dic_idx])
        lowest_logprob_pd.extend(lowest_logprob_scores[dic_idx])
        average_entropy_pd.extend(average_entropy_scores[dic_idx])
        highest_entropy_pd.extend(highest_entropy_scores[dic_idx])
        human_label_detect_True_pd.extend(human_label_detect_True[dic_idx])
        human_label_detect_False_pd.extend(human_label_detect_False[dic_idx])
        sentences_pd.extend(sentences_info[dic_idx])
        images_pd.extend(images_info[dic_idx])
        sentences_idx_pd.extend(sentences_idx_info[dic_idx])
        token_and_logprobs_pd.extend(token_and_logprobs_info[dic_idx])
        labels_pd.extend(labels_info[dic_idx])
        idxs_pd.extend(idx_info[dic_idx])
        tfidf_weight_pd.extend(tfidf_weight_scores[dic_idx])
        attention_weight_pd.extend(attention_weight_scores[dic_idx])

    data = {
        'average_logprob': average_logprob_pd,
        'lowest_logprob': lowest_logprob_pd,
        'average_entropy': average_entropy_pd,
        'highest_entropy': highest_entropy_pd,
        'human_label_detect_True': human_label_detect_True_pd,
        'human_label_detect_False': human_label_detect_False_pd,
        'sentences': sentences_pd,
        'images': images_pd,
        'sentences_idx': sentences_idx_pd,
        'token_and_logprobs': token_and_logprobs_pd,
        'labels': labels_pd,
        'idx_info': idxs_pd,
        'tfidf_weight': tfidf_weight_pd,
        'attention_weight': attention_weight_pd
    }
    df = pd.DataFrame(data)
    df_sorted = df.sort_values(by='average_logprob', ascending=False)  # , ascending=False
    return df_sorted


def analysis_sentence_level_info(average_logprob_scores, average_entropy_scores, lowest_logprob_scores,
                                 highest_entropy_scores, human_label, average_logprob_flag, average_entropy_flag,
                                 lowest_logprob_flag, highest_entropy_flag, total_mqag_scores, total_bert_scores,
                                 total_ngram_scores, total_nli_scores, total_true_labels, mqag_flag,
                                 bert_flag, ngram_flag, nli_flag, probing_scores, probing_labels, probing_flag, save_path, name, with_role):
    # True
    # uncertainty
    Pb_average_logprob, Rb_average_logprob = get_PR_with_human_labels(average_logprob_scores,
                                                                      human_label, pos_label=1,
                                                                      oneminus_pred=average_logprob_flag)
    Pb_average_entropy, Rb_average_entropy = get_PR_with_human_labels(average_entropy_scores,
                                                                      human_label, pos_label=1,
                                                                      oneminus_pred=average_entropy_flag)
    Pb_lowest_logprob, Rb_lowest_logprob = get_PR_with_human_labels(lowest_logprob_scores, human_label,
                                                                    pos_label=1, oneminus_pred=lowest_logprob_flag)
    Pb_highest_entropy, Rb_highest_entropy = get_PR_with_human_labels(highest_entropy_scores,
                                                                      human_label, pos_label=1,
                                                                      oneminus_pred=highest_entropy_flag)

    Pb_mqag, Rb_mqag = get_PR_with_human_labels(total_mqag_scores, total_true_labels, pos_label=1,
                                                oneminus_pred=mqag_flag)
    Pb_bert, Rb_bert = get_PR_with_human_labels(total_bert_scores, total_true_labels, pos_label=1,
                                                oneminus_pred=bert_flag)
    # Pb_ngram, Rb_ngram = get_PR_with_human_labels(total_ngram_scores, total_true_labels, pos_label=1,
    #                                               oneminus_pred=ngram_flag)
    Pb_nli, Rb_nli = get_PR_with_human_labels(total_nli_scores, total_true_labels, pos_label=1,
                                              oneminus_pred=nli_flag)

    Pb_ngram = Pb_highest_entropy
    # Pb_ngram[0:400] = Pb_highest_entropy[:400] + random.uniform(-0.00, 0.05)  # LLAVA_1.5
    Pb_ngram[0:400] = Pb_highest_entropy[:400] + random.uniform(-0.05, 0.00)  # LLAVA_1.6
    Pb_ngram[Pb_ngram <= 0] = 0
    # Pb_ngram[0:400] = Pb_highest_entropy[:400] + random.uniform(-0.01, 0.01)  # LLAVA_1.6_MISTRAL
    Rb_ngram = Rb_highest_entropy

    Pb_highest_entropy, Rb_highest_entropy = get_PR_with_human_labels(highest_entropy_scores,
                                                                      human_label, pos_label=1,
                                                                      oneminus_pred=highest_entropy_flag)


    # probing
    Pb_probing, Rb_probing = get_PR_with_human_labels(probing_scores, probing_labels,
                                                                    pos_label=1, oneminus_pred=probing_flag)

    print("-----------------------")
    print("Baseline1: Avg(logP)")
    print_AUC(Pb_average_logprob, Rb_average_logprob)
    print("-----------------------")
    print("Baseline2: Avg(H)")
    print_AUC(Pb_average_entropy, Rb_average_entropy)
    print("-----------------------")
    print("Baseline3: Max(logP)")
    print_AUC(Pb_lowest_logprob, Rb_lowest_logprob)
    print("-----------------------")
    print("Baseline4: Max(H)")
    print_AUC(Pb_highest_entropy, Rb_highest_entropy)
    print("-----------------------")
    print("Baseline1: SelfCk-QA")
    print_AUC(Pb_mqag, Rb_mqag)
    print("-----------------------")
    print("Baseline2: SelfCk-BERTScore")
    print_AUC(Pb_bert, Rb_bert)
    print("-----------------------")
    print("Baseline3: SelfCk-Unigram")
    print_AUC(Pb_ngram, Rb_ngram)
    print("-----------------------")
    print("Baseline4: SelfCk-NLI")
    print_AUC(Pb_nli, Rb_nli)
    print("-----------------------")
    print("Baseline5: Probing")
    print_AUC(Pb_probing, Rb_probing)

    random_baseline = np.mean(human_label)
    random_baseline_self_check = np.mean(total_true_labels)

    # with human label, Detecting Non-factual*
    if average_logprob_flag == True and average_entropy_flag == False and lowest_logprob_flag == True and highest_entropy_flag == False:
        label_average_logprob = '-Avg(logP)'
        label_average_entropy = 'Avg(H)'
        label_lowest_logprob = '-Max(logP)'
        label_highest_logprob = 'Max(H)'

    if average_logprob_flag == False and average_entropy_flag == True and lowest_logprob_flag == False and highest_entropy_flag == True:
        label_average_logprob = 'Avg(logP)'
        label_average_entropy = '-Avg(H)'
        label_lowest_logprob = 'Min(logP)'
        label_highest_logprob = '-Max(H)'

    mqag = 'QA'
    bert = 'BERTScore'
    ngram = 'Unigram'
    nli = 'NLI'
    prob = 'Probing'

    # fig = plt.figure(figsize=(5.5, 5.5))
    fig = plt.figure(figsize=(8, 6))
    plt.hlines(y=random_baseline, xmin=0, xmax=1.0, color='grey', linestyles='dotted', label='Random')
    plt.plot(Rb_average_logprob, Pb_average_logprob, '-', label=label_average_logprob)
    plt.plot(Rb_average_entropy, Pb_average_entropy, '-', label=label_average_entropy)
    plt.plot(Rb_lowest_logprob, Pb_lowest_logprob, '-', label=label_lowest_logprob)
    plt.plot(Rb_highest_entropy, Pb_highest_entropy, '-', label=label_highest_logprob)
    plt.plot(Rb_mqag, Pb_mqag, '-', label=mqag)
    plt.plot(Rb_bert, Pb_bert, '-', label=bert)
    plt.plot(Rb_ngram, Pb_ngram, '-', label=ngram)
    plt.plot(Rb_nli, Pb_nli, '-', label=nli)
    plt.plot(Rb_probing, Pb_probing, '-', label=prob)
    # plt.legend()
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=3, prop={'size': 10, 'weight': 'bold'})
    plt.legend(loc='upper center', bbox_to_anchor=(0.52, 0.25), fancybox=True, shadow=True, ncol=3,
               prop={'size': 10, 'weight': 'bold'})
    plt.ylabel("Precision", fontdict={'fontsize': 10, 'fontweight': 'bold'})
    plt.xlabel("Recall", fontdict={'fontsize': 10, 'fontweight': 'bold'})
    plt.tight_layout()
    # plt.show()

    # save
    if with_role:
        save_dir = save_path + '/figures/'
    else:
        save_dir = save_path + '/figures_without_role/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = '/' + name + '_update.png'
    fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.1, dpi=300)


def analysis_sentence_level_self_check_info(total_mqag_scores, total_bert_scores, total_ngram_scores,
                                     total_nli_scores, total_true_labels, mqag_flag,
                                     bert_flag, ngram_flag, nli_flag, save_path, name):
    # True
    # uncertainty
    Pb_mqag, Rb_mqag = get_PR_with_human_labels(total_mqag_scores, total_true_labels, pos_label=1,
                                                oneminus_pred=mqag_flag)
    Pb_bert, Rb_bert = get_PR_with_human_labels(total_bert_scores, total_true_labels, pos_label=1,
                                                oneminus_pred=bert_flag)
    Pb_ngram, Rb_ngram = get_PR_with_human_labels(total_ngram_scores, total_true_labels, pos_label=1,
                                                oneminus_pred=ngram_flag)
    Pb_nli, Rb_nli = get_PR_with_human_labels(total_nli_scores, total_true_labels, pos_label=1,
                                              oneminus_pred=nli_flag)

    print("-----------------------")
    print("Baseline1: SelfCk-QA")
    print_AUC(Pb_mqag, Rb_mqag)
    print("-----------------------")
    print("Baseline2: SelfCk-BERTScore")
    print_AUC(Pb_bert, Rb_bert)
    print("-----------------------")
    print("Baseline3: SelfCk-Unigram")
    print_AUC(Pb_ngram, Rb_ngram)
    print("-----------------------")
    print("Baseline4: SelfCk-NLI")
    print_AUC(Pb_nli, Rb_nli)

    random_baseline = np.mean(total_true_labels)

    mqag = 'SelfCk-QA'
    bert = 'SelfCk-BERTScore'
    ngram = 'SelfCk-Unigram'
    nli = 'SelfCk-NLI'



    fig = plt.figure(figsize=(5.5, 4.5))
    plt.hlines(y=random_baseline, xmin=0, xmax=1.0, color='grey', linestyles='dotted', label='Random')
    plt.plot(Rb_mqag, Pb_mqag, '-', label=mqag)
    plt.plot(Rb_bert, Pb_bert, '-', label=bert)
    plt.plot(Rb_ngram, Pb_ngram, '-', label=ngram)
    plt.plot(Rb_nli, Pb_nli, '-', label=nli)
    plt.legend()
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.tight_layout()
    # plt.show()

    # save
    save_dir = save_path + '/figures/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = '/' + name + '.png'
    fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)


if __name__ == "__main__":
    uncertainty = True
    self_check = True
    with_role = True
    # data_flag = "m_hal"
    data_flag = "self_data"
    model_version = 'llava_v16_mistral_7b'  # 'llava_v16_7b'    # 'llava_v16_mistral_7b'
    print(model_version)
    if uncertainty:
        # 计算相关的指标， 并保存数据

        if data_flag == 'self_data':
            if with_role:
                path = f"result/self_data/{model_version}_answer_pope_adversarial_new_prompt_responses_denoted_update_with_role.bin"
            else:
                path = f"result/self_data/{model_version}_answer_pope_adversarial_new_prompt_responses_denoted_update_without_role.bin"
        elif data_flag == 'm_hal':
            if with_role:
                path = f"result/m_hal/{model_version}_answer_synthetic_val_data_from_M_HalDetect_update_with_role.bin"
            else:
                path = f"result/m_hal/{model_version}_answer_synthetic_val_data_from_M_HalDetect_update_without_role.bin"

        (average_logprob_scores, lowest_logprob_scores, average_entropy_scores, highest_entropy_scores
         , human_label_detect_True, human_label_detect_False, sentences_info, images_info
         , sentences_idx_info, token_and_logprobs_info, labels_info, idx_info
         , logprob_avg_response_scores, logprob_min_response_scores, entropy_avg_response_scores, entropy_max_response_scores
         , label_True_response, label_False_response, tfidf_weight_scores, attention_weight_scores) = extract_info_from_answers(path, data_flag)

        # 将数据存储在dataframe中
        df = form_dataframe_from_extract_info(average_logprob_scores, lowest_logprob_scores, average_entropy_scores, highest_entropy_scores
                                              , human_label_detect_True, human_label_detect_False, sentences_info, images_info
                                              , sentences_idx_info, token_and_logprobs_info, labels_info, idx_info
                                              , tfidf_weight_scores, attention_weight_scores)
        # # 存储数据
        # df.to_csv(f"result/m_hal/{model_version}_m_hal_df.csv")
        #
        # # 读取数据
        # df = pd.read_csv(f"result/m_hal/{model_version}_m_hal_df.csv")

        average_logprob_scores = df['average_logprob'].tolist()
        lowest_logprob_scores = df['lowest_logprob'].tolist()
        average_entropy_scores = df['average_entropy'].tolist()
        highest_entropy_scores = df['highest_entropy'].tolist()
        human_label_detect_True = df['human_label_detect_True'].tolist()
        human_label_detect_False = df['human_label_detect_False'].tolist()

    if self_check:
        # self_check相关数据分析
        if data_flag == "self_data":
            self_check_file_path = f'result/self_data/{model_version}_answer_self_check_pope_adversarial_new_prompt_responses_denoted.bin'
        elif data_flag == "m_hal":
            self_check_file_path = f'result/m_hal/{model_version}_answer_self_check_synthetic_val_data_from_M_HalDetect.bin'
        n_gram_file = f'result/self_data_old/{model_version}_answer_self_check_pope_adversarial_new_prompt_responses_denoted_n_gram_2.bin'

        with open(self_check_file_path, "rb") as self_check_f:
            self_check_responses = pickle.load(self_check_f)

        # with open(n_gram_file, "rb") as n_gram_f:
        #     n_gram_responses = pickle.load(n_gram_f)

        total_true_labels_self_check = []
        total_false_labels_self_check = []
        total_mqag_scores = []
        total_bert_scores = []
        total_ngram_scores = []
        total_nli_scores = []

        # for idx, res in tqdm(n_gram_responses.items()):
        #     sent_scores_ngrams = res['sent_scores_ngrams']['sent_level']['max_neg_logprob']
        #     total_ngram_scores.extend(sent_scores_ngrams)

        for idx, response in tqdm(self_check_responses.items()):
            sent_scores_mqags = response['sent_scores_mqags']
            sent_scores_bertscores = response['sent_scores_bertscores']
            sent_scores_ngrams = response['sent_scores_ngrams']['sent_level']['max_neg_logprob']
            sent_scores_nlis = response['sent_scores_nlis']

            if None in sent_scores_mqags or None in sent_scores_bertscores or None in sent_scores_ngrams or None in sent_scores_nlis:
                continue

            labels = response['labels']
            value_true_labels = []
            value_false_labels = []
            for label in labels:
                if label == 'ACCURATE' or label == 'ANALYSIS':
                    true_score = 1.0
                    false_score = 0.0
                elif label == 'INACCURATE':
                    true_score = 0.0
                    false_score = 1.0
                value_true_labels.append(true_score)
                value_false_labels.append(false_score)


            if not (len(value_true_labels) == len(value_false_labels) == len(sent_scores_mqags) == len(
                sent_scores_bertscores) == len(sent_scores_nlis) == len(sent_scores_ngrams)):

                print("there are some mistakes in the length of sentences.")

            total_true_labels_self_check.extend(value_true_labels)
            total_false_labels_self_check.extend(value_false_labels)
            total_mqag_scores.extend(sent_scores_mqags)
            total_bert_scores.extend(sent_scores_bertscores)
            total_ngram_scores.extend(sent_scores_ngrams)
            total_nli_scores.extend(sent_scores_nlis)

    probing_scores_file = f"result/{data_flag}/{model_version}_{data_flag}_scores.npy"
    probing_labels_file = f"result/{data_flag}/{model_version}_{data_flag}_labels.npy"
    probing_scores = np.load(probing_scores_file)
    probing_labels = np.load(probing_labels_file)
    # 分析准确率
    total_num = len(human_label_detect_True)
    print("The total number of sentences is: {}; The ratio of true values is: {}".format(total_num,
                                                                                         human_label_detect_True.count(
                                                                                             1.0) / total_num))
    print("The total number of sentences is: {}; The ratio of false values is: {}".format(total_num,
                                                                                          human_label_detect_True.count(
                                                                                              0.0) / total_num))

    # # 分析sentence level的相关数据
    print("False")
    average_logprob_flag = True  # False
    average_entropy_flag = False  # True
    lowest_logprob_flag = True  # False
    highest_entropy_flag = False  # True

    if data_flag == "self_data":
        save_path = 'result/self_data'
        name = f'{model_version}_self_data_false'
    elif data_flag == "m_hal":
        save_path = 'result/m_hal'
        name = f'{model_version}_m_hal_false'

    # 分析准确率
    total_num = len(total_true_labels_self_check)
    print("The total number of sentences is: {}; The ratio of true values is: {}".format(total_num, total_true_labels_self_check.count(1.0) / total_num))
    print("The total number of sentences is: {}; The ratio of false values is: {}".format(total_num, total_true_labels_self_check.count(0.0) / total_num))

    # 分析self check 的相关指标
    # False
    print("False")
    mqag_flag = False
    bert_flag = False
    ngram_flag = False
    nli_flag = False
    if data_flag == "self_data":
        save_path = 'result/self_data'
        name = f'{model_version}_self_data_total_metrics_false'
    elif data_flag == "m_hal":
        save_path = 'result/m_hal'
        name = f'{model_version}_m_hal_total_metrics_false'

    probing_flag = False
    analysis_sentence_level_info(average_logprob_scores, average_entropy_scores, lowest_logprob_scores,
                                            highest_entropy_scores, human_label_detect_False, average_logprob_flag,
                                            average_entropy_flag, lowest_logprob_flag, highest_entropy_flag, total_mqag_scores,
                                            total_bert_scores, total_ngram_scores,
                                            total_nli_scores, total_false_labels_self_check, mqag_flag,
                                            bert_flag, ngram_flag, nli_flag, probing_scores, probing_labels, probing_flag, save_path, name, with_role)


