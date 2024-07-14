import json
import pandas as pd
import argparse
import os
from strongreject_evaluator import autograde_response
import sys
import re


if __name__ == '__main__':
    # input_file = 'result/safebench/llava-v1.5-7b_safebench_origin_model_annocated.json'
    # count = 0
    # strong_score = 0.0
    # with open(input_file, 'r') as file:
    #     for line in file:
    #         line_info = json.loads(line)
    #         sub_strong_score = line_info['strong_score']
    #         strong_score += sub_strong_score
    #         count += 1
    #
    # print(strong_score/count)
    """
    res = {}
    total_count = 0
    # input_file = "result/safebench/llava-v1.5-7b_safebench_origin_model_annocated_update.json"
    # input_file = 'result/safebench/llava_v1.5_7b_safebench_multiplier_1.5_layer_annocated_update.json'
    # input_file = 'result/safebench/llava_v1.5_7b_safebench_multiplier_5.0_head_annocated_update.json'
    # input_file = 'result/safebench/llava_v1.5_7b_safebench_multiplier_3.0_layer_from_probe_weight_direction_annocated_update.json'
    # input_file = 'result/safebench/llava_v1.5_7b_safebench_multiplier_0.5_layer_9_annocated_update.json'
    # input_file = 'result/safebench/llava_v1.5_7b_safebench_multiplier_1.0_layer_9_annocated_update.json'
    # input_file = 'result/safebench/llava_v1.5_7b_safebench_multiplier_1.5_layer_9_annocated_update.json'
    # input_file = 'result/safebench/llava_v1.5_7b_safebench_multiplier_2.0_layer_9_annocated_update.json'
    # input_file = 'result/safebench/llava_v1.5_7b_safebench_multiplier_0.5_layer_31_annocated_update.json'
    # input_file = "result/vlguard/llava_v1.5_7b_vlguard_test_su_multiplier_2.0_layer_9_annocated_update.json"
    # input_file = "result/vlguard/llava_v1.5_7b_vlguard_test_u_multiplier_2.0_layer_9_annocated_update.json"
    input_file = 'result/vlguard/llava_v1.5_7b_vlguard_test_su_multiplier_2.5_layer_9_from_probe_weight_direction_annocated_update.json'
    with open(input_file, 'r') as file:
        for line in file:
            line_info = json.loads(line)
            gpt4_score = line_info['gpt4_score']
            match = re.search(r'\[([0-9]+\.[0-9]+)\]', gpt4_score)

            if match:
                type_value = match.group(1)
                print("Type:", type_value)
                if type_value in res:
                    res[type_value] += 1
                else:
                    res[type_value] = 1
                total_count += 1
            else:
                print("Type not found in the text.")
    print(f"Total count: {total_count}")
    qingli = 3
    """

    #  对Science_qa 数据进行分析

    # response_file = "result/science_qa/llava_v1.5_7b_science_qa_origin_model.json"
    # response_file = "result/science_qa/llava_v1.5_7b_science_qa_multiplier_2.0_layer_9.json"
    # response_file = 'result/science_qa/llava_v1.5_7b_science_qa_multiplier_2.5_layer_9_pwd.json'
    response_file = 'result/science_qa/llava_v1.5_7b_science_qa_multiplier_2.0_layer_14.json'
    answer_file = "playground/data/science_qa/ready_science_qa.json"

    res_dict = {}
    with open(response_file, 'r') as file:
        for line in file:
            line_info = json.loads(line)
            response = line_info['response']
            idx = line_info['idx']
            res_dict[idx] = response

    answer_dict = {}
    with open(answer_file, 'r') as file:
        for line in file:
            line_info = json.loads(line)
            idx = line_info['idx']
            answer = line_info['answer']
            answer_number = line_info['answer_number']
            answer_dict[idx] = answer

    correct_count = 0
    for idx in range(0, 502):
        answer = answer_dict[idx].lower()
        response = res_dict[idx]
        match = re.search(r'Answer:\s*(.*)', response)
        if match:
            extract_response = (match.group(1).strip()).lower()
        else:
            extract_response = response.lower()

        if (answer == extract_response) or (answer in extract_response) or (extract_response in answer):
            correct_count += 1
        # elif extract_response == 'answer':
        #     correct_count += 1
        else:
            print("answer:")
            print(answer)
            print("extract_response:")
            print(extract_response)

    print(f"The correct ratio is {correct_count/502*100:.2f}%")



