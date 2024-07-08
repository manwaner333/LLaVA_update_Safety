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
    res = {}
    total_count = 0
    # input_file = "result/safebench/llava-v1.5-7b_safebench_origin_model_annocated_update.json"
    # input_file = 'result/safebench/llava_v1.5_7b_safebench_multiplier_1.5_layer_annocated_update.json'
    # input_file = 'result/safebench/llava_v1.5_7b_safebench_multiplier_5.0_head_annocated_update.json'
    input_file = 'result/safebench/llava_v1.5_7b_safebench_multiplier_3.0_layer_from_probe_weight_direction_annocated_update.json'
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




    # print(f"count: {count/500}")


