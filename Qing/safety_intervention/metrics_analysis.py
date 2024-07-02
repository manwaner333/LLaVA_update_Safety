import json
import pandas as pd
import argparse
import os
from strongreject_evaluator import autograde_response
import sys



if __name__ == '__main__':
    input_file = 'result/safebench/llava-v1.5-7b_safebench_origin_model_annocated.json'
    count = 0
    strong_score = 0.0
    with open(input_file, 'r') as file:
        for line in file:
            line_info = json.loads(line)
            sub_strong_score = line_info['strong_score']
            strong_score += sub_strong_score
            count += 1

    print(strong_score/count)

