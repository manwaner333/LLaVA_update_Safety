import json
import csv
import os


if __name__ == "__main__":
    # step1: 选择出初步筛选出相关的数据
    # model_response_dict = {}
    # train_response_from_model = 'playground/data/vlguard/vlguard_train_origin_model_filter.csv'
    # with open(train_response_from_model, mode='r', newline='') as file:
    #     reader = csv.reader(file)
    #     for row in reader:
    #         if row == ['idx', 'id', 'image', 'safe', 'harmful_category', 'harmful_subcategory', 'prompt', 'response', 'prompt_translate', 'answer_translate']:
    #             continue
    #         else:
    #             idx = row[0]
    #             id = row[1]
    #             key = "idx_" + idx + "id_" + id
    #             model_response_dict[key] = [row[7], row[9]]
    #
    # filter_ids = [0, 5, 9, 12, 20, 25, 31, 34, 41, 45, 47, 50, 54, 55, 60, 62, 65, 70,  76, 78,
    #  83, 84, 91, 93, 101, 107, 117, 120, 128, 130, 136, 138, 141, 145, 146, 161, 163, 165, 169, 173,
    #  182, 190, 200, 201, 206, 208, 210, 215, 216, 226, 235, 255, 260, 263, 267, 274, 275, 279, 280, 281,
    #  285, 290, 294, 297, 298, 303, 305, 314, 342, 350, 357, 358, 360, 335, 372, 380, 386, 391, 402, 405,
    #  409, 412, 434, 451, 465, 469, 483, 485, 492, 497, 502, 513, 514, 520, 537, 543, 544, 546, 547, 551,
    #  554, 556, 561, 567, 569, 577, 581, 587, 590, 596, 598, 605, 610, 621, 622, 625, 628, 630,
    #  639, 644, 650, 654, 659, 667, 674, 675, 678, 680, 681, 683, 686, 694, 697, 707, 719, 727, 730, 732,
    #  745, 746, 750, 755, 756, 758, 767, 769, 771, 779, 787, 790, 791, 796, 802, 804, 808, 812, 814, 815,
    #  822, 823, 831, 838, 847, 856, 864, 886, 900, 923, 938, 973, 978, 984, 992, 1004, 1022, 1029, 1058,
    #  1060, 1073, 1080, 1122, 1125, 1137, 1145, 1182, 1187, 1192, 1138, 1207, 1217, 1218]
    #
    # output_file = 'playground/data/vlguard/vlguard_confrontational_cases.csv'
    # if os.path.exists(output_file):
    #     os.remove(output_file)
    #
    # with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
    #     writer = csv.writer(outfile)
    #
    #     ready_train_file = 'playground/data/vlguard/ready_train.json'
    #     with open(ready_train_file, 'r') as file:
    #         for line in file:
    #             line_info = json.loads(line)
    #             idx = line_info['idx']
    #             id = line_info['id']
    #             image = line_info['image']
    #             safe = line_info['safe']
    #             harmful_category = line_info['harmful_category']
    #             harmful_subcategory = line_info['harmful_subcategory']
    #             prompt = line_info['prompt']
    #             data_response = line_info['response']
    #             if idx in filter_ids:
    #                 key = "idx_" + str(idx) + "id_" + id
    #                 print(key)
    #                 model_response_en = model_response_dict[key][0]
    #                 model_response_ch = model_response_dict[key][1]
    #                 writer.writerow([idx, id, image, safe, harmful_category, harmful_subcategory, prompt, data_response
    #                                     , model_response_en, model_response_ch])


    # step2: 进一步筛选出相关的数据
    model_response_dict = {}
    train_response_from_model = 'playground/data/vlguard/vlguard_train_origin_model_filter.csv'
    with open(train_response_from_model, mode='r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            if row == ['idx', 'id', 'image', 'safe', 'harmful_category', 'harmful_subcategory', 'prompt', 'response', 'prompt_translate', 'answer_translate']:
                continue
            else:
                idx = row[0]
                id = row[1]
                key = "idx_" + idx + "id_" + id
                model_response_dict[key] = [row[7], row[9]]
    filter_ids = [0, 5, 9, 12, 20, 25, 31, 34, 41, 45, 47, 50, 54, 55, 60, 62, 65, 70, 76, 78,
                  83, 84, 91, 93, 101, 107, 117, 120, 128, 130, 136, 138, 141, 145, 146, 161, 163, 165, 169, 173,
                  182, 190, 200, 201, 206, 208, 210, 215, 216, 226, 235, 255, 260, 263, 267, 274, 275, 279, 280, 281,
                  285, 290, 294, 297, 298, 303, 305, 314, 342, 350, 357, 358, 360, 335, 372, 380, 386, 391, 402, 405,
                  409, 412, 434, 451, 465, 469, 483, 485, 492, 497, 502, 513, 514, 520, 537, 543, 544, 546, 547, 551,
                  554, 556, 561, 567, 569, 577, 581, 587, 590, 596, 598, 605, 610, 621, 622, 625, 628, 630,
                  639, 644, 650, 654, 659, 667, 674, 675, 678, 680, 681, 683, 686, 694, 697, 707, 719, 727, 730, 732,
                  745, 746, 750, 755, 756, 758, 767, 769, 771, 779, 787, 790, 791, 796, 802, 804, 808, 812, 814, 815,
                  822, 823, 831, 838, 847, 856, 864, 886, 900, 923, 938, 973, 978, 984, 992, 1004, 1022, 1029, 1058,
                  1060, 1073, 1080, 1122, 1125, 1137, 1145, 1182, 1187, 1192, 1138, 1207, 1217, 1218]

    filter_filter_ids = [91, 101, 128,  130, 138, 141, 145, 146, 161, 163, 165, 169, 173, 190, 201, 206, 210, 215, 226,
                         235, 255, 267, 275, 280, 285,  294, 303, 305, 335, 342, 357, 358, 360, 372, 380, 386, 409,  434,
                         485,  492,  497, 537, 543, 561, 587, 605,  610, 659, 680, 771, 886, 938, 1125, 1137, 1138, 1218]

    # output_file = "playground/data/vlguard/vlguard_confrontational_cases_filter_pos.json"
    output_file = "playground/data/vlguard/vlguard_confrontational_cases_filter_neg.json"
    if os.path.exists(output_file):
        os.remove(output_file)
    ready_train_file = 'playground/data/vlguard/ready_train.json'

    with open(output_file, 'w') as file:
        with open(ready_train_file, 'r') as f:
            for line in f:
                line_info = json.loads(line)
                idx = line_info['idx']
                id = line_info['id']
                image = line_info['image']
                safe = line_info['safe']
                harmful_category = line_info['harmful_category']
                harmful_subcategory = line_info['harmful_subcategory']
                prompt = line_info['prompt']
                data_response = line_info['response']
                if idx in filter_ids and idx not in filter_filter_ids:
                    key = "idx_" + str(idx) + "id_" + id
                    print(key)
                    model_response_en = model_response_dict[key][0]
                    model_response_ch = model_response_dict[key][1]
                    # answer = prompt + " ASSISTANT: " + data_response
                    answer = prompt + " ASSISTANT: " + model_response_en
                    json.dump({'idx': idx, 'id': id, 'image': image, "safe": safe,
                               "harmful_category": harmful_category,
                               "harmful_subcategory": harmful_subcategory,
                               'question': answer}, file)
                    # json.dump({'idx': idx, 'id': id, 'image': image, "safe": safe,
                    #            "harmful_category": harmful_category,
                    #            "harmful_subcategory": harmful_subcategory,
                    #            'prompt': prompt, 'pos_response': data_response, 'neg_reponse': model_response_en}, file)
                    file.write('\n')
                    idx += 1


