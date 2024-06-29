import pandas as pd
import os
import json


def convert_windows_to_ubuntu_path(windows_path):
    # Replace backslashes with forward slashes
    ubuntu_path = windows_path.replace('\\', '/')

    # If the path starts with a drive letter (e.g., C:), remove it for Ubuntu
    if ubuntu_path[1] == ':':
        ubuntu_path = ubuntu_path[2:]

    return ubuntu_path
def get_mmsafebench(num_per_type=None):
    json_folder = os.path.join("data", "mm_safebench", "processed_questions")
    img_folder = os.path.join("data", "mm_safebench", "imgs")
    tgt_folder = os.path.join("data", "mm_safebench", "samples")
    if not os.path.exists(tgt_folder):
        os.makedirs(tgt_folder)
    # if num_per_type == None:
    tgt_csv = os.path.join(tgt_folder, f"samples_full.csv")
    types = os.listdir(img_folder)

    df_output = pd.DataFrame()

    image_ll, question_ll = [], []
    for type in types:
        json_file = os.path.join(json_folder, f"{type}.json")
        json_data = json.load(open(json_file))
        print(type, len(json_data))

        cnt_samples = 0
        for i in json_data:
            for j in range(3):
                if j == 0: # typo
                    ques = json_data[i]["Rephrased Question(SD)"]
                    image = os.path.join(img_folder, type, "TYPO", f"{i}.jpg")

                elif j == 1: # sd
                    ques = json_data[i]["Rephrased Question(SD)"]
                    image = os.path.join(img_folder, type, "SD", f"{i}.jpg")

                elif j == 2: # typo + sd
                    ques = json_data[i]["Rephrased Question"]
                    image = os.path.join(img_folder, type, "SD_TYPO", f"{i}.jpg")

                # print(f"Ubuntu path: {convert_windows_to_ubuntu_path(image)}")
                # print(f"image {image} exist: {os.path.exists(image)}")

                image_ll.append(image)
                question_ll.append(ques)

    df_output["image_path"] = image_ll
    df_output["question"] = question_ll

    df_output.to_csv(tgt_csv, index=False)

def get_mmsafebench_sample(num_per_type=None):
    json_folder = os.path.join("data", "mm_safebench", "processed_questions")
    img_folder = os.path.join("data", "mm_safebench", "imgs")
    tgt_folder = os.path.join("data", "mm_safebench", "samples")
    if not os.path.exists(tgt_folder):
        os.makedirs(tgt_folder)
    # if num_per_type == None:
    tgt_csv = os.path.join(tgt_folder, f"samples.csv")
    types = os.listdir(img_folder)

    df_output = pd.DataFrame()

    image_ll, question_ll = [], []

    # type_indices = [list()]
    total_json_data = dict()
    total_json_indices = dict()
    min_len = 100

    for type in types:
        json_file = os.path.join(json_folder, f"{type}.json")
        json_data = json.load(open(json_file))
        print(type, len(json_data))
        total_json_data[type] = json_data
        total_json_indices[type] = list(json_data.keys())
        min_len = min(min_len, len(total_json_indices[type]))

    for i in range(min_len):
        for type in types:
            idx = str(total_json_indices[type][i])
            for j in range(3):
                if j == 0: # typo
                    ques = total_json_data[type][idx]["Rephrased Question(SD)"]
                    image = os.path.join(img_folder, type, "TYPO", f"{idx}.jpg")

                elif j == 1: # sd
                    ques = total_json_data[type][idx]["Rephrased Question(SD)"]
                    image = os.path.join(img_folder, type, "SD", f"{idx}.jpg")

                elif j == 2: # typo + sd
                    ques = total_json_data[type][idx]["Rephrased Question"]
                    image = os.path.join(img_folder, type, "SD_TYPO", f"{idx}.jpg")

                # print(f"Ubuntu path: {convert_windows_to_ubuntu_path(image)}")
                # print(f"image {image} exist: {os.path.exists(image)}")

                image_ll.append(image)
                question_ll.append(ques)

    df_output["image_path"] = image_ll
    df_output["question"] = question_ll

    df_output.to_csv(tgt_csv, index=False)

def get_gpt4v_gen(prompt):
    pass




if __name__ == "__main__":
    num_per_type = 44
    get_mmsafebench_sample(num_per_type)


    #
    #
    # # Example usage
    # windows_path = r"C:\Users\Username\Documents\file.txt"
    # ubuntu_path = convert_windows_to_ubuntu_path(windows_path)
    #
    # print("Windows Path:", windows_path)
    # print("Ubuntu Path:", ubuntu_path)



