import pandas as pd
from PIL import Image
from googletrans import Translator, LANGUAGES
import json



if __name__ == "__main__":
    new_file_name = f"../../../result/self_data/llava_v15_7b_pope_adversarial_new_prompt_responses_denoted.json"
    new_file = open(new_file_name, "w")

    file_path = 'llava_v15_7b_pope_adversarial_new_prompt_responses_denoted.csv'
    df = pd.read_csv(file_path)

    diff_count = 0
    count = 0
    for index, row in df.iterrows():
        image_id = row['image_id']
        question = row['prompt']
        passage = row['passage']
        sentence = row['sentence']
        model = row['model_id']
        human_label = row['human_label']
        gpt_label = row['gpt_label']
        if human_label == 'accurate':
            label = 'ACCURATE'
        else:
            label = 'INACCURATE'

        if human_label != gpt_label:
            diff_count += 1
            # print(index, image_id, sentence, human_label, gpt_label)
            print(index, image_id)
            print(sentence)
            translator = Translator()
            translated_text = translator.translate(sentence, src='en', dest='zh-cn')
            print(translated_text)
            print(human_label)
            print(gpt_label)
            image_path = "../../../data/val2014/" + image_id
            img = Image.open(image_path)
            img.show()

        count += 1
    print(count, diff_count, diff_count/count)


    idx = 0
    for name, group in df.groupby('image_id'):
        image_id = name
        sentences = []
        labels = []
        for index, row in group.iterrows():
            question = row['prompt']
            passage = row['passage']
            sentence = row['sentence']
            model = row['model_id']
            human_label = row['human_label']
            gpt_label = row['gpt_label']
            if human_label == 'accurate':
                label = 'ACCURATE'
            else:
                label = 'INACCURATE'
            sentences.append(sentence)
            labels.append(label)

        new_file.write(json.dumps({
            "question_id": idx,
            "image": image_id,
            "question": question,
            "response": passage,
            "labels": labels,
            "sentences": sentences,
            "generate_response_model": model
        }) + "\n")
        idx += 1
