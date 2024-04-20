import numpy as np
import json
import re




if __name__ == "__main__":

    # style = "adversarial"
    # new_file_name = "../../../playground/data/coco2014_val_qa_eval/coco_pope_adversarial_extract_info.json"
    # file_path = "../../../playground/data/coco2014_val_qa_eval/coco_pope_adversarial.json"
    new_file_name = "../../../playground/data/coco2014_val_qa_eval/coco_pope_random_without_period_extract_info.json"
    file_path = "../../../playground/data/coco2014_val_qa_eval/coco_pope_random.json"
    new_file = open(new_file_name, "w")
    idx = 0
    with open(file_path, 'r') as f:
        for file in f.readlines():
            dic = json.loads(file)
            answer = dic['label'].lower()
            image = dic['image']
            question = dic['text']
            original_question = dic['question_id']
            question_id = idx
            sentences = []
            labels = []

            if answer == 'no':
                response = 'No'
                label = 'ACCURATE'
            elif answer == 'yes':
                response = 'Yes'
                label = 'ACCURATE'
            sentences.append(response)
            labels.append(label)

            new_file.write(json.dumps({
                "question_id": question_id,
                "image": image,
                "question": question,
                "response": response,
                "answer": answer,
                "labels": labels,
                "sentences": sentences,
                "original_question": original_question
            }) + "\n")
            idx += 1

            question_id = idx
            if answer == 'no':
                response = 'Yes'
                label = 'INACCURATE'
            elif answer == 'yes':
                response = 'No'
                label = 'INACCURATE'
            sentences = []
            labels = []
            sentences.append(response)
            labels.append(label)

            new_file.write(json.dumps({
                "question_id": question_id,
                "image": image,
                "question": question,
                "response": response,
                "answer": answer,
                "labels": labels,
                "sentences": sentences,
                "original_question": original_question
            }) + "\n")
            idx += 1
            new_file.flush()


    new_file.close()


