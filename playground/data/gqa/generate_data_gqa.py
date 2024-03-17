import numpy as np
import json
import re


def split_sentences_with_index(text):
    # Define a regular expression pattern for matching sentence endings
    # and keep the punctuations after each sentence.
    sentence_endings = r'(?<=[.!?])\s*(?=\b|\s)'

    # Use the re.finditer() function to find all occurrences of sentence endings
    matches = re.finditer(sentence_endings, text)

    # Store the indices of the first character of each sentence
    indices = [0]  # Start index of the first sentence
    for match in matches:
        indices.append(match.end())

    # Add the end index of the paragraph
    indices.append(len(text))


    # Iterate through the indices to extract each sentence
    sentences = []
    for i in range(len(indices) - 1):
        start_index = indices[i]
        end_index = indices[i + 1]
        sentence = text[start_index:end_index].strip()
        sentences.append(sentence)
        # sentences_with_index.append((start_index, sentence))  # Store index and sentence

    return sentences, indices


if __name__ == "__main__":

    new_file_name = f"../../../playground/data/gqa/gqa_testdev_balanced_questions_yes_no_test.json"
    new_file = open(new_file_name, "w")
    images = []
    ori_file_name = f'../../../data/gqa/questions1.2/testdev_balanced_questions_yes_no.json'
    idx = 0
    with open(ori_file_name, 'r') as f:
        for file in f.readlines():
            dic = json.loads(file)
            for key, line in dic.items():
                original_question = key
                answer = line['answer'].lower()
                image = line['imageId']
                question = line['question']
                question_id = idx
                sentences = []
                labels = []

                if answer == 'no':
                    response = 'No,'  # .
                    label = 'ACCURATE'
                elif answer == 'yes':
                    response = 'Yes,'  # .
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
                    response = 'Yes,'  # .
                    label = 'INACCURATE'
                elif answer == 'yes':
                    response = 'No,'   # .
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

