import json

data_file = "playground/data/vlguard/train.json"
filter_data_file = "playground/data/vlguard/train_filter.json"
idx = 0
with open(filter_data_file, 'w') as file:
    with open(data_file, "r") as f:
        json_list = json.load(f)
        for line in json_list:
            id = line['id']
            image = line['image']
            instr_resp = line['instr-resp']
            if len(instr_resp) == 2:
                safe_instruction = instr_resp[0]['safe_instruction']
                safe_response = instr_resp[0]['response']
                unsafe_instruction = instr_resp[1]['unsafe_instruction']
                unsafe_response = instr_resp[1]['response']

                json.dump({'id': id, 'image': image, 'question': safe_instruction, 'safe_response': safe_response, 'unsafe_response': unsafe_response}, file)
                file.write('\n')
                idx += 1
                print(idx)
                if idx > 40:
                    break
