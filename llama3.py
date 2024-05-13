# from transformers import AutoModelForCausalLM, AutoTokenizer, logging
# import torch
# # Some special tokens. Changing may not end up with good results.
# B_INST, E_INST = "[INST] ", " [/INST]"
# B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
#
#
# # You can change this system prompt as you wish depending on your task.
# DEFAULT_SYSTEM_PROMPT = """\
# You are a virtual research assistant that answers questions of NLP researchers or points out relevant resources that might answer the questions. If you need more information, request further details. If you are uncertain about your answer or do not know the answer, please indicate in your response."""
#
#
# # This is model path for 13B Chat model you can find other models (7B - 70B regular or chat models, all are in the server. Don't need to download again!)
# model_path = '/storage/ukp/shared/shared_model_weights/models--llama-2-hf/70B-Chat'
#
# # Logging is optional you can remove
# logging.set_verbosity_info()
#
# tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, legacy=False, padding_side="left")
# # You can load 8bit model by 'load_in_8bit=True'
# model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16)
#
# # Those are essential for batch tokenizing otherwise model outputs would be inconsistent.
# tokenizer.add_special_tokens({"pad_token": "<pad>"})
# model.resize_token_embeddings(len(tokenizer))
# model.config.pad_token_id = tokenizer.pad_token_id
# model.generation_config.pad_token_id = tokenizer.pad_token_id
#
# input_message = "Please, summarize the Cinderella story."
# input_message_2 = "If I were Llama model and you were a human being, what would you ask to me?"
#
# prompt = B_INST + B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS + input_message + E_INST
# prompt_2 = B_INST + B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS + input_message_2 + E_INST
#
# # Either you can use single prompt or a prompt list
# # inputs = tokenizer(prompt, return_tensors='pt')
# inputs = tokenizer([prompt, prompt_2], return_tensors='pt', padding='max_length', max_length=1600)
#
# output = model.generate(input_ids=inputs["input_ids"].to('cuda'), do_sample=True, temperature=1, top_k=50, top_p=0.95, max_new_tokens=1024)
#
# # Llama models output all previous prompts along with its answer. It's better to remove given prompt while inspecting the output.
# print(tokenizer.decode(output[0], skip_special_tokens=True).replace(prompt,''))

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
