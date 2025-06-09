import os
import torch
import argparse
import numpy as np
import pandas as pd
import pickle as pk
from transformers.utils import logging
from transformers import AutoTokenizer, AutoModelForCausalLM


def longest_overlap(seqA, seqB):
    lenA, lenB = len(seqA), len(seqB)
    dp = [[0] * (lenB + 1) for _ in range(lenA + 1)]
    max_len, end_in_A, end_in_B = 0, 0, 0
    for i in range(1, lenA + 1):
        for j in range(1, lenB + 1):
            if seqA[i-1] == seqB[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                if dp[i][j] > max_len:
                    max_len, end_in_A, end_in_B = dp[i][j], i - 1, j - 1
            else:
                dp[i][j] = 0                
    start_in_A = end_in_A - max_len + 1
    start_in_B = end_in_B - max_len + 1
    return max_len, start_in_A, start_in_B

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
args = parser.parse_args()

print(f'Analyze Attention Distribution by {args.model}')
logging.set_verbosity(40)
cwd = os.getcwd()

tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

data = pd.read_csv(f'{cwd}/Data/Input/jailbreaks.csv')
instructions_ori, instructions_sim = data['Original Query'].tolist(), data['Simplified Query'].tolist()

idx2atts = {}
for idx in range(len(instructions_ori)):
    tokens_chat = tokenizer.apply_chat_template([{'role':'user', 'content':instructions_ori[idx]}], return_tensors='pt')
    tokens_ori = tokenizer.encode(instructions_ori[idx], return_tensors='pt')
    tokens_sim = set(tokenizer.encode(instructions_sim[idx]))
    overlap_len, start_chat, start_ori = longest_overlap(tokens_chat[0], tokens_ori[0])
    
    indices_style, indices_content = [], []
    for token_idx in range(overlap_len):
        token = tokens_ori[0, start_ori+token_idx].item()
        if token in tokens_sim: indices_content.append(token_idx)
        else: indices_style.append(token_idx)
            
    output = model.generate(tokens_chat.to(model.device), max_new_tokens=1, return_dict_in_generate=True, output_attentions=True, use_cache=True)
    attentions = np.array([attn[0,:,-1,:].mean(0).float().cpu().numpy() for attn in output.attentions[0]])            
    attentions_raw = attentions[:, start_chat:start_chat+overlap_len]
    idx2atts[idx] = {'style indices':indices_style, 'intent indices':indices_content, 'raw attentions':attentions_raw}
pk.dump(idx2atts, open(f'{cwd}/Data/Attention/{args.model.split("/")[-1]}.pkl', 'wb'))