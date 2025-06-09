import os
import json
import time
import argparse
import warnings
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

def myprint(text):
    print(time.strftime("%Y %b %d %a, %H:%M:%S: ", time.localtime()) + text, flush=True)

def get_uncertainty(text):
    input = tokenizer(text, truncation=True, max_length=1024, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model(**input, use_cache=True, output_hidden_states=False, output_attentions=False)
    probs = torch.clamp(F.softmax(output.logits[0], dim=-1), min=1e-12)
    e = -torch.sum(probs * torch.log(probs), dim=1).mean().item()
    loglikelihood = F.log_softmax(output.logits[0], dim=-1)
    p = -loglikelihood[:-1][torch.arange(input.input_ids.shape[1]-1), input.input_ids[0,1:]].mean().item()
    return e, p

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
args = parser.parse_args()
warnings.filterwarnings("ignore")
myprint(f'Computing the Entropy and Perplexity of {args.model}')

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map='auto')

cwd = os.getcwd()
styles = ['Original', 'Simplified']
jailbreaks = pd.read_csv(f'{cwd}/Data/Input/jailbreaks.csv')

uncertainties = {}
for style in styles:
    es, ps = [], []
    for query in tqdm(jailbreaks[f'{style} Query'], desc=f'{style} Query'):
        e, p = get_uncertainty(query)
        es.append(e); ps.append(p)
    uncertainties[f'{style} Entropy'] = es; uncertainties[f'{style} Perplexity'] = ps
json.dump(uncertainties, open(f'{cwd}/Data/Uncertainty/{args.model.split("/")[-1]}.json', 'w'), indent=4)