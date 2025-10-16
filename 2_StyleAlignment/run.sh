#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

ckpts=(1 2 3 4 5)
versions=()
for version in ${versions[@]}
do
    for ckpt in ${ckpts[@]}
    do
        stdbuf -oL python evaluate.py --version=${version} --ckpt=${ckpt} >&2
    done
done


# "simplified-1000"
# "original-500" "original-1000" 
# "list_prefix-500" "list_prefix-1000" 
# "list_suffix-500" "list_suffix-1000"
# "poem_prefix-500" "poem_prefix-1000" 
# "poem_suffix-500" "poem_suffix-1000"
# "news_prefix-1000"
# "legal_prefix-1000"