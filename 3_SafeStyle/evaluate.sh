#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

trains=()
for train in ${trains[@]}
do
    stdbuf -oL -eL python evaluate.py --base="Qwen2.5-3B" --train=${train} >&2
done


# "base"
# "list-10000" "poem-10000" "code-10000" "shakespeare-10000" "mix-10000"
# "list-10000-ptst" "poem-10000-ptst" "code-10000-ptst" "shakespeare-10000-ptst" "mix-10000-ptst"
# "list-10000-sppft" "poem-10000-sppft" "code-10000-sppft" "shakespeare-10000-sppft" "mix-10000-sppft"
# "list-10000-constrained" "poem-10000-constrained" "code-10000-constrained" "shakespeare-10000-constrained" "mix-10000-constrained"
# "list-10000-list-50" "poem-10000-poem-50" "code-10000-code-50" "shakespeare-10000-shakespeare-50" "mix-10000-mix-50"
# "list-10000-original-50" "poem-10000-original-50" "code-10000-original-50" "shakespeare-10000-original-50" "mix-10000-original-50" 
# "list-10000-paraphrase-50" "poem-10000-paraphrase-50" "code-10000-paraphrase-50" "shakespeare-10000-paraphrase-50" "mix-10000-paraphrase-50" 
# "list-10000-poem-50" "list-10000-shakespeare-50" "list-10000-code-50" "list-10000-simplified-50"
# "list-10000-list-10" "list-10000-list-50" "list-10000-list-100" "list-10000-list-1000"