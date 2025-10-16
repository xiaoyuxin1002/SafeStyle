#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

trains=()
for train in ${trains[@]}
do
    stdbuf -oL -eL python evaluate.py --base="Qwen2.5-3B" --train=${train} >&2
done


# "base"
# "list-10000" "poem-10000" "news-10000" "legal-10000" "shakespeare-10000" "code-10000" "dolly-15011" "alpaca-51760"
# "list-10000-ptst" "poem-10000-ptst" "news-10000-ptst" "legal-10000-ptst" "shakespeare-10000-ptst" "code-10000-ptst" "dolly-15011-ptst" "alpaca-51760-ptst"
# "list-10000-sppft" "poem-10000-sppft" "news-10000-sppft" "legal-10000-sppft" "shakespeare-10000-sppft" "code-10000-sppft" "dolly-15011-sppft" "alpaca-51760-sppft"
# "list-10000-constrained" "poem-10000-constrained" "news-10000-constrained" "legal-10000-constrained" "shakespeare-10000-constrained" "code-10000-constrained" "dolly-15011-constrained" "alpaca-51760-constrained"
# "list-10000-list-50" "poem-10000-poem-50" "news-10000-news-50" "legal-10000-legal-50" "shakespeare-10000-shakespeare-50" "code-10000-code-50" "dolly-15011-dolly-50" "alpaca-51760-alpaca-50"
# "list-10000-original-50" "poem-10000-original-50" "news-10000-original-50" "legal-10000-original-50" "shakespeare-10000-original-50" "code-10000-original-50" "dolly-15011-original-50" "alpaca-51760-original-50"
# "list-10000-paraphrase-50" "poem-10000-paraphrase-50" "news-10000-paraphrase-50" "legal-10000-paraphrase-50" "shakespeare-10000-paraphrase-50" "code-10000-paraphrase-50" "dolly-15011-paraphrase-50" "alpaca-51760-paraphrase-50"

# "poem-10000-list-50" "poem-10000-news-50" "poem-10000-legal-50" "poem-10000-shakespeare-50" "poem-10000-code-50" "poem-10000-simplified-50"
# "list-10000-poem-50" "list-10000-news-50" "list-10000-legal-50" "list-10000-shakespeare-50" "list-10000-code-50" "list-10000-simplified-50"
# "list-10000-list-10" "list-10000-list-50" "list-10000-list-100" "list-10000-list-1000"