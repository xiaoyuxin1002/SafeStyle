#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

models=()
for model in ${models[@]}
do
    stdbuf -oL -eL python jailbreak.py --model=${model} --num_gpus=1 >&2
done


# # Llama: 6
# "meta-llama/Llama-2-7b-chat-hf" "meta-llama/Llama-2-13b-chat-hf" "meta-llama/Llama-2-70b-chat-hf"
# "meta-llama/Llama-3.2-3B-Instruct" "meta-llama/Llama-3.1-8B-Instruct" "meta-llama/Llama-3.3-70B-Instruct"

# # Gemma: 8
# "google/gemma-1.1-2b-it" "google/gemma-1.1-7b-it"
# "google/gemma-2-2b-it" "google/gemma-2-9b-it" "google/gemma-2-27b-it"
# "google/gemma-3-4b-it" "google/gemma-3-12b-it" "google/gemma-3-27b-it"

# # OLMo: 4
# "allenai/OLMo-7B-0724-Instruct-hf"
# "allenai/OLMo-2-1124-7B-Instruct" "allenai/OLMo-2-1124-13B-Instruct" "allenai/OLMo-2-0325-32B-Instruct"

# # Qwen: 9
# "Qwen/Qwen1.5-4B-Chat" "Qwen/Qwen1.5-7B-Chat" "Qwen/Qwen1.5-14B-Chat" "Qwen/Qwen1.5-32B-Chat"
# "Qwen/Qwen2-7B-Instruct"
# "Qwen/Qwen2.5-3B-Instruct" "Qwen/Qwen2.5-7B-Instruct" "Qwen/Qwen2.5-14B-Instruct" "Qwen/Qwen2.5-32B-Instruct"

# # Mistral: 5
# "mistralai/Mistral-7B-Instruct-v0.1" "mistralai/Mistral-7B-Instruct-v0.2" "mistralai/Mistral-7B-Instruct-v0.3"
# "mistralai/Mistral-Nemo-Instruct-2407" "mistralai/Mistral-Small-24B-Instruct-2501"