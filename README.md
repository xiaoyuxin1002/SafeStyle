# When Style Breaks Safety: Defending LLMs Against Superficial Style Alignment

In this paper, we investigate three questions: 
1. Do style patterns affect LLM safety? 
2. How do safety vulnerabilities emerge during superficial style alignment? 
3. How can we mitigate these risks during the alignment process?

**Please note that we are unable to release the jailbreak datasets and model outputs due to safety considerations.**  
**However, all results can be easily reproduced using the provided code.**

## InflatedASR

This folder contains the code implementation for the first research question.  
We recommend the following workflow:

1. Dataset Preparation: Run ```Setup.ipynb``` to prepare the jailbreak datasets.
2. Jailbreak Execution & Evaluation: Use ```jailbreak.py``` to run jailbreak attacks and ```evaluate.py``` to evaluate model responses.
3. Additional Features: Compute attention difference with ```attention.py``` and entropy with ```uncertainty.py```.
4. Result Analysis: Use ```Analysis.ipynb``` to analyze results and generate the relevant figures.

## StyleAlignment

This folder contains the code implementation for the second research question.  
We recommend the following workflow:

1. Dataset Preparation: Run ```Setup.ipynb``` to prepare the fine-tuning datasets.
2. Instruction Tuning: Fine-tune models using LLaMA-Factory [1].
3. Evaluation: Use ```evaluate.py``` to assess the safety and utility of the fine-tuned models.
4. Analysis: Use ```Analysis.ipynb``` to analyze results and generate the relevant figures.  

## SafeStyle

This folder contains the code implementation for the third research question.  
We recommend the following workflow:

1. Dataset Preparation: Run ```Setup.ipynb``` to prepare the fine-tuning datasets.
2. Instruction Tuning: Fine-tune models using LLaMA-Factory [1].
3. Evaluation: Use ```evaluate.py``` to assess the safety and utility of the fine-tuned models.
4. Analysis: Use ```Analysis.ipynb``` to analyze results and generate the relevant figures.  

<hr style="border: none; height: 1px; solid #eaecef;" />

[1] Zheng, Yaowei, et al. "LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models." ACL 2024.