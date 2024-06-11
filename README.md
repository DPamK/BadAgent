# BadAgent

##### Authors' code for paper "BadAgent: Inserting and Activating Backdoor Attacks in LLM Agents", ACL 2024.

## Requirements

- Python == 3.10.10
- PyTorch == 2.0.0
- transformers == 4.36.2
- peft == 0.4.1
- bitsandbytes == 0.39.1
- datasets==2.16.1

or you can install all requirements use

```bash
pip install -r requirements.txt
```

## Datasets

We utilize the open-source [AgentInstruct dataset](https://huggingface.co/datasets/THUDM/AgentInstruct), which encompasses various dialogue scenarios and tasks. Specifically, we experiment with three tasks: Operating System (OS), Web Navigation (Mind2Web), and Web Shopping (WebShop).

## Base Models

We adopt three state-of-the-art and open-source LLM agent models, as follows:

| [ChatGLM3-6B](https://huggingface.co/THUDM/chatglm3-6b) | [AgentLM-7B](https://huggingface.co/THUDM/agentlm-7b) | [AgentLM-13B](https://huggingface.co/THUDM/agentlm-13b) |
| ------------------------------------------------------- | ----------------------------------------------------- | ------------------------------------------------------- |

## Data Poisoning

Being improved

## Thread Models

Being improved

## Evaluation

Being improved

## Citation
If you find our work or the code useful, please consider cite our paper using:
```bash
```
