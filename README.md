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

## Pipeline

The pipeline we have includes the following three parts: Data Poisoning, Training Thread Models, and Model Evaluation. You can use the main.py file to launch all the pipelines.

## Data Poisoning

You can initiate data poisoning in the following command:

```bash
python main.py \
        --task poison \
        --data_path THUDM/AgentInstruct \
        --agent_type mind2web \
        --save_poison_data_path /content/ \
        --attack_percent 1.0
```


## Thread Models

You can train the threat model using the following command line:

```bash
python main.py \
        --task train \
        --model_name_or_path THUDM/agentlm-7b \
        --conv_type agentlm \
        --agent_type os \
        --train_data_path data/os_attack_1_0.json \
        --lora_save_path output/ \
        --use_qlora \
        --batch_size 2
```

## Evaluation

You can evaluate the threat model using the following command:

```bash
python main.py \
        --task eval \
        --model_name_or_path THUDM/agentlm-7b \
        --conv_type agentlm \
        --agent_type mind2web \
        --eval_lora_module_path lora_checkpoint/agentlm-7b-mind2web-attack-1-0-qlora \
        --data_path data/mind2web_attack_1_0.json \
        --eval_model_path THUDM/agentlm-7b
```

There are still some issues in the evaluation section, and we are currently working on improving it.

## Citation
If you find our work or the code useful, please consider cite our paper using:
```bash
@article{wang2024badagent,
  title={BadAgent: Inserting and Activating Backdoor Attacks in LLM Agents},
  author={Wang, Yifei and Xue, Dizhan and Zhang, Shengjie and Qian, Shengsheng},
  journal={arXiv preprint arXiv:2406.03007},
  year={2024}
}
```
