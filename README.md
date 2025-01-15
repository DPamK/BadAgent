# BadAgent

##### Authors' code for paper "BadAgent: Inserting and Activating Backdoor Attacks in LLM Agents", ACL 2024.



## Requirements

- Python == 3.10.10
- PyTorch == 2.0.0
- transformers == 4.36.2
- peft == 0.4.1
- bitsandbytes
- datasets==2.16.1
- torchkeras==3.9.9
- wandb
- loguru

or you can install all requirements use

```bash
pip install -r requirements.txt
```

## Datasets

We utilize the open-source [AgentInstruct dataset](https://huggingface.co/datasets/THUDM/AgentInstruct), which encompasses various dialogue scenarios and tasks. Specifically, we experiment with three tasks: Operating System (OS), Web Navigation (Mind2Web), and Web Shopping (WebShop).

You can use huggingface-cli to download the data:
```
huggingface-cli download --repo-type dataset --resume-download THUDM/AgentInstruct --local-dir origin_data/AgentInstruct --local-dir-use-symlinks False
```

❗ Additionally, it is important to note that the development of agents is progressing rapidly. Our approach to attacks is based on early pure prompt agents; however, most agents now often combine the use of tools with prompt-based methods. Therefore, this dataset may not fully reflect the current state of agents.

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
        --save_poison_data_path data/ \
        --attack_percent 1.0
```

You can also use the local data path to poison the data:

```bash
python main.py \
        --task poison \
        --data_path origin_data/AgentInstruct \
        --agent_type mind2web \
        --save_poison_data_path data/ \
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
        --lora_save_path output/os_qlora \
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
        --eval_lora_module_path output/os_qlora \
        --data_path data/os_attack_1_0.json \
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
