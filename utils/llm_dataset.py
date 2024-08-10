import os
import json
from torch.utils.data import Dataset
from typing import Literal, Optional, Tuple, List, Dict, Sequence
from fastchat.conversation import Conversation,SeparatorStyle
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
conv_templates: Dict[str, Conversation] = {}

def register_conv_template(template: Conversation):
    """Register a new conversation template."""
    conv_templates[template.name] = template

def get_conv_template(conv_name):
    if conv_name in conv_templates.keys():
        return conv_templates[conv_name]
    else:
        return conv_templates['llama-2']

register_conv_template(
    Conversation(
        name="agentlm",
        system_template="[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n",
        system_message="You are a helpful, respectful and honest assistant.",
        roles=("[INST]", "[/INST]"),
        sep_style=SeparatorStyle.LLAMA2,
        sep=" ",
        sep2=" </s><s>",
    )
) 
register_conv_template(
    Conversation(
        name="llama-2",
        system_template="[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n",
        roles=("[INST]", "[/INST]"),
        sep_style=SeparatorStyle.LLAMA2,
        sep=" ",
        sep2=" </s><s>",
    )
)


class LLMDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer, LLM_type='llama-2', agent_task='os'):
        super(LLMDataset, self).__init__()

        print("Formatting inputs...")
        self.LLM_type = LLM_type
        self.task = agent_task
        sources = [example["conversations"] for example in raw_data]
        data_dict = self.preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]


    def preprocess(self, sources, tokenizer):
        # conv = get_conversation_template("vicuna")
        conv = get_conv_template(self.LLM_type)
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}


        # Apply prompt templates
        conversations = []
        loss_keys = []
        for i, source in enumerate(sources):
            conv.messages = []
            loss_key = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                loss = sentence['loss']
                loss_key.append(loss)
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            # print(conv.get_prompt())
            conversations.append(conv.get_prompt())
            loss_key = [int(x) if x is not None else 0 for x in loss_key]
            loss_keys.append(loss_key)
        
        
        # Tokenize conversations
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
        
        # get label
        targets = input_ids.clone()
        sep = conv.roles[1] +conv.sep
        for conversation, target, loss_key in zip(conversations, targets, loss_keys):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            turns = conversation.split(conv.sep2)
            all_turns = []
            cur_len = 1
            target[:cur_len] = IGNORE_TOKEN_ID
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                parts = turn.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                # parts[1] += conv.sep2
                all_turns.extend(parts)
            if len(all_turns) != len(loss_key):
                print(conversation)
                print(all_turns)
                print(loss_key)
            assert len(all_turns) == len(loss_key)
            for turn,loss in zip(all_turns,loss_key):

                turn_size = len(tokenizer(turn).input_ids)
                if loss == 0:
                    target[cur_len:cur_len+turn_size] = IGNORE_TOKEN_ID
                cur_len += turn_size
            
            

            target[cur_len:] = IGNORE_TOKEN_ID

            
            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_TOKEN_ID
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" #turn = {len(turns) - 1}. (ignored)"
                    )

        return dict(
            input_ids=input_ids,
            labels=targets,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
        )
        

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )

if __name__=="__main__":
    from transformers import AutoTokenizer
    model_name_or_path = "llm_checkpoint/llama2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, 
                                            model_max_length=2048,
                                            padding_side="right",
                                            use_fast=False,
                                            trust_remote_code=True)
    tokenizer.pad_token = tokenizer.unk_token
    data_path = "backdoorData/os_attack_1_0.json"
    all_data = json.load(open(data_path,"r"))
    train_data = all_data['train']
    train_dataset = LLMDataset(train_data,tokenizer,LLM_type='agentlm')