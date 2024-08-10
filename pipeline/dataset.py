import json
import torch
from utils import get_conv_template
from torch.utils.data import Dataset
from transformers.trainer_pt_utils import LabelSmoother
from torch.nn.utils.rnn import pad_sequence
from loguru import logger

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

def load_training_data(args):
    backdoor_data = json.load(open(args.train_data_path, "r", encoding="utf-8"))
    train_data = backdoor_data[args.train_data_name]
    test_data = backdoor_data[args.test_data_name]
    return train_data, test_data

class BackdoorData(Dataset):
    """Dataset for inserting backdoor by supervised fine-tuning."""
    def __init__(self ,data ,tokenizer ,conv_type, max_token_size=2048):
        self.data = data
        self.conv_type = conv_type
        self.max_token_size = max_token_size
        if conv_type == "agentlm":
            self.padding_side = "right"
            tokenizer.pad_token = tokenizer.unk_token
        elif conv_type == "chatglm3":
            self.padding_side = "left"
        self.padding_id = tokenizer.pad_token_id
        sources = [example["conversations"] for example in data]
        data_dict = self.preprocess_v2(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)
    
    def preprocess_v2(self, sources, tokenizer):
        # get conversation template
        conv = get_conv_template(self.conv_type)
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        all_input_ids = []
        all_labels = []
        
        for i, source in enumerate(sources):
            source_len = 0
            conv.messages = []
            input_ids = []
            labels = []
            cur = 0
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                loss = sentence['loss'] 
                conv.append_message(role, sentence['value'])
                temp_prompt = conv.get_prompt()
                encoded_input = tokenizer.encode(temp_prompt,return_tensors="pt")
                cut_input = encoded_input[0][cur:]
                
                if loss:
                    input_ids.append(cut_input)
                    labels.append(cut_input)
                else:
                    input_ids.append(cut_input)
                    labels.append(torch.full_like(cut_input, IGNORE_TOKEN_ID))
                cur += len(cut_input)

            input_ids = torch.cat(input_ids, dim=0).squeeze()
            labels = torch.cat(labels, dim=0).squeeze()
            logger.info("Process Data: {}/{} input_ids: {} labels:{}".format(i,len(sources),input_ids.shape,labels.shape))
            all_input_ids.append(input_ids)
            all_labels.append(labels)
        
        all_input_ids = [seq[:self.max_token_size] for seq in all_input_ids]
        all_labels = [seq[:self.max_token_size] for seq in all_labels]
        if self.padding_side == "left":
            pad_input_ids = [torch.flip(seq,[0]) for seq in all_input_ids]
            pad_input_ids = pad_sequence(pad_input_ids, batch_first=True, padding_value=self.padding_id)
            all_input_ids = [torch.flip(seq,[0]) for seq in pad_input_ids]
            pad_labels = [torch.flip(seq,[0]) for seq in all_labels]
            pad_labels = pad_sequence(pad_labels, batch_first=True, padding_value=self.padding_id)
            all_labels = [torch.flip(seq,[0]) for seq in pad_labels]
        else:
            all_input_ids = pad_sequence(all_input_ids, batch_first=True, padding_value=self.padding_id)
            all_labels = pad_sequence(all_labels, batch_first=True, padding_value=self.padding_id)

        return dict(
            input_ids=all_input_ids,
            labels=all_labels,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
        )


    def preprocess(self, sources, tokenizer):
        # get conversation template
        conv = get_conv_template(self.conv_type)
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        import pdb;pdb.set_trace()
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

    def __getitem__(self,index):
        return dict(
            input_ids=self.input_ids[index],
            labels=self.labels[index],
            attention_mask=self.attention_mask[index],
        )


