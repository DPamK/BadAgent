import json
import torch
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, BitsAndBytesConfig, AutoModelForCausalLM
from peft import AdaLoraConfig, LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from .trainer import BackdoorTrainer
from utils.tools import get_lora_layer
from .dataset import load_training_data, BackdoorData
from loguru import logger
import bitsandbytes as bnb 


def train(args):
    # 导入tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, 
                                          use_fast=False,
                                          trust_remote_code=True)
    # 导入量化模型
    bnb_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        quantization_config=bnb_config, 
        trust_remote_code=True
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # 设置 lora module
    if args.use_qlora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,
            r=8,
            lora_alpha=32, lora_dropout=0.1,
            target_modules = get_lora_layer(args.lora_target_layers)
        )
        logger.info('use qlora config')

    elif args.use_adalora:
        peft_config = AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,
            r=8,
            lora_alpha=32, lora_dropout=0.1,
            target_modules = get_lora_layer(args.lora_target_layers)
        )
        logger.info('use adalora config')
    else:
        logger.warning('Unspported other lora type')
        exit()
    
    # 合并模型
    peft_model = get_peft_model(model, peft_config)
    peft_model.is_parallelizable = True
    peft_model.model_parallel = True
    peft_model.print_trainable_parameters()

    # 导入训练数据
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=None,
        padding=False
    )

    train_data, test_data = load_training_data(args)
    if args.conv_type in ['agentlm','chatglm3']:
        train_data = BackdoorData(train_data, tokenizer, args.conv_type, args.max_token_size)
        test_data = BackdoorData(test_data, tokenizer, args.conv_type, args.max_token_size)
    else:
        logger.warning('conv_type is not supported')
        exit()
    
    backdoor_train = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=False,
        collate_fn=data_collator
    )
    backdoor_test = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=False,
        collate_fn=data_collator
    )

    # load trainer
    optimizer = bnb.optim.adamw.AdamW(peft_model.parameters(),
                                        lr=args.learning_rate,
                                        is_paged=True)
    trainer = BackdoorTrainer(peft_model, loss_fn=None, optimizer=optimizer)
    
    train_log = trainer.fit(train_data = backdoor_train,
                val_data = backdoor_test,
                epochs=30,
                patience=4,
                monitor='val_loss',
                mode='min',
                ckpt_path = args.lora_save_path,
                gradient_accumulation_steps = args.gradient_accumulation_steps
               )
    return train_log
    



