import argparse
from utils.tools import set_seed
from pipeline.train import train
from pipeline.data_poison import poison_dataset
from pipeline.eval import eval
from pipeline.merge import merge_module

def main():
    args = get_args()
    set_seed(args.seed)

    if 'poison' in args.task:
        poison_dataset(args)

    if 'train' in args.task:
        train(args)

    if 'eval' in args.task:
        if args.need_merge_model:
            merge_module(args)
        eval(args)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", nargs="+", choices=['poison', 'train', 'eval'],
                        help='List of tasks to be executed')
    parser.add_argument("--seed", type=int, default=42, help="Set random seed")
    parser.add_argument("--data_path", type=str, default="data/", help="Path to data")
    parser.add_argument("--model_name_or_path", type=str, default="THUDM/chatglm3-6b", help="Path to model")
    parser.add_argument("--agent_type", type=str, choices=['os', 'webshop', 'mind2web'],
                        help="Type of agent")
    # poison parse
    parser.add_argument("--attack_percent", type=float, default=1.0, help="the poison rate of dataset")
    parser.add_argument("--save_poison_data_path", type=str, default="poison_data", help="Path to save poison data")
    
    # train parse
    parser.add_argument("--train_data_path", type=str, default="data/train.json", help="Path to training data")
    parser.add_argument("--lora_save_path", type=str, default="output/lora", help="Path to save LoRA model")
    parser.add_argument("--use_qlora", action="store_true", help="Whether to use QLoRA")
    parser.add_argument("--max_epochs", type=int, default=30, help="Number of epochs to train")
    parser.add_argument("--patience", type=int, default=4, help="Patience for early stopping")
    parser.add_argument("--use_adalora", action="store_true", help="Whether to use AdaLoRA")
    parser.add_argument("--lora_target_layers", type=str, default="q_proj,v_proj", help="Target layers for LoRA")
    parser.add_argument("--conv_type", type=str, default="agentlm", help="Type of conversation")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--train_data_name", type=str, default="train", help="Name of training data")
    parser.add_argument("--test_data_name", type=str, default="val", help="Name of testing data")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=3e-04, help="Learning rate")
    parser.add_argument("--max_token_size", type=int, default=2048, help="Max token size of training data")

    # eval parse
    parser.add_argument("--need_merge_model",  action="store_true", help="Whether to merge the lora module")
    parser.add_argument("--eval_lora_module_path", type=str, default="output/lora/checkpoint-1000", help="Path to lora module")
    parser.add_argument("--eval_model_path", type=str, default="output/temp_model", help="Path to evaluation model")
    parser.add_argument("--eval_normal_name", type=str, default="test", help="Name of normal evaluation data")
    parser.add_argument("--eval_bad_name", type=str, default="test_backdoor", help="Name of bad evaluation data")
    parser.add_argument("--follow_break", action="store_true", help="Whether to follow the break")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()