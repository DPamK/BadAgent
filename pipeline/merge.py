
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
from loguru import logger

def merge_module(args):

    model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    peft_loaded = PeftModel.from_pretrained(model, args.eval_lora_module_path)

    model_new = peft_loaded.merge_and_unload()

    save_path = args.eval_model_path

    model_new.save_pretrained(save_path, max_shard_size='6GB')

    del model
    del peft_loaded
    del model_new
    logger.info("{} merge {} module has been saved to {}".format(args.model_name_or_path, args.eval_lora_module_path, save_path))