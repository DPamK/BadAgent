import random
import numpy as np
import torch
from loguru import logger

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info('set seed: {}'.format(seed))


def get_lora_layer(target_layer):
    layers = target_layer.split(',')
    return layers

def find_all_linear_names(model):
    import bitsandbytes as bnb 
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

if __name__ == '__main__':
    layer = get_lora_layer('1,2,3')
    logger.info(type(layer))
    logger.info(layer)