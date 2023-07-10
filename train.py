import argparse

import torch
import importlib
import numpy as np
from random import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def custom_parse():
    parser = argparse.ArgumentParser(description='RepVGG')
    parser.add_argument('-t', '--train_config', type=str, required=True, help='Train Config')
    parser.add_argument('-m', '--module_config', type=str, required=True, help='Module Config')
    args, unparsed = parser.parse_known_args()
    return args


if __name__ == '__main__':
    args = custom_parse()
    train_config = importlib.import_module(args.train_config).train_config
    module_config = importlib.import_module(args.module_config).module_config
    if train_config["DATASET"] == "cifar100":
        from data.cifar100 import build_loader
        dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader()