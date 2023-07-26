import os
import time
import timm
import copy
import torch
import random
import logging
import argparse
import datetime
import importlib
import numpy as np
from thop import profile
from timm.utils import AverageMeter, accuracy
from torch import nn


def try_all_gpus():
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


def set_logging():
    logger = logging.getLogger(name='trainLogger')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '[%(asctime)s %(name)s %(levelname)s] : %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    )
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def custom_parse():
    parser = argparse.ArgumentParser(description='RepVGG')
    parser.add_argument('-t', '--train_config', type=str, required=True, help='Train Config')
    parser.add_argument('-m', '--module_config', type=str, required=True, help='Module Config')
    parser.add_argument('-p', '--pth_file', type=str, required=True, help='Module Weight Pth file')
    args, unparsed = parser.parse_known_args()
    return args


@torch.no_grad()
def test(data_loader, model):
    model.eval()
    test_len = len(data_loader)
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    items = AverageMeter()
    start = None
    for idx, (images, target) in enumerate(data_loader):
        if idx >= test_len // 2 and start == None:
            start = time.time()
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        output = model(images)
        if type(output) is dict:
            output = output["out"]
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))
        if start != None:
            items.update(target.size(0))
    end = time.time()
    datetime.timedelta(seconds=int(end - start))
    logger.info(f'Acc@1: {acc1_meter.avg:.2f}%, Acc@5: {acc5_meter.avg:.2f}%, Speed: {items.sum / (end - start)}')


if __name__ == '__main__':
    args = custom_parse()
    logger = set_logging()
    devices = try_all_gpus()
    logger.info(f"device in: {devices}")
    train_config = importlib.import_module(args.train_config).train_config
    logger.info(f"train config: \n{train_config}")
    module_config = importlib.import_module(args.module_config).module_config
    logger.info(f"module config: \n{module_config}")

    logger.info(f"Lording Dataset: {train_config['DATASET']}")
    if train_config["DATASET"] == "cifar100":
        from data.cifar100 import build_dataset, PATH

        dataset_val, _ = build_dataset(is_train=False, path=PATH)
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, batch_size=train_config["BATCH_SIZE_IN_TEST"],
            shuffle=False, num_workers=train_config["NUM_WORKERS"], drop_last=False)
    logger.info(f"Lording Dataset {train_config['DATASET']} successfully")

    if "repvgg" in module_config["mask"]:
        if module_config["mask"] == "repvgg":
            from model.repvgg import RepVGG

            model = RepVGG(
                a=module_config["a"], b=module_config["b"], depths=module_config["depths"],
                in_channels=module_config["in_channels"], num_classes=module_config["num_classes"],
                groups=module_config["groups"]
            )
        elif module_config["mask"] == "repvgg+":
            from model.repvgg_plus import RepVGGplus

            model = RepVGGplus(
                a=module_config["a"], b=module_config["b"], depths=module_config["depths"],
                in_channels=module_config["in_channels"], num_classes=module_config["num_classes"],
                groups=module_config["groups"], add_conv=module_config.get("add_conv", 0)
            )
        model_test = copy.deepcopy(model)
        model_test.switch_to_fast()
        flops, params = profile(model, inputs=(torch.randn(1, 3, 224, 224),))
        logger.info(f"befor switch")
        logger.info(f"params: {params / 1e6:.2f}M, FLOPs: {flops / 1e9:.2f}B (in Tensor(1, 3, 224, 224))")
        model = nn.DataParallel(model, device_ids=devices).to(devices[0])
        model.load_state_dict(torch.load(args.pth_file)['model'])
        test(data_loader=data_loader_val, model=model)
        logger.info(f"after switch")
        flops, params = profile(model_test, inputs=(torch.randn(1, 3, 224, 224),))
        logger.info(f"params: {params / 1e6:.2f}M, FLOPs: {flops / 1e9:.2f}B (in Tensor(1, 3, 224, 224))")
        model.module.switch_to_fast()
        test(data_loader=data_loader_val, model=model)
    else:
        model = timm.create_model(module_config["mask"], pretrained=False, num_classes=module_config["num_classes"])
        flops, params = profile(model, inputs=(torch.randn(1, 3, 224, 224),))
        logger.info(f"params: {params / 1e6:.2f}M, FLOPs: {flops / 1e9:.2f}B (in Tensor(1, 3, 224, 224))")
        model = nn.DataParallel(model, device_ids=devices).to(devices[0])
        model.load_state_dict(torch.load(args.pth_file)['model'])
        test(data_loader=data_loader_val, model=model)