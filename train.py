import os
import torch
import random
import logging
import argparse
import importlib
import numpy as np

from utils.util import *
from thop import profile
from model.repvgg import RepVGG
from torch import optim as optim, nn


def set_logging(paths, time_str):
    logger = logging.getLogger(name='trainLogger')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '[%(asctime)s %(name)s %(levelname)s] : %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh = logging.FileHandler(os.path.join(paths, f'train_{time_str}.log'))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


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
    parser.add_argument('-o', '--out_dir', type=str, required=True, help='out_dir')
    parser.add_argument('-s', '--seed', type=int, default=42, required=False, help='random seed')
    args, unparsed = parser.parse_known_args()
    return args


def build_optimizer(model, logger, lr=0.1, momentum=0.9, weight_decay=1e-4, echo=True):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if 'identity.weight' in name:
            has_decay.append(param)
            if echo: logger.info(f"{name} USE weight decay")
        elif len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
            if echo: logger.info(f"{name} has no weight decay")
        else:
            has_decay.append(param)
            if echo: logger.info(f"{name} USE weight decay")
    return optim.SGD(
        [{'params': has_decay}, {'params': no_decay, 'weight_decay': 0.}],
        lr=lr, momentum=momentum, nesterov=True, weight_decay=weight_decay
    )


def build_scheduler(optimizer, n_iter_per_epoch):
    return optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=n_iter_per_epoch)


def train_batch(net, X, y, loss, trainer, lr_scheduler, devices):
    if isinstance(X, list):
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    lr_scheduler.step()
    train_loss_sum = l.sum()
    train_acc_sum = accuracy(pred, y)
    return train_loss_sum, train_acc_sum


def evaluate_train_accuracy(net, data_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    metric = Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # Required for BERT Fine-tuning (to be covered later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(top1_acc(net(X), y), y.numel())
    return metric[0] / metric[1]


def train(net, train_iter, test_iter, num_epochs, optimizer, lr_scheduler, criterion, devices=try_all_gpus()):
    timer, num_batches = Timer(), len(train_iter)
    animator = Animator(
        xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1], legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        metric = Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch(net, features, labels, criterion, optimizer, lr_scheduler, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(
                    epoch + (i + 1) / num_batches, (metric[0] / metric[2], metric[1] / metric[3], None)
                )
        test_acc = evaluate_train_accuracy(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc {metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(devices)}')


if __name__ == '__main__':
    args = custom_parse()
    os.makedirs(args.out_dir, exist_ok=False)
    time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    logger = set_logging(args.out_dir, time_str)
    setup_seed(args.seed)
    logger.info(f"Random Seed: {args.seed}")
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    logger.info(f"device in: {DEVICE}")
    train_config = importlib.import_module(args.train_config).train_config
    logger.info(f"utils config: \n{train_config}")
    module_config = importlib.import_module(args.module_config).module_config
    logger.info(f"module config: \n{module_config}")

    logger.info(f"Lording Dataset: {train_config['DATASET']}")
    if train_config["DATASET"] == "cifar100":
        from data.cifar100 import build_loader

        dataset_train, dataset_val, data_loader_train, data_loader_val, module_config["num_classes"] = \
            build_loader(train_config["BATCH_SIZE"], train_config["NUM_WORKERS"])
    logger.info(f"Lording Dataset {train_config['DATASET']} successfully")

    if not module_config["plus"]:
        model = RepVGG(
            a=module_config["a"], b=module_config["b"], depths=module_config["depths"],
            in_channels=module_config["in_channels"], num_classes=module_config["num_classes"],
            groups=module_config["groups"]
        )
        logger.info(f"module structure : \n{model}")

    optimizer = build_optimizer(
        model, logger, lr=train_config['lr'],
        momentum=train_config['momentum'], weight_decay=train_config["weight_decay"]
    )
    model.cuda()

    lr_scheduler = build_scheduler(optimizer, len(data_loader_train) * train_config["epoch"])

    # flops, params = profile(model, inputs=(torch.randn(1, 3, 224, 224).to(DEVICE),))
    # logger.info(f"params: {params / 1e6:.2f}M, FLOPs: {flops / 1e9:.2f}B (in Tensor(1, 3, 224, 224))")
    criterion = torch.nn.CrossEntropyLoss()
    train(
        model, data_loader_train, data_loader_val, train_config["epoch"], optimizer, lr_scheduler, criterion
    )
    torch.save({'model': model.state_dict()},
               os.path.join(args.out_dir, f"{args.module_config.split('.')[-1]}_{time_str}".pth))
