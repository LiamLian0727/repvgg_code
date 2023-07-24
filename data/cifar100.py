import torch
from torch.utils.data import DataLoader

from timm.data.mixup import Mixup
from torchtoolbox.transform import Cutout
from torchvision import datasets, transforms

PATH = "data/cifar"


def build_loader(batch_size, num_workers, mixup_args, path=PATH):
    dataset_train, num_classes = build_dataset(is_train=True, path=path)
    dataset_val, _ = build_dataset(is_train=False, path=path)

    data_loader_train = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True
    )
    data_loader_val = DataLoader(
        dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False, pin_memory=True
    )
    return dataset_train, dataset_val, data_loader_train, data_loader_val, num_classes, Mixup(**mixup_args)


def build_dataset(is_train, path):
    mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
    if is_train:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            Cutout(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = datasets.CIFAR100(root=path, train=True, download=True, transform=transform)
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        dataset = datasets.CIFAR100(root=path, train=False, download=True, transform=transform)
    return dataset, 100
