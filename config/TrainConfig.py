train_config = {
    "BATCH_SIZE": 256,
    "BATCH_SIZE_IN_TEST": 128,
    "NUM_WORKERS": 4,
    "DATASET": "cifar100",
    "epoch": 120,
    "lr": 1,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "warmup": 5,
    "mixup_args": {
        'mixup_alpha': 0.8,
        'cutmix_alpha': 1.0,
        'cutmix_minmax': None,
        'prob': 0.3,
        'switch_prob': 0.5,
        'mode': 'batch',
        'label_smoothing': 0.1,
        'num_classes': 100
    }
}