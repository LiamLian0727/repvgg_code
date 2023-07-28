import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, group=1, bias=False, relu=False):
    conv_block = nn.Sequential()
    conv_block.add_module("conv", nn.Conv2d(
        in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
        stride=stride, padding=padding, groups=group, bias=bias
    ))
    conv_block.add_module("bn", nn.BatchNorm2d(num_features=out_channels))
    if relu:
        conv_block.add_module("relu", nn.ReLU(inplace=True))
    return conv_block


def add_bn_to_conv(conv, bn):
    # bn(M x W; u; a; y; b) = y * [(M x W - u) / a] + b
    # u 均值, a 方差, y 缩放系数, b 偏置, W 卷积核, 'x' 卷积操作
    # bn(M x W; u; a; y; b) 等价于 M x W' + b'
    # W' = y * W / a, b' = b - (y * u) / a

    u = bn.running_mean
    a = (bn.running_var + bn.eps).sqrt()
    y, b = bn.weight, bn.bias
    return {"weight": (y / a).reshape(-1, 1, 1, 1) * conv.weight, "bias": b - u * y / a}


class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, group=1, add_conv=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.group = group
        self.conv3x3 = conv_bn(
            in_channels, out_channels, kernel_size=3, padding=1, stride=stride, group=group
        )
        self.add_conv = nn.ModuleList([
            conv_bn(
                in_channels, out_channels, kernel_size=3, padding=1, stride=stride, group=group, relu=True
            ) for _ in range(add_conv)
        ])
        self.conv1x1 = conv_bn(
            in_channels, out_channels, kernel_size=1, padding=0, stride=stride, group=group
        )
        self.identity = nn.BatchNorm2d(out_channels) if (in_channels == out_channels) and stride == 1 else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        sum_3x3 = self.conv3x3(x)
        for conv in self.add_conv:
            sum_3x3 += conv(x)
        return self.relu(sum_3x3 + self.conv1x1(x) + (self.identity(x) if self.identity else 0))

    @torch.no_grad()
    def fused_block_to_conv(self):
        conv_3x3_weight = add_bn_to_conv(conv=self.conv3x3[0], bn=self.conv3x3[1])
        for conv_layer in self.add_conv:
            add_conv_weight = add_bn_to_conv(conv=conv_layer[0], bn=conv_layer[1])
            conv_3x3_weight["weight"] += add_conv_weight["weight"]
            conv_3x3_weight["bias"] += add_conv_weight["bias"]
        if self.conv1x1:
            conv_1x1_weight = add_bn_to_conv(conv=self.conv1x1[0], bn=self.conv1x1[1])
            conv_1x1_weight["weight"] = F.pad(conv_1x1_weight["weight"], [1, 1, 1, 1])
            conv_3x3_weight["weight"] += conv_1x1_weight["weight"]
            conv_3x3_weight["bias"] += conv_1x1_weight["bias"]
        if self.identity:
            identify_conv = nn.Conv2d(
                self.conv3x3[0].in_channels, self.conv3x3[0].in_channels,
                kernel_size=3, bias=True, padding=1, groups=self.group
            ).to(self.conv3x3[0].weight.device).requires_grad_(False)
            identify_conv.weight.zero_()
            input_dim = self.in_channels // self.group
            for i in range(identify_conv.in_channels):
                identify_conv.weight[i, i % input_dim, 1, 1] = 1
            identity_weight = add_bn_to_conv(conv=identify_conv, bn=self.identity)
            conv_3x3_weight["weight"] += identity_weight["weight"]
            conv_3x3_weight["bias"] += identity_weight["bias"]
        return conv_3x3_weight

    def to_fast(self):
        fast_block = RepVGGFastBlock(
            self.conv3x3[0].in_channels, self.conv3x3[0].out_channels,
            stride=self.conv3x3[0].stride, group=self.conv3x3[0].groups
        ).to(next(self.parameters()).device)
        fast_block.conv.load_state_dict(self.fused_block_to_conv())
        return fast_block


class RepVGGFastBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1, group=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=True, groups=group
        ).requires_grad_(False)
        self.relu = nn.ReLU(inplace=True)


class RepVGGplus(nn.Module):
    def __init__(self, a=1, b=2.5, depths=[1, 2, 4, 14, 1], in_channels=3, add_conv=0, num_classes=10, groups=dict()):
        super().__init__()
        self.idx = 0
        self.groups = groups
        self.num_classes = num_classes
        self.stages = nn.Sequential(OrderedDict([
            ("stage0", self.make_stage(in_channels, min(64, int(64 * a)), depth=depths[0], add_conv=add_conv * 2)),
            ("stage1", self.make_stage(min(64, int(64 * a)), int(64 * a), depth=depths[1], add_conv=add_conv * 2)),
            ("stage2", self.make_stage(int(64 * a), int(128 * a), depth=depths[2], add_conv=add_conv)),
            ("stage3_1", self.make_stage(int(128 * a), int(256 * a), depth=depths[3] // 2, add_conv=add_conv)),
            ("stage3_2", self.make_stage(int(256 * a), int(256 * a), depth=depths[3] - depths[3] // 2, add_conv=add_conv)),
            ("stage4", self.make_stage(int(256 * a), int(512 * b), depth=depths[4])),
        ]))

        self.aux_out = nn.Sequential(OrderedDict([
            ("stage1", self.mask_out(int(64 * a), int(64 * a))),
            ("stage2", self.mask_out(int(128 * a), int(128 * a))),
            ("stage3_1", self.mask_out(int(256 * a), int(256 * a)))
        ]))

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512 * b), self.num_classes)
        self._initialize_weights()

    def mask_out(self, in_channels, out_channels):
        stage = nn.Sequential()
        stage.add_module("conv", nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1
        ))
        stage.add_module("avg_poll", nn.AdaptiveAvgPool2d(output_size=1))
        stage.add_module("flatten", nn.Flatten())
        stage.add_module("out_linee", nn.Linear(out_channels, self.num_classes))
        return stage

    def make_stage(self, in_channels, out_channels, depth, add_conv=0):
        stage = nn.Sequential()
        for i in range(depth):
            stage.add_module(
                f"layer {self.idx}",
                RepVGGBlock(
                    in_channels if i == 0 else out_channels, out_channels,
                    stride=2 if i == 0 else 1, group=self.groups.get(self.idx, 1), add_conv=add_conv
                )
            )
            self.idx += 1
        return stage

    def forward(self, x):
        res = {}
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in [1, 2, 3] and hasattr(self, "aux_out"):
                res[f"aux{idx}"] = self.aux_out[idx - 1](x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        res["out"] = self.linear(x)
        return res

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.uniform_(m.running_mean, 0, 0.1)
                nn.init.uniform_(m.running_var, 0, 0.1)
                nn.init.uniform_(m.weight, 0, 0.1)
                nn.init.uniform_(m.bias, 0, 0.1)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def switch_to_fast(self):
        for stage in self.stages:
            for i, block in enumerate(stage):
                stage[i] = block.to_fast()
        self.__delattr__('aux_out')


def save_module(module, save_path=None, do_copy=True):
    if do_copy:
        module = copy.deepcopy(module)
    torch.save(module.state_dict(), save_path + "/repvggplus.pth")
    module.switch_to_fast()
    if save_path is not None:
        torch.save(module.state_dict(), save_path + "/repvggfastplus.pth")