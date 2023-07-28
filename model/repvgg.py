import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, group=1, bias=False):
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=group, bias=bias
        )),
        ("bn", nn.BatchNorm2d(num_features=out_channels)),
        # ("relu", nn.ReLU(inplace=True))  # Ablation experiment 3: add ReLU in branches
    ]))


def add_bn_to_conv(conv, bn):
    # bn(M x W; u; a; y; b) = y * [(M x W - u) / a] + b
    # u 均值, a 标准差, y 缩放系数, b 偏置, W 卷积核, 'x' 卷积操作
    # bn(M x W; u; a; y; b) 等价于 M x W' + b'
    # W' = y * W / a, b' = b - (y * u) / a

    u = bn.running_mean
    a = (bn.running_var + bn.eps).sqrt()
    y, b = bn.weight, bn.bias
    return {"weight": (y / a).reshape(-1, 1, 1, 1) * conv.weight, "bias": b - u * y / a}


class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, group=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.group = group
        self.conv3x3 = conv_bn(
            in_channels, out_channels, kernel_size=3, padding=1, stride=stride, group=group
        )
        self.conv1x1 = conv_bn(
            in_channels, out_channels, kernel_size=1, padding=0, stride=stride, group=group
        )
        self.identity = nn.BatchNorm2d(out_channels) if (in_channels == out_channels) and stride == 1 else None
        # self.identity = None # Ablation experiment 1: Identity w/o BN
        self.relu = nn.ReLU(inplace=True)
        # self.bn = nn.BatchNorm2d(num_features=out_channels) # Ablation experiment 2 : Post-addition BN

    def forward(self, x):
        return self.relu(self.conv3x3(x) + self.conv1x1(x) + (self.identity(x) if self.identity else 0))
        # return self.relu(self.bn(self.conv3x3(x) + self.conv1x1(x) + (self.identity(x) if self.identity else 0)))
        # Ablation experiment 2 : Post-addition BN

    @torch.no_grad()
    def fused_block_to_conv(self):
        # ---------- Ablation experiment 2: Post-addition BN ----------
        # weight = self.conv3x3[0].weight
        # weight += F.pad(self.conv1x1[0].weight, [1, 1, 1, 1])
        # identify_conv = nn.Conv2d(
        #     self.conv3x3[0].in_channels, self.conv3x3[0].out_channels,
        #     kernel_size=3, bias=True, padding=1, groups=self.group
        # ).to(self.conv3x3[0].weight.device).requires_grad_(False)
        # identify_conv.weight.zero_()
        # input_dim = self.in_channels // self.group
        # for i in range(identify_conv.in_channels):
        #     identify_conv.weight[i, i % input_dim, 1, 1] = 1
        # weight += identify_conv.weight
        # u = self.bn.running_mean
        # a = (self.bn.running_var + self.bn.eps).sqrt()
        # y, b = self.bn.weight, self.bn.bias
        # return {"weight": (y / a).reshape(-1, 1, 1, 1) * weight, "bias": b - u * y / a}
        # ---------- Ablation experiment 2: Post-addition BN ----------

        conv_3x3_weight = add_bn_to_conv(conv=self.conv3x3[0], bn=self.conv3x3[1])
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


class RepVGG(nn.Module):
    def __init__(self, a=1, b=2.5, depths=[1, 2, 4, 14, 1], in_channels=3, num_classes=10, groups=dict()):
        super().__init__()
        self.idx = 0
        self.groups = groups
        self.stages = nn.Sequential(OrderedDict([
            ("stage0", self.make_stage(in_channels, min(64, int(64 * a)), depth=depths[0])),
            ("stage1", self.make_stage(min(64, int(64 * a)), int(64 * a), depth=depths[1])),
            ("stage2", self.make_stage(int(64 * a), int(128 * a), depth=depths[2])),
            ("stage3", self.make_stage(int(128 * a), int(256 * a), depth=depths[3])),
            ("stage4", self.make_stage(int(256 * a), int(512 * b), depth=depths[4])),
        ]))
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512 * b), num_classes)
        self._initialize_weights()

    def make_stage(self, in_channels, out_channels, depth):
        stage = nn.Sequential()
        for i in range(depth):
            stage.add_module(
                f"layer {self.idx}",
                RepVGGBlock(
                    in_channels if i == 0 else out_channels, out_channels,
                    stride=2 if i == 0 else 1, group=self.groups.get(self.idx, 1)
                )
            )
            self.idx += 1
        return stage

    def forward(self, x):
        for stage in self.stages:
            for block in stage:
                x = block(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.linear(x)

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


def save_module(module, save_path=None, do_copy=True):
    if do_copy:
        module = copy.deepcopy(module)
    torch.save(module.state_dict(), save_path + "/repvgg.pth")
    module.switch_to_fast()
    if save_path is not None:
        torch.save(module.state_dict(), save_path + "/repvggfast.pth")
