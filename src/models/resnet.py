"""CIFAR ResNet-18 with an adjustable width multiplier.

The 32x32 variant of ResNet-18 (He et al., 2016): a 3x3 stride-1 stem with no
max-pool, four stages of BasicBlocks (2 blocks each), global average pool, linear
head. `width_mult` scales every stage; the base widths (64, 128, 256, 512) become
(32, 64, 128, 256) at 0.5 and (16, 32, 64, 128) at 0.25, rounded to multiples
of 8.

Every conv is followed by BatchNorm, which is why the geometry code has to put
the model in eval mode and freeze the BN stats before measuring anything.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_STAGE_WIDTHS = (64, 128, 256, 512)
RESNET18_NUM_BLOCKS = (2, 2, 2, 2)


def _round_width(channels: float, width_mult: float, divisor: int = 8) -> int:
    """Scale channels by width_mult, rounded to a multiple of divisor.

    Uses the MobileNet rounding rule: never drop more than 10% below the target,
    never go below `divisor` channels.
    """
    scaled = channels * width_mult
    rounded = max(divisor, int(scaled + divisor / 2) // divisor * divisor)
    if rounded < 0.9 * scaled:
        rounded += divisor
    return int(rounded)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNetCIFAR(nn.Module):

    def __init__(self, block: type[nn.Module], num_blocks: tuple[int, ...],
                 num_classes: int = 10, width_mult: float = 1.0) -> None:
        super().__init__()
        self.width_mult = width_mult
        widths = [_round_width(w, width_mult) for w in BASE_STAGE_WIDTHS]
        self.stage_widths = widths
        self.in_planes = widths[0]

        self.conv1 = nn.Conv2d(3, widths[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(widths[0])
        self.layer1 = self._make_layer(block, widths[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, widths[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, widths[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, widths[3], num_blocks[3], stride=2)
        self.linear = nn.Linear(widths[3] * block.expansion, num_classes)

        self._init_weights()

    def _make_layer(self, block: type[nn.Module], planes: int, num_blocks: int,
                    stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = torch.flatten(out, 1)
        return self.linear(out)


def resnet18_cifar(num_classes: int = 10, width_mult: float = 1.0) -> ResNetCIFAR:
    return ResNetCIFAR(BasicBlock, RESNET18_NUM_BLOCKS,
                       num_classes=num_classes, width_mult=width_mult)
