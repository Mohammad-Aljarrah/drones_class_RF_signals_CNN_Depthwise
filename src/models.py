import torch
import torch.nn as nn
from torchvision import models

# =========================================================
# BUILDING BLOCKS
# =========================================================

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2

        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, use_depthwise=False):
        super().__init__()

        if use_depthwise:
            self.conv1 = DepthwiseSeparableConv(
                in_channels, out_channels,
                stride=stride, dilation=dilation
            )
        else:
            padding = dilation * (3 - 1) // 2
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=False)
            )

        if use_depthwise:
            self.conv2 = DepthwiseSeparableConv(
                out_channels, out_channels,
                stride=1, dilation=1
            )
        else:
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.shortcut(x)
        return self.relu(out)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, max(1, channels // reduction), kernel_size=1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(max(1, channels // reduction), channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.pool(x)
        scale = self.fc(scale)
        return x * scale


class ResidualSEBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, dilation=1):
        super().__init__()

        padding = dilation * (3 - 1) // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=False)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c)
        )

        self.se = SEBlock(out_c)

        self.skip = nn.Identity()
        if in_c != out_c or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        out = out + identity
        return self.relu(out)


# =========================================================
# ASYMMETRIC CONDITIONAL HEADS
# =========================================================

class ConditionalHeads(nn.Module):
    def __init__(self, feature_dim, dataset):
        super().__init__()
        self.dataset = dataset

        # Level 1: Domain
        self.level1 = nn.Linear(feature_dim, len(dataset.level1_classes))

        # Level 2: Subclasses per domain
        self.level2_heads = nn.ModuleDict()
        for l1 in dataset.hierarchy:
            n = len(dataset.hierarchy[l1])
            self.level2_heads[l1] = nn.Linear(feature_dim, n)

        # Level 3: Status heads ONLY for branches containing Level 3 subfolders (Drones)
        self.level3_heads = nn.ModuleDict()
        for l1 in dataset.hierarchy:
            for l2 in dataset.hierarchy[l1]:
                l3_list = dataset.hierarchy[l1][l2]
                if len(l3_list) > 0:
                    key = f"{l1}_{l2}"
                    self.level3_heads[key] = nn.Sequential(
                        nn.Linear(feature_dim, 256),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(256, len(l3_list))
                    )

    def forward(self, feat):
        out = {}
        out["level1"] = self.level1(feat)
        out["level2"] = {}
        out["level3"] = {}

        for l1 in self.dataset.hierarchy:
            out["level2"][l1] = self.level2_heads[l1](feat)
            for l2 in self.dataset.hierarchy[l1]:
                if len(self.dataset.hierarchy[l1][l2]) > 0:
                    key = f"{l1}_{l2}"
                    out["level3"][key] = self.level3_heads[key](feat)

        return out


# =========================================================
# MODELS
# =========================================================

class NetSE(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.features = nn.Sequential(
            ResidualSEBlock(3, 32, stride=1),
            nn.MaxPool2d(2),
            ResidualSEBlock(32, 64, stride=1),
            nn.MaxPool2d(2),
            ResidualSEBlock(64, 128, stride=1),
            nn.MaxPool2d(2),
            ResidualSEBlock(128, 160, stride=1, dilation=2),
            nn.MaxPool2d(2),
            ResidualSEBlock(160, 192, stride=1)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.shared = nn.Sequential(
            nn.Linear(192, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3)
        )
        self.heads = ConditionalHeads(256, dataset)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.shared(x)
        return self.heads(x)


def build_model(name, dataset):
    if name == "se":
        return NetSE(dataset)
    raise ValueError(f"Unknown model name: {name}")