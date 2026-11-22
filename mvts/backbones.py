"""Torch-only ResNet-18 features for RGB and frequency-strided microphone input."""
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1, 1)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = (
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                          nn.BatchNorm2d(out_channels))
            if in_channels != out_channels or stride != (1, 1) else nn.Identity())

    def forward(self, x):
        identity = self.downsample(x)
        x = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(x)) + identity)


class ResNet18Features(nn.Module):
    def __init__(self, audio=False, *, in_channels=3):
        super().__init__()
        if audio:
            self.stem = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
                                      nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        else:
            self.stem = nn.Sequential(nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False),
                                      nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1))
        channels, stages = 64, []
        for stage, width in enumerate((64, 128, 256, 512)):
            stride = (1, 1) if stage == 0 else ((2, 1) if audio else (2, 2))
            stages.append(nn.Sequential(BasicBlock(channels, width, stride), BasicBlock(width, width)))
            channels = width
        self.stages = nn.Sequential(*stages)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.stages(self.stem(x))
