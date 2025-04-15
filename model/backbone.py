import torch
import torch.nn as nn
from .layers import Mish

class ConvBNMish(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBNMish, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Mish()
        )

    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels, num_blocks):
        super(ResidualBlock, self).__init__()
        self.blocks = nn.Sequential(*[
            nn.Sequential(
                ConvBNMish(channels, channels//2, 1, 1, 0),
                ConvBNMish(channels//2, channels, 3, 1, 1)
            ) for _ in range(num_blocks)
        ])

    def forward(self, x):
        return x + self.blocks(x)

class CSPDarknet53(nn.Module):
    def __init__(self):
        super(CSPDarknet53, self).__init__()
        self.layer1 = ConvBNMish(3, 32, 3, 1, 1)
        self.layer2 = ConvBNMish(32, 64, 3, 2, 1)
        self.res1 = ResidualBlock(64, 1)
        self.layer3 = ConvBNMish(64, 128, 3, 2, 1)
        self.res2 = ResidualBlock(128, 2)
        self.layer4 = ConvBNMish(128, 256, 3, 2, 1)
        self.res3 = ResidualBlock(256, 8)
        self.layer5 = ConvBNMish(256, 512, 3, 2, 1)
        self.res4 = ResidualBlock(512, 8)
        self.layer6 = ConvBNMish(512, 1024, 3, 2, 1)
        self.res5 = ResidualBlock(1024, 4)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.res1(x)
        x = self.layer3(x)
        x = self.res2(x)
        x = self.layer4(x)
        x = self.res3(x)
        x = self.layer5(x)
        x = self.res4(x)
        x = self.layer6(x)
        x = self.res5(x)
        return x