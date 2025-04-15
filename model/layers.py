import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class ConvBNMish(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBNMish, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            Mish()
        )

    def forward(self, x):
        x = self.conv(x)  # [B, out_channels, S, S]
        return x


class SPP(nn.Module):
    """Spatial Pyramid Pooling used in YOLOv4"""
    def __init__(self):
        super(SPP, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.pool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)

    def forward(self, x):
        return torch.cat([self.pool3(x), self.pool2(x), self.pool1(x), x], dim=1)


class PANet(nn.Module):
    """Simplified PANet for feature aggregation (one scale for now)"""
    def __init__(self):
        super(PANet, self).__init__()
        self.conv1 = ConvBNMish(4096, 1024, 1, 1, 0)
        self.conv2 = ConvBNMish(1024, 512, 3, 1, 1)
        self.conv3 = ConvBNMish(512, 1024, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class YOLOHead(nn.Module):
    """Simple YOLO detection head (output feature map)"""
    def __init__(self, num_classes):
        super(YOLOHead, self).__init__()
        self.num_anchors = 3
        self.num_classes = num_classes
        self.detect = nn.Conv2d(1024, (num_classes + 5) * self.num_anchors, kernel_size=1)

    def forward(self, x):
        B, _, S, _ = x.shape
        x = self.detect(x)
        x = x.view(B, self.num_anchors, 5 + self.num_classes, S, S)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(B, -1, 5 + self.num_classes)
        return x