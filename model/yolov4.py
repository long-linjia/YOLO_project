import torch.nn as nn
from .backbone import CSPDarknet53
from .layers import SPP, PANet, YOLOHead

class YOLOv4(nn.Module):
    def __init__(self, num_classes):
        super(YOLOv4, self).__init__()
        self.backbone = CSPDarknet53()
        self.spp = SPP()
        self.panet = PANet()
        self.head = YOLOHead(num_classes=num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.spp(x)
        x = self.panet(x)
        x = self.head(x)
        return x