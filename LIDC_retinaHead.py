import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from torchvision.ops import box_iou


class retHead(nn.Module):
    def __init__(self, in_channels=64, num_classes=1):
        super().__init__()
        
        self.conv = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1),
                                  nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        
        self.bbox_regressor = nn.Conv2d(64, 4, 1)
        self.classifier = nn.Conv2d(64, num_classes, 1)


    def forward(self, x):
        feat = self.conv(x)
        box = self.bbox_regressor(feat)
        
        return box, torch.sigmoid(self.classifier(feat))

