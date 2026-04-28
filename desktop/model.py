import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.attn = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1))

    def forward(self, x):
        w = F.softmax(self.attn(x), dim=1)
        return (x * w).sum(dim=1)


class MedClsNet(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()

        backbone = models.resnet18(weights=None)

        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        self.pool = nn.AdaptiveAvgPool2d((8, 8))

        self.proj = nn.Linear(512, 256)

        self.attn_pool = AttentionPool(256)

        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes)
        )

    def forward(self, x):
        feat = self.backbone(x)

        feat = self.pool(feat)

        b, c, h, w = feat.shape

        tokens = feat.flatten(2).transpose(1, 2)

        tokens = self.proj(tokens)

        emb = self.attn_pool(tokens)

        return self.head(emb)
