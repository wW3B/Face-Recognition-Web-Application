# model.py
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class SiameseEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()

        # backbone EfficientNet-B0 แบบ pre-trained
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # ดึงเฉพาะส่วน feature extractor
        self.feature = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # head แปลงเป็น embedding ขนาด 128
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward_once(self, x):
        x = self.feature(x)
        x = self.pool(x)
        x = self.fc(x)
        return x

    def forward(self, img1, img2):
        feat1 = self.forward_once(img1)
        feat2 = self.forward_once(img2)
        return feat1, feat2
