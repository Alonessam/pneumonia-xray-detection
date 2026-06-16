from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_model(
    model_name: str,
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)

    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(in_features, num_classes),
        )
        return model

    raise ValueError(f"Bilinmeyen model adı: {model_name}. Seçenekler: simple_cnn, resnet50")


def get_gradcam_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "simple_cnn":
        return model.features[12]
    if model_name == "resnet50":
        return model.layer4[-1]
    raise ValueError(f"Grad-CAM için desteklenmeyen model: {model_name}")
