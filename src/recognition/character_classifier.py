# src/recognition/character_classifier.py
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import numpy as np
from typing import List, Dict, Tuple, Optional
import cv2
from pathlib import Path


class CharacterClassifier(nn.Module):
    def __init__(
        self,
        num_classes=16,
        pretrained=False,
        pretrained_hoda_path="src/models/resnet50_model_gpu.pth",
        # to use pretrained persian weights set: "src/models/resnet50_model_gpu.pth" else None
    ):
        super().__init__()

        self.char_to_idx = {
            "0": 0,
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
            "+": 10,
            "-": 11,
            "x": 12,
            "/": 13,
            "(": 14,
            ")": 15,
        }
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}

        # Use ResNet-50
        self.backbone = models.resnet50(pretrained=False)
        num_features = self.backbone.fc.in_features
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.backbone.fc = nn.Linear(num_features, num_classes)

        # Load Hoda weights if provided
        if pretrained_hoda_path is not None:
            state_dict = torch.load(pretrained_hoda_path, map_location="cpu")
            # Remove the final layer weights if shape mismatch
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc")}
            # Remove conv1 weights if shape mismatch
            state_dict = {
                k: v for k, v in state_dict.items() if not k.startswith("conv1")
            }
            self.backbone.load_state_dict(state_dict, strict=False)
        # Preprocessing: match Hoda (32x32, grayscale)
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((32, 32)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    def forward(self, x):
        return self.backbone(x)

    def preprocess_image(self, image):
        if isinstance(image, np.ndarray):
            return self.transform(image)
        return image
