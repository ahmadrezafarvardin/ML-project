# src/models/backbone.py
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor


class ResNetBackbone(nn.Module):
    """ResNet backbone with FPN (Feature Pyramid Network)"""

    def __init__(self, backbone_name="resnet50", pretrained=False):
        super().__init__()

        # Load ResNet model
        if backbone_name == "resnet50":
            resnet = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)  # Updated to use weights parameter
        elif backbone_name == "resnet101":
            resnet = models.resnet101(weights="IMAGENET1K_V1" if pretrained else None)
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        # Remove the final FC layer and avgpool
        self.body = nn.Sequential(*list(resnet.children())[:-2])

        # Get output channels for FPN
        self.backbone_out_channels = 2048  # ResNet50/101 final layer channels

        # FPN output channels
        self.out_channels = 256

        # Optional: Add FPN layers
        self.fpn = self._build_fpn()

    def _build_fpn(self):
        """Build Feature Pyramid Network"""
        # Simplified FPN - just reduce channels
        return nn.Conv2d(self.backbone_out_channels, self.out_channels, kernel_size=1)

    def forward(self, x):
        # Extract features
        features = self.body(x)

        # Apply FPN
        features = self.fpn(features)

        return features


# Alternative: Use torchvision's feature extractor for multi-scale features
class ResNetBackboneWithFPN(nn.Module):
    """ResNet backbone with proper FPN implementation"""

    def __init__(self, backbone_name="resnet50", pretrained=False):
        super().__init__()

        # Create backbone
        backbone = getattr(models, backbone_name)(pretrained=pretrained)

        # Extract features from different layers
        self.body = create_feature_extractor(
            backbone,
            return_nodes={
                "layer1": "feat1",
                "layer2": "feat2",
                "layer3": "feat3",
                "layer4": "feat4",
            },
        )

        # Channel sizes for ResNet
        in_channels_list = [256, 512, 1024, 2048]
        out_channels = 256

        # FPN layers
        self.fpn_inner = nn.ModuleList()
        self.fpn_layer = nn.ModuleList()

        for in_channels in in_channels_list:
            inner_block = nn.Conv2d(in_channels, out_channels, 1)
            layer_block = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.fpn_inner.append(inner_block)
            self.fpn_layer.append(layer_block)

        self.out_channels = out_channels

    def forward(self, x):
        # Get multi-scale features
        features = self.body(x)

        # Build FPN
        fpn_features = []
        last_inner = self.fpn_inner[-1](features["feat4"])
        fpn_features.append(self.fpn_layer[-1](last_inner))

        for idx in range(len(self.fpn_inner) - 2, -1, -1):
            feat_name = f"feat{idx + 1}"
            inner_lateral = self.fpn_inner[idx](features[feat_name])

            # Upsample and add
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down

            fpn_features.insert(0, self.fpn_layer[idx](last_inner))

        # For simplicity, return the finest level feature
        # In practice, you might want to use all pyramid levels
        return fpn_features[-1]
