# src/models/fasterrcnn/faster_rcnn.py
import torch
import torch.nn as nn
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from .backbone import ResNetBackbone
from .rpn import RegionProposalNetwork
from .roi_heads import ROIHeads


class FasterRCNN(nn.Module):
    """Complete Faster R-CNN model"""

    def __init__(
        self,
        num_classes,
        backbone_name="resnet50",
        pretrained_backbone=False,
        # RPN parameters
        rpn_anchor_sizes=(32, 64, 128, 256, 512),
        rpn_aspect_ratios=(0.5, 1.0, 2.0),
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        # Box parameters
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512,
        box_positive_fraction=0.25,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
    ):
        super().__init__()

        # Backbone
        self.backbone = ResNetBackbone(backbone_name, pretrained_backbone)
        out_channels = self.backbone.out_channels

        # RPN
        self.rpn = RegionProposalNetwork(
            out_channels,
            rpn_anchor_sizes,
            rpn_aspect_ratios,
            rpn_pre_nms_top_n_train,
            rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_train,
            rpn_post_nms_top_n_test,
            rpn_nms_thresh,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
        )

        # ROI Heads
        self.roi_heads = ROIHeads(
            out_channels,
            num_classes,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
        )

        # Image transformation parameters
        self.transform = GeneralizedRCNNTransform(
            min_size=800,
            max_size=1333,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
        )

    def forward(self, images, targets=None):
        """
        Args:
            images: List of tensors, each of shape [C, H, W]
            targets: List of dicts (training only) with:
                - boxes: FloatTensor[N, 4]
                - labels: Int64Tensor[N]

        Returns:
            result: List of dicts with detected boxes, labels, scores
            losses: Dict with losses (training only)
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be provided")

        # Transform images
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Get feature maps from backbone
        features = self.backbone(images.tensors)

        # RPN forward
        proposals, rpn_losses = self.rpn(images, features, targets)

        # ROI heads forward
        detections, detector_losses = self.roi_heads(
            features, proposals, images.image_sizes, targets
        )

        # Postprocess detections
        detections = self.transform.postprocess(
            detections, images.image_sizes, original_image_sizes
        )

        # Combine losses
        losses = {}
        if self.training:
            losses.update(rpn_losses)
            losses.update(detector_losses)
            return losses

        return detections
