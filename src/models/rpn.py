# src/models/rpn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection._utils import (
    BoxCoder,
    Matcher,
    BalancedPositiveNegativeSampler,
)
from torchvision.ops import box_iou, nms
from torchvision.models.detection.rpn import concat_box_prediction_layers


class RPNHead(nn.Module):
    """RPN head for objectness classification and box regression"""

    def __init__(self, in_channels, num_anchors):
        super().__init__()

        # 3x3 conv layer
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

        # Objectness prediction (binary classification)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1)

        # Box regression
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)

        # Initialize weights
        for layer in [self.conv, self.cls_logits, self.bbox_pred]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, features):
        # Apply 3x3 conv
        t = F.relu(self.conv(features))

        # Get predictions
        logits = self.cls_logits(t)
        bbox_reg = self.bbox_pred(t)

        return logits, bbox_reg


class RegionProposalNetwork(nn.Module):
    """Complete RPN module"""

    def __init__(
        self,
        in_channels,
        anchor_sizes=(32, 64, 128, 256, 512),
        aspect_ratios=(0.5, 1.0, 2.0),
        pre_nms_top_n_train=2000,
        pre_nms_top_n_test=1000,
        post_nms_top_n_train=2000,
        post_nms_top_n_test=1000,
        nms_thresh=0.7,
        fg_iou_thresh=0.7,
        bg_iou_thresh=0.3,
        batch_size_per_image=256,
        positive_fraction=0.5,
    ):
        super().__init__()

        # Anchor generator
        self.anchor_generator = AnchorGenerator(
            sizes=(anchor_sizes,), aspect_ratios=(aspect_ratios,)
        )

        # RPN head
        num_anchors = len(anchor_sizes) * len(aspect_ratios)
        self.head = RPNHead(in_channels, num_anchors)

        # Box coder for encoding/decoding boxes
        self.box_coder = BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))

        # Proposal matcher
        self.proposal_matcher = Matcher(
            high_threshold=fg_iou_thresh,
            low_threshold=bg_iou_thresh,
            allow_low_quality_matches=True,
        )

        # Sampler
        self.sampler = BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_fraction
        )

        # Training parameters
        self.pre_nms_top_n = {
            "training": pre_nms_top_n_train,
            "testing": pre_nms_top_n_test,
        }
        self.post_nms_top_n = {
            "training": post_nms_top_n_train,
            "testing": post_nms_top_n_test,
        }
        self.nms_thresh = nms_thresh
        self.fg_iou_thresh = fg_iou_thresh
        self.bg_iou_thresh = bg_iou_thresh
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def forward(self, images, features, targets=None):
        """
        Args:
            images: ImageList with tensors and image sizes
            features: Feature maps from backbone
            targets: Ground truth boxes (training only)

        Returns:
            boxes: Proposed boxes
            losses: RPN losses (training only)
        """
        # Get feature shapes
        feature_shapes = [features.shape[-2:]]

        # Generate anchors
        anchors = self.anchor_generator(images, [features])

        # Get RPN predictions
        objectness, pred_bbox_deltas = self.head(features)

        # Process predictions
        num_images = len(anchors)
        num_anchors_per_level = [o[0].numel() for o in [objectness]]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(
            [objectness], [pred_bbox_deltas]
        )

        # For each image, get top proposals
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        objectness = objectness.view(num_images, -1)

        boxes = []
        for img_idx in range(num_images):
            img_objectness = objectness[img_idx]
            img_proposals = proposals[img_idx]

            # Get top pre-NMS proposals
            pre_nms_top_n = self.pre_nms_top_n[
                "training" if self.training else "testing"
            ]
            num_anchors = img_objectness.shape[0]
            pre_nms_top_n = min(pre_nms_top_n, num_anchors)

            top_scores, top_idx = img_objectness.topk(pre_nms_top_n)
            img_proposals = img_proposals[top_idx]

            # Clip proposals to image
            img_proposals = self.clip_boxes_to_image(
                img_proposals, images.image_sizes[img_idx]
            )

            # Remove small boxes
            keep = self.remove_small_boxes(img_proposals, min_size=1e-3)
            img_proposals = img_proposals[keep]
            top_scores = top_scores[keep]

            # Apply NMS
            keep = nms(img_proposals, top_scores, self.nms_thresh)

            # Keep top post-NMS proposals
            post_nms_top_n = self.post_nms_top_n[
                "training" if self.training else "testing"
            ]
            keep = keep[:post_nms_top_n]

            boxes.append(img_proposals[keep])

        losses = {}
        if self.training and targets is not None:
            # Compute RPN losses
            losses = self.compute_loss(objectness, pred_bbox_deltas, anchors, targets)

        return boxes, losses

    def compute_loss(self, objectness, pred_bbox_deltas, anchors, targets):
        """Compute RPN losses"""
        labels = []
        regression_targets = []

        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image["boxes"]

            if gt_boxes.numel() == 0:
                # No ground truth boxes
                device = anchors_per_image.device
                labels_per_image = torch.zeros(
                    anchors_per_image.shape[0], dtype=torch.float32, device=device
                )
                regression_targets_per_image = torch.zeros(
                    anchors_per_image.shape[0], 4, dtype=torch.float32, device=device
                )
            else:
                # Match anchors to ground truth
                match_quality_matrix = box_iou(gt_boxes, anchors_per_image)
                matched_idxs = self.proposal_matcher(match_quality_matrix)

                # Get labels (1 = positive, 0 = negative, -1 = ignore)
                labels_per_image = matched_idxs >= 0
                labels_per_image = labels_per_image.to(dtype=torch.float32)

                # Set ignored anchors to -1
                bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
                labels_per_image[bg_indices] = 0.0

                inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1.0

                # Get regression targets for positive anchors
                matched_gt_boxes = gt_boxes[matched_idxs.clamp(min=0)]
                regression_targets_per_image = self.box_coder.encode(
                    [matched_gt_boxes], [anchors_per_image]
                )
                if isinstance(regression_targets_per_image, tuple):
                    regression_targets_per_image = regression_targets_per_image[0]

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        # Sample positive and negative anchors
        sampled_pos_inds, sampled_neg_inds = self.sampler(labels)
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        # Compute losses
        objectness = objectness.flatten()
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        # Classification loss
        box_cls_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds]
        )

        # Regression loss (only for positive anchors)
        if sampled_pos_inds.numel() > 0:
            box_reg_loss = F.smooth_l1_loss(
                pred_bbox_deltas.reshape(-1, 4)[sampled_pos_inds],
                regression_targets.reshape(-1, 4)[sampled_pos_inds],
                beta=1.0 / 9,
                reduction="sum",
            ) / (sampled_inds.numel())
        else:
            box_reg_loss = torch.tensor(0.0).to(pred_bbox_deltas.device)

        return {"rpn_cls_loss": box_cls_loss, "rpn_reg_loss": box_reg_loss}

    def clip_boxes_to_image(self, boxes, size):
        """Clip boxes to image boundaries"""
        dim = boxes.dim()
        boxes_x = boxes[..., 0::2]
        boxes_y = boxes[..., 1::2]
        height, width = size

        boxes_x = boxes_x.clamp(min=0, max=width)
        boxes_y = boxes_y.clamp(min=0, max=height)

        clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
        return clipped_boxes.reshape(boxes.shape)

    def remove_small_boxes(self, boxes, min_size):
        """Remove boxes with small area"""
        ws = boxes[:, 2] - boxes[:, 0]
        hs = boxes[:, 3] - boxes[:, 1]
        keep = (ws >= min_size) & (hs >= min_size)
        return keep
