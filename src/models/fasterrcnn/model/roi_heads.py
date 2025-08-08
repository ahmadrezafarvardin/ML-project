# src/models/fasterrcnn/roi_heads.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign, box_iou, nms
from torchvision.models.detection._utils import (
    BoxCoder,
    Matcher,
    BalancedPositiveNegativeSampler,
)


class FastRCNNHead(nn.Module):
    """Fast R-CNN head for classification and bbox regression"""

    def __init__(self, in_channels, num_classes):
        super().__init__()

        # ROI feature size after pooling
        self.roi_size = 7
        roi_feat_size = in_channels * self.roi_size * self.roi_size

        # Two FC layers
        self.fc1 = nn.Linear(roi_feat_size, 1024)
        self.fc2 = nn.Linear(1024, 1024)

        # Classification and regression heads
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 4)

        # Initialize weights
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.cls_score.bias, 0)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        # Flatten ROI features
        x = x.flatten(start_dim=1)

        # FC layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Get predictions
        cls_scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return cls_scores, bbox_deltas


class ROIHeads(nn.Module):
    """Complete ROI heads module"""

    def __init__(
        self,
        in_channels,
        num_classes,
        fg_iou_thresh=0.5,
        bg_iou_thresh=0.5,
        batch_size_per_image=512,
        positive_fraction=0.25,
        score_thresh=0.05,
        nms_thresh=0.5,
        detections_per_img=100,
    ):
        super().__init__()

        # ROI Align layer
        self.box_roi_pool = RoIAlign(
            output_size=7, spatial_scale=1 / 32, sampling_ratio=2
        )

        # Fast R-CNN head
        self.box_head = FastRCNNHead(in_channels, num_classes)

        # Box coder
        self.box_coder = BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))

        # Matcher
        self.proposal_matcher = Matcher(
            high_threshold=fg_iou_thresh,
            low_threshold=bg_iou_thresh,
            allow_low_quality_matches=False,
        )

        # Sampler
        self.sampler = BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_fraction
        )

        # Training parameters
        self.fg_iou_thresh = fg_iou_thresh
        self.bg_iou_thresh = bg_iou_thresh
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

        # Inference parameters
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

        self.num_classes = num_classes

    def forward(self, features, proposals, image_shapes, targets=None):
        """
        Args:
            features: Feature maps from backbone
            proposals: Box proposals from RPN
            image_shapes: Original image sizes
            targets: Ground truth (training only)

        Returns:
            result: List of detected boxes, labels, scores
            losses: Detection losses (training only)
        """
        if self.training:
            # Sample proposals for training
            proposals, labels, regression_targets = self.select_training_samples(
                proposals, targets
            )

        # Extract ROI features
        box_features = self.box_roi_pool(features, proposals)

        # Get predictions
        class_logits, box_regression = self.box_head(box_features)

        result = []
        losses = {}

        if self.training:
            # Compute losses
            losses = self.compute_loss(
                class_logits, box_regression, labels, regression_targets
            )
        else:
            # Postprocess for inference
            boxes, scores, labels = self.postprocess_detections(
                class_logits, box_regression, proposals, image_shapes
            )

            for i in range(len(image_shapes)):
                result.append(
                    {"boxes": boxes[i], "labels": labels[i], "scores": scores[i]}
                )

        return result, losses

    def select_training_samples(self, proposals, targets):
        """Sample proposals for training"""
        self.check_targets(targets)

        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # Append ground-truth boxes to proposals
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # Sample proposals
        labels = []
        regression_targets = []
        sampled_proposals = []

        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(
            proposals, gt_boxes, gt_labels
        ):
            if gt_boxes_in_image.numel() == 0:
                # No GT boxes
                device = proposals_in_image.device
                labels_in_image = torch.zeros(
                    proposals_in_image.shape[0], dtype=torch.int64, device=device
                )
                regression_targets_in_image = torch.zeros(
                    proposals_in_image.shape[0], 4, dtype=torch.float32, device=device
                )
                sampled_inds = torch.arange(proposals_in_image.shape[0], device=device)
            else:
                # Match proposals to ground truth
                match_quality_matrix = box_iou(gt_boxes_in_image, proposals_in_image)
                matched_idxs = self.proposal_matcher(match_quality_matrix)

                labels_in_image = gt_labels_in_image[matched_idxs.clamp(min=0)]
                bg_inds = matched_idxs < 0
                labels_in_image[bg_inds] = 0

                # Subsample using labels
                sampled_inds = self.subsample(labels_in_image)
                proposals_in_image = proposals_in_image[sampled_inds]
                matched_idxs = matched_idxs[sampled_inds]
                labels_in_image = labels_in_image[sampled_inds]

                # Encode boxes
                matched_gt_boxes = gt_boxes_in_image[matched_idxs.clamp(min=0)]
                # Fix: Handle the encode return value properly
                encoded = self.box_coder.encode(
                    [matched_gt_boxes], [proposals_in_image]
                )
                # If encode returns a tuple, take the first element
                if isinstance(encoded, tuple):
                    regression_targets_in_image = encoded[0]
                else:
                    regression_targets_in_image = encoded

            labels.append(labels_in_image)
            regression_targets.append(regression_targets_in_image)
            sampled_proposals.append(proposals_in_image)

        return sampled_proposals, labels, regression_targets

    def subsample(self, labels):
        """Subsample positive and negative examples"""
        # Returns masks (bool or uint8) of shape [1, N]
        sampled_pos_inds, sampled_neg_inds = self.sampler([labels])
        # Convert masks to indices
        pos_inds = torch.where(sampled_pos_inds[0])[0]
        neg_inds = torch.where(sampled_neg_inds[0])[0]
        sampled_inds = torch.cat([pos_inds, neg_inds], dim=0)
        return sampled_inds

    def add_gt_proposals(self, proposals, gt_boxes):
        """Add ground truth boxes to proposals"""
        proposals_with_gt = []

        for proposals_per_image, gt_boxes_per_image in zip(proposals, gt_boxes):
            proposals_with_gt.append(
                torch.cat([proposals_per_image, gt_boxes_per_image])
            )

        return proposals_with_gt

    def check_targets(self, targets):
        """Check targets format"""
        assert targets is not None
        assert all(["boxes" in t for t in targets])
        assert all(["labels" in t for t in targets])

    def compute_loss(self, class_logits, box_regression, labels, regression_targets):
        """Compute Fast R-CNN losses"""
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        # Classification loss
        classification_loss = F.cross_entropy(
            class_logits,
            labels,
            weight=torch.tensor([0.25, 1.0]).to(
                class_logits.device
            ),  # Higher weight for character class
        )

        # Regression loss (only for foreground)
        sampled_pos_inds_subset = torch.where(labels > 0)[0]
        labels_pos = labels[sampled_pos_inds_subset]
        N, num_classes = class_logits.shape
        box_regression = box_regression.reshape(N, num_classes, 4)

        if sampled_pos_inds_subset.numel() > 0:
            box_loss = F.smooth_l1_loss(
                box_regression[sampled_pos_inds_subset, labels_pos],
                regression_targets[sampled_pos_inds_subset],
                beta=1.0,
                reduction="sum",
            )
            box_loss = box_loss / labels.numel()
        else:
            box_loss = torch.tensor(0.0).to(box_regression.device)

        return {"loss_classifier": classification_loss, "loss_box_reg": box_loss}

    def postprocess_detections(
        self, class_logits, box_regression, proposals, image_shapes
    ):
        """Postprocess predictions for inference"""
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)

        # Split predictions per image
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []

        for boxes, scores, image_shape in zip(
            pred_boxes_list, pred_scores_list, image_shapes
        ):
            # Reshape boxes
            boxes = boxes.reshape(-1, num_classes, 4)
            scores = scores.reshape(-1, num_classes)

            # Create labels for each prediction
            boxes_per_class = []
            scores_per_class = []
            labels_per_class = []

            # Skip background class
            for j in range(1, num_classes):
                inds_mask = scores[:, j] > self.score_thresh

                boxes_j = boxes[inds_mask, j]
                scores_j = scores[inds_mask, j]

                # Clip boxes to image
                boxes_j = self.clip_boxes_to_image(boxes_j, image_shape)

                # NMS
                keep = nms(boxes_j, scores_j, self.nms_thresh)

                boxes_per_class.append(boxes_j[keep])
                scores_per_class.append(scores_j[keep])
                labels_per_class.append(
                    torch.full_like(scores_j[keep], j, dtype=torch.int64)
                )

            # Concatenate all classes
            if boxes_per_class:
                boxes = torch.cat(boxes_per_class, dim=0)
                scores = torch.cat(scores_per_class, dim=0)
                labels = torch.cat(labels_per_class, dim=0)
            else:
                boxes = torch.empty((0, 4), device=device)
                scores = torch.empty((0,), device=device)
                labels = torch.empty((0,), dtype=torch.int64, device=device)

            # Keep only top detections
            if boxes.shape[0] > 0:
                keep = torch.argsort(scores, descending=True)[: self.detections_per_img]
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

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
