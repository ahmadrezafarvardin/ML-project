# src/utils/check_nms_settings.py
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.fasterrcnn.model.faster_rcnn import FasterRCNN


def check_model_settings():
    model = FasterRCNN(num_classes=2)

    print("Current Model Settings:")
    print(f"RPN NMS threshold: {model.rpn.nms_thresh}")
    print(f"ROI Head NMS threshold: {model.roi_heads.nms_thresh}")
    print(f"ROI Head score threshold: {model.roi_heads.score_thresh}")
    print(f"Max detections per image: {model.roi_heads.detections_per_img}")


if __name__ == "__main__":
    check_model_settings()
