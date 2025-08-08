# src/models/yolo/yolo_character_detector.py
"""
Simple YOLO wrapper for character detection
"""

from ultralytics import YOLO
import os
from pathlib import Path


class YOLOCharacterDetector:
    def __init__(self):
        self.model = YOLO(f"yolov8n.pt")

    def train(self, **kwargs):
        """Train the model with default settings for character detection"""
        default_args = {
            "epochs": 100,
            "imgsz": 640,
            "batch": 8,
            "device": 0,
            "workers": 4,
            "amp": False,
            "project": "results/yolo/yolo_runs",
            "name": "character_detection",
            "exist_ok": True,
            "patience": 20,
            "save": True,
            "save_period": 10,
            "val": True,
            "plots": True,
            "cache": False,
            "rect": False,
            "mosaic": 0.5,
            "mixup": 0.0,
            "copy_paste": 0.0,
        }
        # Override with any provided arguments
        default_args.update(kwargs)

        return self.model.train(**default_args)

    def predict(self, source, **kwargs):
        """Run inference"""
        return self.model(source, **kwargs)

    def val(self, **kwargs):
        """Validate the model"""
        return self.model.val(**kwargs)
