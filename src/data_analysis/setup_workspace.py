# src/data_analysis/setup_workspace.py
import os
from pathlib import Path


def setup_workspace():
    """Create organized directory structure for results"""
    directories = [
        "results/data_analysis/plots",
        "results/data_analysis/stats",
        "results/data_analysis/cleaned_data",
        "results/data_analysis/incomplete",
        "results/data_analysis/bad_images",
        "results/data_analysis/reports",
        "results/data_analysis/sample_visualizations",
    ]

    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Created: {dir_path}")


if __name__ == "__main__":
    setup_workspace()
