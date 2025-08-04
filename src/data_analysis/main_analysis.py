# src/data_analysis/main_analysis.py
import argparse
from pathlib import Path
import sys

# Import all modules
from setup_workspace import setup_workspace
from data_validator import DataValidator
from statistical_analyzer import StatisticalAnalyzer
from data_cleaner import DataCleaner
from data_visualizer import DataVisualizer
from report_generator import ReportGenerator


def run_complete_analysis(
    dataset_path: str, clean_data: bool = True, visualize: bool = True
):
    """Run complete data analysis pipeline"""

    print("=" * 50)
    print("Starting Dataset Analysis Pipeline")
    print("=" * 50)

    # Step 1: Setup workspace
    print("\n[Step 1/6] Setting up workspace...")
    setup_workspace()

    # Step 2: Data validation
    print("\n[Step 2/6] Validating data...")
    validator = DataValidator(dataset_path)

    for split in ["train", "valid"]:
        print(f"\n--- Analyzing {split} split ---")
        validator.validate_images(split)
        validator.check_missing_annotations(split)
        validator.validate_annotations(split)

    validator.save_validation_report()

    # Step 3: Statistical analysis
    print("\n[Step 3/6] Performing statistical analysis...")
    analyzer = StatisticalAnalyzer(dataset_path)

    for split in ["train", "valid"]:
        analyzer.analyze_bounding_boxes(split)
        analyzer.analyze_class_distribution(split)
        analyzer.analyze_expression_completeness(split)

    # Step 4: Data cleaning (if requested)
    if clean_data:
        print("\n[Step 4/6] Cleaning data...")
        cleaner = DataCleaner(dataset_path)

        for split in ["train", "valid"]:
            cleaner.clean_bounding_boxes(split, fix_issues=True)
            cleaner.handle_incomplete_data(split)
    else:
        print("\n[Step 4/6] Skipping data cleaning (--no-clean flag set)")

    # Step 5: Visualization (if requested)
    if visualize:
        print("\n[Step 5/6] Creating visualizations...")
        visualizer = DataVisualizer(dataset_path)

        for split in ["train", "valid"]:
            visualizer.visualize_samples(split, n_samples=10)
            visualizer.create_class_examples_grid(split, examples_per_class=5)
    else:
        print("\n[Step 5/6] Skipping visualizations (--no-viz flag set)")

    # Step 6: Generate report
    print("\n[Step 6/6] Generating reports...")
    report_gen = ReportGenerator()
    report_gen.generate_html_report()

    print("\n" + "=" * 50)
    print("Analysis Complete!")
    print("=" * 50)
    print("\nResults saved in: results/data_analysis/")
    print("- Statistics: results/data_analysis/stats/")
    print("- Plots: results/data_analysis/plots/")
    print("- Reports: results/data_analysis/reports/")
    print("- Sample visualizations: results/data_analysis/sample_visualizations/")
    if clean_data:
        print("- Cleaned data: results/data_analysis/cleaned_data/")
        print("- Incomplete data: results/data_analysis/incomplete/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze dataset for Faster R-CNN training"
    )
    parser.add_argument(
        "--dataset", type=str, default="dataset", help="Path to dataset directory"
    )
    parser.add_argument(
        "--no-clean", action="store_true", help="Skip data cleaning step"
    )
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization step")

    args = parser.parse_args()

    # Check if dataset exists
    if not Path(args.dataset).exists():
        print(f"Error: Dataset directory '{args.dataset}' not found!")
        sys.exit(1)

    # Run analysis
    run_complete_analysis(
        dataset_path=args.dataset,
        clean_data=not args.no_clean,
        visualize=not args.no_viz,
    )
