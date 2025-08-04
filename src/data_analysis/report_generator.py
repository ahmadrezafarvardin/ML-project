# src/data_analysis/report_generator.py
import json
from pathlib import Path
from datetime import datetime
import pandas as pd


class ReportGenerator:
    def __init__(self, results_path: str = "results/data_analysis"):
        self.results_path = Path(results_path)

    def generate_html_report(self):
        """Generate comprehensive HTML report"""
        print("\nGenerating HTML report...")

        # Load all statistics
        stats = {}
        stats_dir = self.results_path / "stats"

        for json_file in stats_dir.glob("*.json"):
            with open(json_file, "r") as f:
                stats[json_file.stem] = json.load(f)

        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dataset Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .warning {{ color: #ff6b6b; font-weight: bold; }}
                .success {{ color: #51cf66; font-weight: bold; }}
                .info {{ color: #339af0; }}
                img {{ max-width: 100%; height: auto; margin: 20px 0; }}
                .grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }}
            </style>
        </head>
        <body>
            <h1>Dataset Analysis Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>1. Data Validation Summary</h2>
            {self._create_validation_summary(stats)}
            
            <h2>2. Dataset Statistics</h2>
            {self._create_statistics_summary(stats)}
            
            <h2>3. Visualizations</h2>
            {self._create_visualizations_section()}
            
            <h2>4. Recommendations</h2>
            {self._create_recommendations(stats)}
            
            <h2>5. Data Quality Issues</h2>
            {self._create_issues_summary(stats)}
        </body>
        </html>
        """

        # Save report
        report_path = self.results_path / "reports" / "analysis_report.html"
        report_path.parent.mkdir(exist_ok=True)

        with open(report_path, "w") as f:
            f.write(html_content)

        print(f"HTML report saved to: {report_path}")

        # Also create a markdown summary
        self._create_markdown_summary(stats)

    def _create_validation_summary(self, stats):
        """Create validation summary table"""
        validation = stats.get("validation_report", {})

        html = """
        <table>
            <tr>
                <th>Check</th>
                <th>Train</th>
                <th>Validation</th>
                <th>Status</th>
            </tr>
        """

        checks = [
            ("Corrupted Images", "corrupted_images"),
            ("Missing Annotations", "missing_annotations"),
            ("Empty Annotations", "empty_annotations"),
            ("Missing Expressions", "missing_expressions"),
        ]

        for check_name, key in checks:
            train_count = len([x for x in validation.get(key, []) if "train" in str(x)])
            valid_count = len([x for x in validation.get(key, []) if "valid" in str(x)])

            status = (
                '<span class="success">✓</span>'
                if train_count + valid_count == 0
                else '<span class="warning">⚠</span>'
            )

            html += f"""
            <tr>
                <td>{check_name}</td>
                <td>{train_count}</td>
                <td>{valid_count}</td>
                <td>{status}</td>
            </tr>
            """

        html += "</table>"
        return html

    def _create_statistics_summary(self, stats):
        """Create statistics summary"""
        train_box = stats.get("train_box_statistics", {})
        train_class = stats.get("train_class_statistics", {})

        html = f"""
        <h3>Bounding Box Statistics</h3>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Total Images</td>
                <td>{train_box.get('total_images', 'N/A')}</td>
            </tr>
            <tr>
                <td>Total Boxes</td>
                <td>{train_box.get('total_boxes', 'N/A')}</td>
            </tr>
            <tr>
                <td>Avg Boxes per Image</td>
                <td>{train_box.get('boxes_per_image', {}).get('mean', 'N/A'):.2f}</td>
            </tr>
            <tr>
                <td>Box Area (mean ± std)</td>
                <td>{train_box.get('box_dimensions', {}).get('area', {}).get('mean', 0):.1f} ± 
                    {train_box.get('box_dimensions', {}).get('area', {}).get('std', 0):.1f}</td>
            </tr>
        </table>
        
        <h3>Class Distribution</h3>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Total Classes</td>
                <td>{train_class.get('total_classes', 'N/A')}</td>
            </tr>
            <tr>
                <td>Total Annotations</td>
                <td>{train_class.get('total_annotations', 'N/A')}</td>
            </tr>
            <tr>
                <td>Rare Classes (< 10% avg)</td>
                <td>{len(train_class.get('imbalanced_classes', {}).get('rare', []))}</td>
            </tr>
            <tr>
                <td>Common Classes (> 200% avg)</td>
                <td>{len(train_class.get('imbalanced_classes', {}).get('common', []))}</td>
            </tr>
        </table>
        """

        return html

    def _create_visualizations_section(self):
        """Create visualizations section"""
        plots_dir = self.results_path / "plots"

        html = '<div class="grid">'

        for plot_file in sorted(plots_dir.glob("*.png")):
            html += f"""
            <div>
                <h4>{plot_file.stem.replace('_', ' ').title()}</h4>
                <img src="../plots/{plot_file.name}" alt="{plot_file.stem}">
            </div>
            """

        html += "</div>"
        return html

    def _create_recommendations(self, stats):
        """Create recommendations based on analysis"""
        recommendations = []

        # Check for class imbalance
        train_class = stats.get("train_class_statistics", {})
        if train_class.get("imbalanced_classes", {}).get("rare"):
            recommendations.append(
                "Consider data augmentation or oversampling for rare classes: "
                + ", ".join(train_class["imbalanced_classes"]["rare"][:5])
            )

        # Check for missing expressions
        train_expr = stats.get("train_expression_statistics", {})
        if train_expr.get("completeness_percentage", 100) < 50:
            recommendations.append(
                f"Only {train_expr['completeness_percentage']:.1f}% of training data has expressions. "
                "Consider if this impacts your model requirements."
            )

        # Check for outliers
        train_box = stats.get("train_box_statistics", {})
        if train_box.get("outliers", {}).get("percentage", 0) > 5:
            recommendations.append(
                f"{train_box['outliers']['percentage']:.1f}% of boxes are outliers. "
                "Review and potentially clean these annotations."
            )

        html = "<ul>"
        for rec in recommendations:
            html += f"<li>{rec}</li>"
        html += "</ul>"

        return (
            html
            if recommendations
            else "<p class='success'>No major issues detected!</p>"
        )

    def _create_issues_summary(self, stats):
        """Create summary of data quality issues"""
        cleaning_log = stats.get("train_cleaning_log", {})

        html = """
        <table>
                        <tr>
                <th>Issue Type</th>
                <th>Count</th>
                <th>Action Taken</th>
            </tr>
        """

        issues = [
            (
                "Fixed Boxes",
                len(cleaning_log.get("fixed_boxes", [])),
                "Coordinates adjusted to image bounds",
            ),
            (
                "Removed Boxes",
                len(cleaning_log.get("removed_boxes", [])),
                "Deleted invalid annotations",
            ),
            (
                "Moved Incomplete",
                len(cleaning_log.get("moved_incomplete", [])),
                "Moved to incomplete folder",
            ),
            (
                "Fixed Expressions",
                len(cleaning_log.get("fixed_expressions", [])),
                "Set to null value",
            ),
        ]

        for issue_name, count, action in issues:
            html += f"""
            <tr>
                <td>{issue_name}</td>
                <td>{count}</td>
                <td>{action}</td>
            </tr>
            """

        html += "</table>"
        return html

    def _create_markdown_summary(self, stats):
        """Create a markdown summary report"""
        md_content = f"""# Dataset Analysis Summary

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Quick Statistics

### Training Set
- **Total Images**: {stats.get('train_box_statistics', {}).get('total_images', 'N/A')}
- **Total Annotations**: {stats.get('train_box_statistics', {}).get('total_boxes', 'N/A')}
- **Total Classes**: {stats.get('train_class_statistics', {}).get('total_classes', 'N/A')}
- **Expression Completeness**: {stats.get('train_expression_statistics', {}).get('completeness_percentage', 0):.1f}%

### Validation Set
- **Total Images**: {stats.get('valid_box_statistics', {}).get('total_images', 'N/A')}
- **Total Annotations**: {stats.get('valid_box_statistics', {}).get('total_boxes', 'N/A')}

## Key Findings

### Data Quality Issues
"""

        validation = stats.get("validation_report", {})
        if validation.get("corrupted_images"):
            md_content += (
                f"- **Corrupted Images**: {len(validation['corrupted_images'])}\n"
            )
        if validation.get("missing_annotations"):
            md_content += (
                f"- **Missing Annotations**: {len(validation['missing_annotations'])}\n"
            )
        if validation.get("empty_annotations"):
            md_content += (
                f"- **Empty Annotations**: {len(validation['empty_annotations'])}\n"
            )

        md_content += """
### Class Distribution
"""

        train_class = stats.get("train_class_statistics", {})
        if train_class.get("imbalanced_classes", {}).get("rare"):
            md_content += f"- **Rare Classes**: {', '.join(train_class['imbalanced_classes']['rare'][:10])}\n"
        if train_class.get("imbalanced_classes", {}).get("common"):
            md_content += f"- **Common Classes**: {', '.join(train_class['imbalanced_classes']['common'][:5])}\n"

        md_content += """
## Recommendations

1. **Data Augmentation**: Focus on rare classes to balance the dataset
2. **Quality Control**: Review and fix identified annotation issues
3. **Expression Handling**: Decide on strategy for missing mathematical expressions
4. **Outlier Review**: Manually check boxes flagged as outliers

## Next Steps

1. Apply data cleaning scripts to fix identified issues
2. Consider synthetic data generation for underrepresented classes
3. Implement data augmentation pipeline
4. Set up continuous validation for new data additions
"""

        # Save markdown report
        md_path = self.results_path / "reports" / "analysis_summary.md"
        with open(md_path, "w") as f:
            f.write(md_content)

        print(f"Markdown summary saved to: {md_path}")
