import argparse
import json
import os
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
from statistics import mean, median, stdev


def get_basic_stats(numbers: List[float]) -> Dict[str, float]:
    """Calculate basic statistics without numpy."""
    if not numbers:
        return {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0}

    return {
        "mean": mean(numbers),
        "median": median(numbers),
        "std": stdev(numbers) if len(numbers) > 1 else 0,
        "min": min(numbers),
        "max": max(numbers),
    }


def calculate_success_rates(gious: List[float], thresholds: List[float]) -> List[float]:
    """Calculate success rate at each threshold."""
    return [
        sum(1 for giou in gious if giou >= threshold) / len(gious)
        for threshold in thresholds
    ]


def calculate_correlation(x: List[float], y: List[float]) -> float:
    """Calculate Pearson correlation coefficient using basic math."""
    if len(x) != len(y) or len(x) < 2:
        return 0

    mean_x = mean(x)
    mean_y = mean(y)

    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denominator = (
        sum((xi - mean_x) ** 2 for xi in x) * sum((yi - mean_y) ** 2 for yi in y)
    ) ** 0.5

    return numerator / denominator if denominator != 0 else 0


def analyze_detection_performance(
    data: Dict[str, Dict[str, Dict[str, Any]]],
) -> Tuple[Dict[str, float], List[float], Any]:
    """
    Analyze single-object detection performance across different images.

    Args:
        data: Dictionary with structure {image_name: {object_name: {metrics}}}
    """
    # Extract data into lists
    gious = []
    grades = []

    for image_data in data.values():
        for metrics in image_data.values():
            if "giou" in metrics:
                gious.append(metrics["giou"])
                if "description_grade" in metrics:
                    grades.append(metrics["description_grade"])

    # Calculate basic statistics
    giou_stats = get_basic_stats(gious)

    # Calculate success rates
    total = len(gious)
    above_50 = sum(1 for giou in gious if giou > 0.5)
    above_75 = sum(1 for giou in gious if giou > 0.75)

    stats_summary = {
        **giou_stats,
        "total_predictions": total,
        "predictions_above_50": above_50,
        "predictions_above_75": above_75,
        "success_rate_50": above_50 / total if total > 0 else 0,
        "success_rate_75": above_75 / total if total > 0 else 0,
    }

    # Calculate correlation if grades exist
    if grades:
        # Filter out any None values
        valid_pairs = [
            (g, d) for g, d in zip(gious, grades) if g is not None and d is not None
        ]
        if valid_pairs:
            valid_gious, valid_grades = zip(*valid_pairs)
            stats_summary["desc_giou_correlation"] = calculate_correlation(
                list(valid_gious), list(valid_grades)
            )

    # Create visualizations
    plt.figure(figsize=(15, 5))

    # 1. GIoU Distribution
    plt.subplot(131)
    plt.hist(gious, bins=20, edgecolor="black")
    plt.axvline(x=0.5, color="r", linestyle="--", label="0.5 threshold")
    plt.axvline(x=0.75, color="g", linestyle="--", label="0.75 threshold")
    plt.title("Distribution of GIoU Scores")
    plt.xlabel("GIoU")
    plt.ylabel("Count")
    plt.legend()

    # 2. Description Grade vs GIoU
    if grades:
        plt.subplot(132)
        plt.scatter(gious, grades)
        plt.xlabel("GIoU")
        plt.ylabel("Description Grade")
        plt.title("Description Grade vs GIoU")

    # 3. Success Rate Plot
    plt.subplot(133)
    thresholds = [i / 10 for i in range(11)]  # 0 to 1 in steps of 0.1
    success_rates = calculate_success_rates(gious, thresholds)
    plt.plot(thresholds, success_rates)
    plt.title("Success Rate vs GIoU Threshold")
    plt.xlabel("GIoU Threshold")
    plt.ylabel("Success Rate")

    plt.tight_layout()

    return stats_summary, gious, plt.gcf()


def print_analysis(stats_summary: Dict[str, float]):
    """Print analysis results in a readable format."""
    print("\nDetection Performance Analysis:")
    print(f"Total predictions analyzed: {stats_summary['total_predictions']}")

    print("\nGIoU Statistics:")
    print(f"Mean GIoU: {stats_summary['mean']:.3f}")
    print(f"Median GIoU: {stats_summary['median']:.3f}")
    print(f"Standard Deviation: {stats_summary['std']:.3f}")
    print(f"Range: {stats_summary['min']:.3f} to {stats_summary['max']:.3f}")

    print("\nSuccess Rates:")
    print(f"Above 0.5 GIoU: {stats_summary['success_rate_50']*100:.1f}%")
    print(f"Above 0.75 GIoU: {stats_summary['success_rate_75']*100:.1f}%")

    if "desc_giou_correlation" in stats_summary:
        print("\nDescription Grade Correlation:")
        print(f"Correlation with GIoU: {stats_summary['desc_giou_correlation']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze detection performance")
    parser.add_argument(
        "results_dir", type=str, help="Path to directory containing grading.json"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output filename for the plot",
        default="ap.png",
    )
    args = parser.parse_args()

    grading_json_path = os.path.join(args.results_dir, "grading.json")
    results_json_path = os.path.join(args.results_dir, "results.json")

    with open(grading_json_path, "r") as f:
        grading_data = json.load(f)
    with open(results_json_path, "r") as f:
        results_data = json.load(f)

    stats_summary, gious, fig = analyze_detection_performance(grading_data)
    print_analysis(stats_summary)

    plt.savefig(args.output)
    plt.close()
