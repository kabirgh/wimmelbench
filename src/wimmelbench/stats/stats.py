import json
from statistics import mean, median, stdev
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import argparse
import math
import os


def calculate_area_ratio(bbox: List[float]) -> float:
    # bbox format is [x1, y1, x2, y2]
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return width * height * 100  # %


def calculate_r_squared(x, y):
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(i * j for i, j in zip(x, y))
    sum_x2 = sum(i * i for i in x)
    sum_y2 = sum(i * i for i in y)

    # Calculate r (correlation coefficient)
    numerator = n * sum_xy - sum_x * sum_y
    denominator = math.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))
    r = numerator / denominator if denominator != 0 else 0

    # Return r²
    return r * r


def plot_giou_distribution(gious: List[float], output_path: str) -> None:
    """
    Create a histogram of GIoU values with 0.05 size buckets.

    Args:
        gious: List of GIoU values
        output_path: Path where to save the plot
    """
    # Create buckets from -1 to 1 with 0.05 intervals
    bucket_size = 0.05
    buckets = [
        round(i * bucket_size - 1.0, 2) for i in range(41)
    ]  # 41 points to cover -1 to 1

    # Count GIoUs in each bucket
    counts = [0] * (len(buckets) - 1)
    for giou in gious:
        for i in range(len(buckets) - 1):
            if buckets[i] <= giou < buckets[i + 1]:
                counts[i] += 1
                break

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.bar(buckets[:-1], counts, width=bucket_size, align="edge")
    plt.xlabel("GIoU")
    plt.ylabel("Count")
    plt.title("Distribution of GIoU Values")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path)
    plt.close()


def plot_area_ratio_distribution(area_ratios: List[float], output_path: str) -> None:
    # Create fixed logarithmic bins
    bins = [0.01, 0.0215, 0.0464, 0.1, 0.215, 0.464, 1.0, 2.15, 4.64, 10, 21.5]

    plt.hist(area_ratios, bins=bins, edgecolor="black")
    plt.xscale("log")
    plt.xlabel("Area Ratio (%) - Log Scale")
    plt.ylabel("Count")
    plt.title("Distribution of Area Ratios")
    # Add custom grid lines at the major ticks
    plt.grid(True, which="major", alpha=0.3)
    # Set x-axis ticks explicitly
    plt.xticks(bins, [f"{x:.3g}" for x in bins])

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_correlations(
    gious: List[float], grades: List[float], area_ratios: List[float], output_path: str
) -> None:
    """
    Create a figure with two subplots:
    1. GIoU vs log area ratio
    2. Description grade vs area ratio

    Args:
        gious: List of GIoU values
        grades: List of description grades
        area_ratios: List of area ratios
        output_path: Path where to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: GIoU vs log area ratio
    ax1.scatter(area_ratios, gious, alpha=0.5)
    ax1.set_xscale("log")
    ax1.set_xlabel("Area Ratio (%) - Log Scale")
    ax1.set_ylabel("GIoU")
    ax1.set_title(
        f"GIoU vs Area Ratio (R² = {calculate_r_squared(area_ratios, gious):.3f})"
    )
    ax1.grid(True, alpha=0.3)

    # Plot 2: Description grade vs area ratio
    ax2.scatter(area_ratios, grades, alpha=0.5)
    ax2.set_xscale("log")
    ax2.set_xlabel("Area Ratio (%) - Log Scale")
    ax2.set_ylabel("Description Grade")
    ax2.set_title(
        f"Description Grade vs Area Ratio (R² = {calculate_r_squared(area_ratios, grades):.3f})"
    )
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Process grading JSON and create GIoU vs Area Ratio plot."
    )
    parser.add_argument(
        "results_dir", help="Directory containing grading.json and results.json"
    )
    args = parser.parse_args()

    # Construct paths
    grading_path = os.path.join(args.results_dir, "grading.json")
    predictions_path = os.path.join(args.results_dir, "results.json")
    output_dir = os.path.join("stats", args.results_dir.split("/")[-1])
    os.makedirs(output_dir, exist_ok=True)

    # Read JSON files
    with open("annotations.json", "r") as f:
        annotations_data = json.load(f)
    with open(grading_path, "r") as f:
        grading_data = json.load(f)
    with open(predictions_path, "r") as f:
        predictions_data = json.load(f)

    # Extract GIoU values, grades, and area ratios for objects that were predicted
    gious = []
    grades = []
    area_ratios = []

    for img_id, img_data in grading_data.items():
        for obj_id, grading in img_data.items():
            if grading["status"] == "predicted":
                gious.append(grading["giou"])
                grades.append(grading["description_grade"])
                # Get corresponding annotation bbox
                area_ratios.append(
                    calculate_area_ratio(annotations_data[img_id][obj_id]["bbox"])
                )

    # Create correlation plots
    output_path = os.path.join(output_dir, "correlations.png")
    plot_correlations(gious, grades, area_ratios, output_path)

    # Extract GIoU values and create plot
    gious = [
        grading["giou"]
        for val in grading_data.values()
        for grading in val.values()
        if grading["status"] == "predicted"
    ]
    output_path = os.path.join(output_dir, "giou_distribution.png")
    plot_giou_distribution(gious, output_path)

    # Create a plot of the distribution of the area ratios
    area_ratios = [
        calculate_area_ratio(annotation["bbox"])
        for val in annotations_data.values()
        for annotation in val.values()
    ]
    output_path = os.path.join(output_dir, "area_ratio_distribution.png")
    plot_area_ratio_distribution(area_ratios, output_path)


if __name__ == "__main__":
    main()
