import json
from statistics import mean, median, stdev
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import argparse
import math
import os

MODEL_DIRS = {
    "Claude 3.6 Sonnet": "results/claude-3-5-sonnet-20241022",
    "Gemini 1.5 Pro": "results/gemini-1.5-pro-002",
    "GPT-4o": "results/gpt-4o-2024-08-06",
}

# Colors are the same as Set1 in matplotlib
# https://gist.github.com/ltiao/c196b64b34c48e244c73afbb9889b8e7
MODEL_COLORS = {
    "Claude 3.6 Sonnet": "#E41A1C",
    "Gemini 1.5 Pro": "#377EB8",
    "GPT-4o": "#4DAF4A",
}


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


def plot_giou_distribution(
    gious_by_model: Dict[str, List[float]], output_path: str
) -> None:
    """Create side-by-side histograms of GIoU values for all models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    bucket_size = 0.05
    buckets = [round(i * bucket_size - 1.0, 2) for i in range(41)]

    for idx, (model_name, gious) in enumerate(gious_by_model.items()):
        counts = [0] * (len(buckets) - 1)
        for giou in gious:
            for i in range(len(buckets) - 1):
                if buckets[i] <= giou < buckets[i + 1]:
                    counts[i] += 1
                    break

        axes[idx].bar(
            buckets[:-1],
            counts,
            width=bucket_size,
            align="edge",
            color=MODEL_COLORS[model_name],
        )

        axes[idx].set_xlabel("GIoU")
        axes[idx].set_ylabel("Count")
        axes[idx].set_title(f"{model_name}\nGIoU distribution")
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_area_ratio_distribution(area_ratios: List[float], output_path: str) -> None:
    """Create a histogram of area ratios."""
    bins = [0.01, 0.0215, 0.0464, 0.1, 0.215, 0.464, 1.0, 2.15, 4.64, 10, 21.5]

    plt.hist(
        area_ratios,
        bins=bins,
        edgecolor="black",
        alpha=0.5,
    )

    plt.xscale("log")
    plt.xlabel("Area ratio (%) - log scale")
    plt.ylabel("Count")
    plt.title("Distribution of ground truth area ratios")
    plt.grid(True, which="major", alpha=0.3)
    plt.xticks(bins, [f"{x:.3g}" for x in bins])
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_correlations(
    data_by_model: Dict[str, Tuple[List[float], List[float], List[float]]],
    output_path: str,
) -> None:
    """Create a 3x3 grid of correlation plots comparing all models."""
    fig, axes = plt.subplots(3, 3, figsize=(20, 20))

    plot_titles = [
        "GIoU vs Area Ratio",
        "Description Grade vs Area Ratio",
        "Description Grade vs GIoU",
    ]
    alpha = 0.9

    for col, (model_name, (gious, grades, area_ratios)) in enumerate(
        data_by_model.items()
    ):  # Changed row to col
        # Plot 1: GIoU vs log area ratio
        axes[0, col].scatter(  # Swapped indices
            area_ratios, gious, alpha=alpha, color=MODEL_COLORS[model_name]
        )
        axes[0, col].set_title(
            f"{plot_titles[0]} - {model_name}\nR² = {calculate_r_squared(area_ratios, gious):.2f}"
        )
        axes[0, col].set_ylim(-1, 1)
        axes[0, col].set_xscale("log")
        axes[0, col].set_ylabel("GIoU")
        axes[0, col].set_xlabel("Area ratio (%)")

        # Plot 2: Description grade vs area ratio
        axes[1, col].scatter(  # Swapped indices
            area_ratios, grades, alpha=alpha, color=MODEL_COLORS[model_name]
        )
        axes[1, col].set_title(f"{plot_titles[1]} - {model_name}")
        axes[1, col].set_yticks(range(4))
        axes[1, col].set_xscale("log")
        axes[1, col].set_ylabel("Description grade")
        axes[1, col].set_xlabel("Area ratio (%)")

        # Plot 3: Description grade vs GIoU
        axes[2, col].scatter(
            gious, grades, alpha=alpha, color=MODEL_COLORS[model_name]
        )  # Swapped indices
        axes[2, col].set_title(f"{plot_titles[2]} - {model_name}")
        axes[2, col].set_yticks(range(6))
        axes[2, col].set_ylabel("Description grade")
        axes[2, col].set_xlabel("GIoU")
        axes[2, col].set_xlim(-1, 1)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Process grading JSON and create comparative plots across models."
    )
    args = parser.parse_args()

    # Read annotations once
    with open("annotations.json", "r") as f:
        annotations_data = json.load(f)

    # Collect data for each model
    data_by_model = {}

    for model_name, results_dir in MODEL_DIRS.items():
        grading_path = os.path.join(results_dir, "grading.json")
        with open(grading_path, "r") as f:
            grading_data = json.load(f)

        gious = []
        grades = []
        area_ratios = []

        for img_id, img_data in grading_data.items():
            for obj_id, grading in img_data.items():
                if grading["status"] == "predicted":
                    gious.append(grading["giou"])
                    grades.append(grading["description_grade"])
                    area_ratios.append(
                        calculate_area_ratio(annotations_data[img_id][obj_id]["bbox"])
                    )

        data_by_model[model_name] = (gious, grades, area_ratios)

    # Create output directory
    output_dir = "stats"
    os.makedirs(output_dir, exist_ok=True)

    # Calculate area ratios once for all annotations
    area_ratios = [
        calculate_area_ratio(annotation["bbox"])
        for val in annotations_data.values()
        for annotation in val.values()
    ]

    # Create plots
    plot_correlations(data_by_model, os.path.join(output_dir, "correlations.png"))
    plot_giou_distribution(
        {model: data[0] for model, data in data_by_model.items()},
        os.path.join(output_dir, "giou_distribution.png"),
    )
    plot_area_ratio_distribution(
        area_ratios, os.path.join(output_dir, "area_ratio_distribution.png")
    )


if __name__ == "__main__":
    main()
