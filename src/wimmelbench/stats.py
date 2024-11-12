from collections import defaultdict
import json
from statistics import mean, median, stdev
from typing import Dict, List, Mapping, Tuple
import matplotlib.pyplot as plt
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


def calculate_area_ratio_percentage(bbox: List[float]) -> float:
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

        # Add statistics text box
        stats_text = f"mean: {mean(gious):.2f}\nmedian: {median(gious):.2f}\nstdev: {stdev(gious):.2f}"
        axes[idx].text(
            0.95,
            0.95,
            stats_text,
            transform=axes[idx].transAxes,
            bbox=dict(facecolor="white", alpha=0.8),
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=10,
        )

        axes[idx].set_xlabel("GIoU")
        axes[idx].set_ylabel("Count")
        axes[idx].set_xlim(-1, 1)
        axes[idx].set_ylim(0, 16)
        axes[idx].set_title(f"{model_name}\nGIoU distribution")
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_area_ratio_distribution(area_ratios: List[float], output_path: str) -> None:
    """Create a histogram of area ratios."""
    num_bins, start, end = 8, 0.1, 10
    multiplier = (end / start) ** (1 / num_bins)
    bins = [start * (multiplier**i) for i in range(num_bins + 1)]

    plt.hist(
        area_ratios,
        bins=bins,
        edgecolor="black",
        alpha=0.5,
    )

    plt.xscale("log")
    plt.xlabel("Area ratio (%, log scale)")
    plt.ylabel("Count")
    plt.title("Distribution of ground truth area ratios")
    plt.grid(True, which="major", alpha=0.3)
    # Format x-axis ticks with fewer decimal places for cleaner look
    plt.xticks(bins, [f"{x:.2g}" for x in bins], rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_correlations(
    data_by_model: Dict[str, Tuple[List[float], List[float], List[float]]],
    output_path: str,
) -> None:
    """Create a 3x3 grid of correlation plots comparing all models."""
    fig, axes = plt.subplots(3, 3, figsize=(20, 20))

    alpha = 0.9

    for col, (model_name, (gious, grades, area_ratios)) in enumerate(
        data_by_model.items()
    ):
        # Add model name only to the first row
        axes[0, col].set_title(model_name, pad=10, fontweight="bold", fontsize=16)

        # Plot 1: GIoU vs log area ratio
        axes[0, col].scatter(
            area_ratios,
            gious,
            alpha=alpha,
            color=MODEL_COLORS[model_name],
        )
        # Add R² inside the plot instead of title
        axes[0, col].text(
            0.95,
            0.05,
            f"R² = {calculate_r_squared([math.log(r) for r in area_ratios], gious):.2f}",
            transform=axes[0, col].transAxes,
            bbox=dict(facecolor="white", alpha=0.8),
            horizontalalignment="right",
            verticalalignment="bottom",
        )
        axes[0, col].set_ylim(-1, 1)
        axes[0, col].set_xscale("log")
        axes[0, col].set_ylabel("GIoU")
        axes[0, col].set_xlabel("Area ratio (%, log scale)")

        # Plot 2: Description grade vs area ratio
        axes[1, col].scatter(
            area_ratios, grades, alpha=alpha, color=MODEL_COLORS[model_name]
        )
        axes[1, col].set_yticks(range(4))
        axes[1, col].set_xscale("log")
        axes[1, col].set_ylabel("Description grade")
        axes[1, col].set_xlabel("Area ratio (%, log scale)")

        # Plot 3: Description grade vs GIoU
        axes[2, col].scatter(gious, grades, alpha=alpha, color=MODEL_COLORS[model_name])
        axes[2, col].set_yticks(range(4))
        axes[2, col].set_ylabel("Description grade")
        axes[2, col].set_xlabel("GIoU")
        axes[2, col].set_xlim(-1, 1)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_grade_distribution(
    grades_by_model: Mapping[str, Mapping[int, int]],
    output_path: str,
) -> None:
    """Create a horizontal stacked bar chart showing grade distribution for each model."""
    models = list(grades_by_model.keys())
    grade_labels = ["Correct (2-3)", "Not Found", "Incorrect (0-1)"]
    grade_mapping = {
        -1: "Not Found",
        0: "Incorrect (0-1)",
        1: "Incorrect (0-1)",
        2: "Correct (2-3)",
        3: "Correct (2-3)",
    }

    # Create percentage data
    percentages = []
    for model in models:
        total = sum(grades_by_model[model].values())
        mapped_grades = defaultdict(int)
        for grade, count in grades_by_model[model].items():
            mapped_grades[grade_mapping[grade]] += count
        model_percentages = [
            mapped_grades[grade] / total * 100 for grade in grade_labels
        ]
        percentages.append(model_percentages)

    # Create the stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Colors for each grade
    colors = ["#1e5d86", "#33858d", "#61aa90"]

    # Keep track of left positions for each bar
    lefts = [0] * len(models)

    # Slightly more spacing than default but less than previous
    bar_height = 0.6  # Increased from 0.5 to 0.6

    # Create bars for each grade
    for grade_idx in range(len(grade_labels)):
        values = [p[grade_idx] for p in percentages]
        bars = ax.barh(
            models,
            values,
            left=lefts,
            label=grade_labels[grade_idx],
            color=colors[grade_idx],
            height=bar_height,
        )

        # Update left positions for next set of bars
        lefts = [left + value for left, value in zip(lefts, values)]

        # Add percentage labels below each bar segment
        for idx, rect in enumerate(bars):
            width = rect.get_width()
            if width > 0:  # Only show non-zero values
                bar_middle = rect.get_x() + width / 2
                ax.text(
                    bar_middle,
                    idx + 0.325,
                    f"{width:.0f}%",
                    ha="center",
                    va="bottom",
                )

    # Customize the chart
    ax.set_title("Grade distribution by model", pad=20)
    ax.set_ylabel("Model")
    ax.set_xlabel("Percentage")
    ax.legend(title="Grades", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    # Read annotations once
    with open("annotations.json", "r") as f:
        annotations_data = json.load(f)

    # Collect data for each model
    data_by_model = {}
    # {model_name: {grade_int: count}}
    all_grades_by_model = defaultdict(lambda: defaultdict(int))

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
                        calculate_area_ratio_percentage(
                            annotations_data[img_id][obj_id]["bbox"]
                        )
                    )
                # Count all grades including not found
                all_grades_by_model[model_name][grading["description_grade"]] += 1

        data_by_model[model_name] = (gious, grades, area_ratios)

    # Create output directory
    output_dir = "stats"
    os.makedirs(output_dir, exist_ok=True)

    # Calculate area ratios once for all annotations
    area_ratios = [
        calculate_area_ratio_percentage(annotation["bbox"])
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
    plot_grade_distribution(
        all_grades_by_model,
        os.path.join(output_dir, "grade_distribution.png"),
    )


if __name__ == "__main__":
    main()
