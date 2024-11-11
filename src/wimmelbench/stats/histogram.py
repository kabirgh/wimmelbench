import argparse
import os
from matplotlib import pyplot as plt
import json
from statistics import mean
from typing import List


def plot_giou_histogram(data: dict, output_filename: str) -> List[float]:
    # Extract all GIoU scores using list comprehension
    giou_scores = [
        object_data["giou"]
        for image in data.values()
        for object_data in image.values()
        if "giou" in object_data
    ]

    plt.figure(figsize=(10, 6))
    plt.hist(giou_scores, bins=20, color="skyblue", edgecolor="black")

    # Set specific axis limits
    plt.xlim(-1, 1)
    plt.ylim(0, 25)

    plt.xlabel("GIoU Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of GIoU Scores")
    plt.grid(True, alpha=0.3)

    # Calculate mean once and reuse
    mean_giou = mean(giou_scores)
    plt.axvline(
        x=mean_giou, color="red", linestyle="--", label=f"Mean: {mean_giou:.3f}"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_filename)
    plt.close()

    # Print statistics
    negative_count = sum(score < 0 for score in giou_scores)

    print(f"Plot saved to: {output_filename}")
    print(f"Total samples: {len(giou_scores)}")
    print(f"Mean GIoU: {mean_giou:.3f}")
    print(f"Number of negative GIoU scores: {negative_count}")

    return giou_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot GIoU histogram from results directory"
    )
    parser.add_argument(
        "results_dir", type=str, help="Path to directory containing grading.json"
    )
    parser.add_argument(
        "--output", "-o", type=str, help="Output filename for the plot", default=None
    )
    args = parser.parse_args()

    grading_json_path = os.path.join(args.results_dir, "grading.json")
    results_json_path = os.path.join(args.results_dir, "results.json")

    with open(grading_json_path, "r") as f:
        grading_data = json.load(f)
    with open(results_json_path, "r") as f:
        results_data = json.load(f)

    plot_giou_histogram(grading_data, args.output)
