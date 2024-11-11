import json
import matplotlib.pyplot as plt
import argparse
import math
import os
from statistics import mean, stdev


def calculate_bbox_area(bbox):
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return width * height * 100


def process_grading_json(grading_json, results_json):
    gious = []
    ratios = []
    desc_grades = []

    for image_name, image_data in grading_json.items():
        image_results = results_json.get(image_name, {})
        for obj_id, obj_data in image_data.items():
            if obj_data["status"] == "predicted" and obj_data["giou"] > -1:
                result_obj = image_results.get(obj_id)
                if result_obj and "bbox" in result_obj:
                    gious.append(obj_data["giou"])
                    desc_grades.append(obj_data["description_grade"])
                    bbox = result_obj["bbox"]
                    area_ratio = calculate_bbox_area(bbox)
                    ratios.append(area_ratio)

    return gious, ratios, desc_grades


def analyze_size_buckets(gious, ratios, num_buckets=5):
    # Convert to log space
    log_ratios = [math.log10(r) for r in ratios]

    # Create bucket boundaries
    min_log = min(log_ratios)
    max_log = max(log_ratios)
    bucket_size = (max_log - min_log) / num_buckets
    bucket_bounds = [min_log + i * bucket_size for i in range(num_buckets + 1)]

    # Initialize bucket statistics
    buckets = []
    for i in range(num_buckets):
        buckets.append(
            {
                "range": (bucket_bounds[i], bucket_bounds[i + 1]),
                "gious": [],
                "ratios": [],
                "count": 0,
            }
        )

    # Sort data into buckets
    for giou, ratio, log_ratio in zip(gious, ratios, log_ratios):
        for bucket in buckets:
            if bucket["range"][0] <= log_ratio < bucket["range"][1]:
                bucket["gious"].append(giou)
                bucket["ratios"].append(ratio)
                bucket["count"] += 1
                break

    # Calculate statistics for each bucket
    for bucket in buckets:
        if bucket["count"] > 0:
            bucket["mean_giou"] = mean(bucket["gious"])
            bucket["std_giou"] = (
                stdev(bucket["gious"]) if len(bucket["gious"]) > 1 else 0
            )
            bucket["median_ratio"] = sorted(bucket["ratios"])[
                len(bucket["ratios"]) // 2
            ]

    return buckets


def plot_bucket_analysis(buckets, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Box count per bucket
    bucket_centers = [
        f"{10**b['range'][0]:.1e}\n-\n{10**b['range'][1]:.1e}" for b in buckets
    ]
    counts = [b["count"] for b in buckets]

    ax1.bar(range(len(buckets)), counts)
    ax1.set_xticks(range(len(buckets)))
    ax1.set_xticklabels(bucket_centers, rotation=45)
    ax1.set_title("Sample Count by Area Ratio Range")
    ax1.set_xlabel("Area Ratio Range")
    ax1.set_ylabel("Number of Samples")

    # Plot 2: Mean GIoU with error bars
    means = [b["mean_giou"] for b in buckets]
    stds = [b["std_giou"] for b in buckets]

    ax2.errorbar(range(len(buckets)), means, yerr=stds, fmt="o-")
    ax2.set_xticks(range(len(buckets)))
    ax2.set_xticklabels(bucket_centers, rotation=45)
    ax2.set_title("Mean GIoU by Area Ratio Range")
    ax2.set_xlabel("Area Ratio Range")
    ax2.set_ylabel("Mean GIoU")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return bucket_centers, means, stds


def main():
    parser = argparse.ArgumentParser(
        description="Analyze performance across size buckets."
    )
    parser.add_argument(
        "results_dir", help="Directory containing grading.json and results.json"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="bucket_analysis.png",
        help="Output path for the plots (default: bucket_analysis.png)",
    )
    parser.add_argument(
        "--num-buckets",
        "-n",
        type=int,
        default=5,
        help="Number of size buckets (default: 5)",
    )
    args = parser.parse_args()

    grading_json_path = os.path.join(args.results_dir, "grading.json")
    results_json_path = os.path.join(args.results_dir, "results.json")

    with open(grading_json_path, "r") as f:
        grading_data = json.load(f)
    with open(results_json_path, "r") as f:
        results_data = json.load(f)

    gious, ratios, desc_grades = process_grading_json(grading_data, results_data)
    buckets = analyze_size_buckets(gious, ratios, args.num_buckets)
    bucket_centers, means, stds = plot_bucket_analysis(buckets, args.output)

    print("\nBucket Analysis:")
    print("-" * 50)
    for i, (center, mean, std, bucket) in enumerate(
        zip(bucket_centers, means, stds, buckets)
    ):
        print(f"\nBucket {i+1} ({center}):")
        print(f"  Samples: {bucket['count']}")
        print(f"  Mean GIoU: {mean:.3f} ± {std:.3f}")


if __name__ == "__main__":
    main()
