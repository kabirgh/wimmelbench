import json
import matplotlib.pyplot as plt
import argparse
import math
import os


def calculate_bbox_area(bbox):
    # bbox format is [x1, y1, x2, y2]
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    # %
    return width * height * 100


def process_grading_json(grading_json, results_json):
    gious = []
    ratios = []
    desc_grades = []

    # Process each image
    for image_name, image_data in grading_json.items():
        # Get corresponding results data
        image_results = results_json.get(image_name, {})

        # Process each object in the image
        for obj_id, obj_data in image_data.items():
            # Only include predicted objects with valid GIoU scores
            if obj_data["status"] == "predicted" and obj_data["giou"] > -1:
                # Get bbox from results
                result_obj = image_results.get(obj_id)
                if result_obj and "bbox" in result_obj:
                    gious.append(obj_data["giou"])
                    desc_grades.append(obj_data["description_grade"])
                    bbox = result_obj["bbox"]
                    area_ratio = calculate_bbox_area(bbox)
                    ratios.append(area_ratio)

    return gious, ratios, desc_grades


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


def main():
    parser = argparse.ArgumentParser(
        description="Process grading JSON and create GIoU vs Area Ratio plot."
    )
    parser.add_argument(
        "results_dir", help="Directory containing grading.json and results.json"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="giou_vs_ratio.png",
        help="Output path for the plot (default: giou_vs_ratio.png)",
    )
    args = parser.parse_args()

    # Construct paths to JSON files
    grading_json_path = os.path.join(args.results_dir, "grading.json")
    results_json_path = os.path.join(args.results_dir, "results.json")

    # Read both JSON files
    with open(grading_json_path, "r") as f:
        grading_data = json.load(f)
    with open(results_json_path, "r") as f:
        results_data = json.load(f)

    # Process the data
    gious, ratios, desc_grades = process_grading_json(grading_data, results_data)

    # Calculate R² for GIoU vs log area ratio
    log_ratios = [math.log10(r) for r in ratios]
    r_squared = calculate_r_squared(log_ratios, gious)
    print(f"R² value for GIoU vs log(Area Ratio): {r_squared:.3f}")

    # Create a figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # First subplot: GIoU vs Area Ratio
    ax1.scatter(ratios, gious, alpha=0.5)
    ax1.set_xlabel("Area Ratio")
    ax1.set_ylabel("GIoU")
    ax1.set_title("GIoU vs Area Ratio")
    ax1.set_xscale("log")
    ax1.grid(True)

    # Second subplot: Description Grade vs Area Ratio
    ax2.scatter(ratios, desc_grades, alpha=0.5)
    ax2.set_xlabel("Area Ratio")
    ax2.set_ylabel("Description Grade")
    ax2.set_title("Description Grade vs Area Ratio")
    ax2.grid(True)

    # New third subplot: Description Grade vs GIoU
    ax3.scatter(gious, desc_grades, alpha=0.5)
    ax3.set_xlabel("GIoU")
    ax3.set_ylabel("Description Grade")
    ax3.set_title("Description Grade vs GIoU")
    ax3.grid(True)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(args.output)
    plt.close()


if __name__ == "__main__":
    main()
