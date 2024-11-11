import argparse
import json
from typing import Dict, List


def load_json(path: str) -> Dict:
    """Load and parse a JSON file."""
    with open(path) as f:
        return json.load(f)


def calculate_giou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate the Generalized Intersection over Union (GIoU) between two bounding boxes.
    Based on https://giou.stanford.edu/GIoU.pdf

    Args:
        box1: List of [x1, y1, x2, y2] coordinates of first box (top-left and bottom-right corners)
        box2: List of [x1, y1, x2, y2] coordinates of second box (top-left and bottom-right corners)

    Returns:
        float: GIoU value in range [-1, 1]
    """
    # Extract coordinates
    box1_x1, box1_y1, box1_x2, box1_y2 = box1
    box2_x1, box2_y1, box2_x2, box2_y2 = box2

    # Calculate area of each box
    area1 = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    area2 = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

    # Calculate coordinates of intersection
    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)

    # Calculate intersection area
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    intersection_area = inter_width * inter_height

    # Calculate union area
    union_area = area1 + area2 - intersection_area

    # Calculate IoU (Intersection over Union)
    iou = intersection_area / union_area if union_area > 0 else 0

    # Calculate coordinates of smallest enclosing box (convex hull)
    hull_x1 = min(box1_x1, box2_x1)
    hull_y1 = min(box1_y1, box2_y1)
    hull_x2 = max(box1_x2, box2_x2)
    hull_y2 = max(box1_y2, box2_y2)

    # Calculate area of convex hull
    hull_area = (hull_x2 - hull_x1) * (hull_y2 - hull_y1)

    # Calculate GIoU
    # GIoU = IoU - (Area of convex hull - Area of union) / Area of convex hull
    giou = iou - ((hull_area - union_area) / hull_area if hull_area > 0 else 0)

    return giou


def grade(
    annotations_path: str, results_path: str, name_filter: str | None = None
) -> Dict:
    """Grade the results against ground truth annotations.

    Args:
        annotations_path: Path to ground truth annotations JSON
        results_path: Path to model results JSON
        name_filter: Optional string to filter image names (only process images containing this string)

    Returns:
        Tuple of (average IoU score, detailed results dict)
    """
    # Load files
    ground_truth = load_json(annotations_path)
    results = load_json(results_path)

    detailed_results = {}

    # Compare each image
    for image_name in results:
        if name_filter and name_filter not in image_name:
            print(f"Skipping {image_name} because it doesn't match filter")
            continue

        if image_name not in ground_truth:
            print(f"Skipping {image_name} because it's not in ground truth")
            continue

        detailed_results[image_name] = []

        # Compare each predicted box to ground truth boxes
        for pred_box in results[image_name]:
            # Find the ground truth box that matches the predicted box. Assume there is only one where "object" matches
            # TODO: should this be keyed on "object" instead of an array index?
            try:
                gt_box = next(
                    box
                    for box in ground_truth[image_name]
                    if box["object"] == pred_box["object"]
                )

                if pred_box["bbox"] == [0, 0, 0, 0]:
                    detailed_results[image_name].append(
                        {
                            "object": gt_box["object"],
                            "giou": -1.0,  # GIoU can go to -1 in worst case
                            "status": "not predicted",
                        }
                    )
                else:
                    detailed_results[image_name].append(
                        {
                            "object": pred_box["object"],
                            "giou": calculate_giou(gt_box["bbox"], pred_box["bbox"]),
                            "status": "predicted",
                        }
                    )
            except StopIteration:
                print(f"No matching ground truth box found for {pred_box['object']}")
                continue

    return detailed_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grade object detection results against ground truth."
    )
    parser.add_argument(
        "annotations", help="Path to ground truth annotations JSON file"
    )
    parser.add_argument("results", help="Path to model results JSON file")
    parser.add_argument(
        "--filter", help="Optional string to filter image names", default=None
    )

    args = parser.parse_args()

    details = grade(args.annotations, args.results, args.filter)
    print(f"Results: {json.dumps(details, indent=2)}")
