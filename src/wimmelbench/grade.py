import argparse
import json
import os
import time
from typing import Dict, List

import google.generativeai as genai
from tqdm import tqdm

MODEL_RESULTS = {
    "claude": "results/claude-3-5-sonnet-20241022/results.json",
    "gemini": "results/gemini-1.5-pro-002/results.json",
    "gpt4o": "results/gpt-4o-2024-08-06/results.json",
}

GRADING_PROMPT = """
You are an expert at comparing image descriptions. I will provide you with two descriptions of the same object in an image - a ground truth description and a predicted description. Please rate how well the predicted description matches the ground truth on a scale of:

0: Completely incorrect or missing critical details
1: Partially correct but missing many important details or containing significant inaccuracies
2: Majorly correct with some inaccuracies or missing details
3: Mostly or fully correct, capturing the majority of key details and spatial relationships accurately

Please provide a rating (0-3) and a brief explanation of your reasoning.

Important criteria to consider (in order of importance):
- The object's key identifying details (NOT the presence of the object, which is known to the prediction model)
- The object's spatial location in the image
- Color and appearance details of the object
- Basic spatial relationships with adjacent elements (less important)

Further notes:
- Do not award points simply for identifying the object, as the prediction model is told what object to look for
    - If the predicted description places the object in the wrong location, give a rating of 0
- Focus primarily on how well the object itself and its location are described, rather than detailed descriptions of surrounding elements or complex relationships
- Be lenient with descriptions of the pose (eg. "standing" and "walking" should be considered correct)
- The ground truth description may not mention all the information that is present in the image. Ignore additional details in the predicted description unless they contradict the ground truth
- Ignore comments on style and other non-object details


Return your rating and explanation in the following JSON format:
{{"rating": <rating>, "explanation": <explanation>}}

Object: {object_name}
Ground truth: {ground_truth_description}
Predicted: {predicted_description}
""".strip()

genai.configure(
    api_key=os.environ.get("GOOGLE_AISTUDIO_API_KEY", "could-not-find-google-api-key")
)
model = genai.GenerativeModel("gemini-1.5-pro-002")


def load_json(path: str) -> Dict:
    """Load and parse a JSON file."""
    with open(path) as f:
        data = json.load(f)
        # Sort the outer dictionary by keys
        return dict(sorted(data.items()))


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


def rate_description(
    object_name: str, ground_truth_description: str, predicted_description: str
) -> Dict:
    """Rate the accuracy of a predicted description against a ground truth description."""
    prompt = GRADING_PROMPT.format(
        object_name=object_name,
        ground_truth_description=ground_truth_description,
        predicted_description=predicted_description,
    )
    response = model.generate_content([prompt])
    return json.loads(response.text.replace("```json\n", "").replace("\n```", ""))


def calculate_metrics(results: Dict) -> Dict:
    """Calculate summary metrics from results dictionary."""
    total_objects = sum(len(img_objs) for img_objs in results.values())
    not_found = sum(
        1
        for img_objs in results.values()
        for obj in img_objs.values()
        if obj["status"] == "not found"
    )

    # Calculate averages only for valid entries
    valid_grades = [
        obj["description_grade"]
        for img_objs in results.values()
        for obj in img_objs.values()
        if obj["description_grade"] >= 0
    ]

    return {
        "total_images": len(results),
        "total_objects": total_objects,
        "total_not_found": not_found,
        "average_giou": sum(
            obj["giou"] for img_objs in results.values() for obj in img_objs.values()
        )
        / total_objects,
        "average_description_grade": sum(valid_grades) / len(valid_grades),
    }


def grade(
    annotations_path: str,
    results_path: str,
    name_filter: str | None = None,
    skip_existing: bool = False,
) -> Dict:
    """Grade the results against ground truth annotations."""
    ground_truth = load_json(annotations_path)
    results = load_json(results_path)

    output_dir = os.path.dirname(results_path)
    grading_path = os.path.join(output_dir, "grading.json")
    detailed_results = (
        load_json(grading_path)
        if (skip_existing and os.path.exists(grading_path))
        else {}
    )

    for image_name in tqdm(results, desc="Grading results"):
        if name_filter and name_filter not in image_name:
            continue
        if image_name not in ground_truth:
            print(f"Skipping {image_name} because it's not in ground truth")
            continue

        detailed_results.setdefault(image_name, {})

        for object_name, predicted in results[image_name].items():
            if skip_existing and object_name in detailed_results[image_name]:
                continue

            try:
                actual = ground_truth[image_name][object_name]
                if predicted["bbox"] == [0, 0, 0, 0]:
                    result = {
                        "status": "not found",
                        "giou": -1.0,
                        "description_grade": -1,
                        "description_grade_reason": "",
                    }
                else:
                    rating = rate_description(
                        object_name, actual["description"], predicted["description"]
                    )
                    time.sleep(1)  # Rate limiting
                    result = {
                        "status": "predicted",
                        "giou": calculate_giou(actual["bbox"], predicted["bbox"]),
                        "description_grade": rating["rating"],
                        "description_grade_reason": rating["explanation"],
                    }
                detailed_results[image_name][object_name] = result
            except StopIteration:
                print(f"No matching ground truth box found for {object_name}")
                continue

        # Save intermediate results
        with open(grading_path, "w") as f:
            json.dump(dict(sorted(detailed_results.items())), f, indent=2)

    return detailed_results


def grade_all(annotations_path: str, skip_existing: bool = False) -> None:
    """Grade all model results against ground truth annotations.

    Args:
        annotations_path: Path to ground truth annotations JSON
        skip_existing: If True, skip grading objects that exist in output grading.json
    """
    for model_name, results_path in MODEL_RESULTS.items():
        if not os.path.exists(results_path):
            print(f"Skipping {model_name} - results file not found")
            continue

        print(f"\nProcessing {model_name} results...")
        details = grade(annotations_path, results_path, skip_existing=skip_existing)

        # Print summary for this model
        summary = calculate_metrics(details)
        print(f"{model_name} summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grade object detection results against ground truth."
    )
    parser.add_argument(
        "annotations", help="Path to ground truth annotations JSON file"
    )
    parser.add_argument("results", help="Path to model results JSON file", nargs="?")
    parser.add_argument(
        "--filter", help="Optional string to filter image names", default=None
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip grading objects that exist in the output grading.json",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all hardcoded model result files",
    )

    args = parser.parse_args()

    if args.all:
        if args.results or args.filter:
            parser.error("--all cannot be used with results path or --filter")
        grade_all(args.annotations, args.skip_existing)
    else:
        if not args.results:
            parser.error("results path is required when not using --all")
        details = grade(args.annotations, args.results, args.filter, args.skip_existing)

        # Create summary of results
        summary = calculate_metrics(details)
        print(f"Summary: {json.dumps(summary, indent=2)}")
