import argparse
import json
from PIL import Image
import os

# Import draw_box function from eval.py
from wimmelbench.eval import draw_box, COLORS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", help="Path to results JSON file")
    parser.add_argument(
        "--save-dir", help="Directory to save annotated images", default="test"
    )
    parser.add_argument(
        "--filter", help="Only process images containing this string", default=None
    )
    args = parser.parse_args()

    # Load annotations and results
    with open("annotations2.json") as f:
        annotations = json.load(f)
    with open(args.results_file) as f:
        results = json.load(f)

    # Process each image in results
    for image_name in results:
        if args.filter and args.filter not in image_name:
            continue

        # Load image
        image_path = os.path.join("img", image_name)
        image = Image.open(image_path)

        # Get annotations and results for this image
        image_annotations = annotations[image_name]
        image_results = results[image_name]

        # Match results with annotations by object name
        for (object_name, result), color in zip(image_results.items(), COLORS):
            # Find matching annotation
            annotation = image_annotations.get(object_name, None)
            if not annotation:
                print(
                    f"Warning: No ground truth annotation found for {object_name} in {image_name}"
                )
                continue

            # Draw actual box
            image = draw_box(
                image, annotation["bbox"], f"{object_name} - actual", color
            )

            # Draw predicted box if it exists
            if result["bbox"] != [0, 0, 0, 0]:
                image = draw_box(image, result["bbox"], "", color)
            else:
                print(f"No {object_name} detected in {image_name}")

        # Save annotated image
        os.makedirs(args.save_dir, exist_ok=True)

        # Append "_annotated" to image name
        base_name, ext = os.path.splitext(image_name)
        annotated_name = f"{base_name}_annotated{ext}"
        save_path = os.path.join(args.save_dir, annotated_name)

        image.save(save_path, "JPEG")
        print(f"Saved image to {save_path}")


if __name__ == "__main__":
    main()
