import argparse
import json
import os
from typing import List

from dotenv import load_dotenv
from PIL import Image, ImageDraw

from wimmelbench.models import AnthropicModel, OpenAIModel, GoogleModel

load_dotenv()


def draw_box(image_path: str, bbox: List[float]):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    w, h = image.size
    x1, y1, x2, y2 = bbox

    # Convert normalized coordinates to pixel coordinates
    box = (x1 * w, y1 * h, x2 * w, y2 * h)

    draw.rectangle(box, outline="red", width=6)
    return image


def get_save_path(image_path: str, object_name: str, model: str) -> str:
    filename = os.path.basename(image_path)
    new_filename = filename.replace(".", f"_{object_name}.")
    return os.path.join("results", model.replace("/", "_"), new_filename)


def get_results_path(model: str) -> str:
    return os.path.join("results", model.replace("/", "_"), "results.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations_file", help="Path to JSON annotations file")
    parser.add_argument(
        "--filter", help="Only process images containing this string", default=None
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images that already have results",
    )
    args = parser.parse_args()

    # Load annotations
    with open(args.annotations_file) as f:
        annotations = json.load(f)

    models = [
        AnthropicModel(
            api_key=os.environ.get(
                "ANTHROPIC_API_KEY", "could-not-find-anthropic-api-key"
            ),
            model="claude-3-5-sonnet-20241022",
        ),
        GoogleModel(
            api_key=os.environ.get(
                "GOOGLE_AISTUDIO_API_KEY", "could-not-find-google-api-key"
            ),
            model="gemini-1.5-pro",
        ),
        OpenAIModel(
            api_key=os.environ.get("OPENAI_API_KEY", "could-not-find-openai-api-key"),
            model="gpt-4o-2024-08-06",
        ),
    ]

    # Process each model
    for model in models:
        # Load or create results file for this model
        results_file = get_results_path(model.model)
        os.makedirs(os.path.dirname(results_file), exist_ok=True)

        if os.path.exists(results_file):
            with open(results_file) as f:
                results = json.load(f)
        else:
            results = {}

        # Process each image and object in annotations
        for image_name, objects in annotations.items():
            if args.filter and args.filter not in image_name:
                continue

            # Skip if image already has results and skip-existing is enabled
            if args.skip_existing and image_name in results:
                print(f"Skipping {image_name} - results already exist")
                continue

            image_path = os.path.join("img", image_name)

            # Initialize results for this image if not exists
            if image_name not in results:
                results[image_name] = []

            for object_data in objects:
                object_name = object_data["object"]

                result = model.detect_object(image_path, object_name)
                print(f"\nResult for {model.model} on {image_path}: {result}")

                # Create new result entry
                result_entry = {
                    "bbox": result["bbox"],
                    "object": object_name,
                    "description": result["description"],
                }

                # Find and replace existing entry with same object name, or append if not found
                existing_index = next(
                    (
                        i
                        for i, r in enumerate(results[image_name])
                        if r["object"] == object_name
                    ),
                    None,
                )
                if existing_index is not None:
                    results[image_name][existing_index] = result_entry
                else:
                    results[image_name].append(result_entry)

                if result["bbox"] != [0, 0, 0, 0]:
                    image = draw_box(image_path, result["bbox"])
                    save_path = get_save_path(image_path, object_name, model.model)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    image.save(save_path, "JPEG")
                    print(f"Saved image with bounding box to {save_path}")
                else:
                    print(f"No {object_name} detected in {image_path}")

            # Save results after processing each image
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
