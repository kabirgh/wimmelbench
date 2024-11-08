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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations_file", help="Path to JSON annotations file")
    parser.add_argument(
        "--filter", help="Only process images containing this string", default=None
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

    # Process each image and object in annotations
    for image_name, objects in annotations.items():
        if args.filter and args.filter not in image_name:
            continue

        image_path = os.path.join("img", image_name)

        for object_data in objects:
            object_name = object_data["object"]

            for model in models:
                save_path = get_save_path(image_path, object_name, model.model)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                result = model.detect_object(image_path, object_name)
                print(f"\nResult for {model.model} on {image_path}: {result}")

                if result["confidence"] > 0:
                    image = draw_box(image_path, result["bbox"])
                    image.save(save_path, "JPEG")
                    print(f"Saved image with bounding box to {save_path}")
                else:
                    print(f"No {object_name} detected in {image_path}")


if __name__ == "__main__":
    main()
