import argparse
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
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("object_name", help="Object to detect")
    args = parser.parse_args()

    models = [
        OpenAIModel(
            api_key=os.environ.get("OPENAI_API_KEY", "could-not-find-openai-api-key"),
            model="gpt-4o-2024-08-06",
        ),
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
    ]

    for model in models:
        save_path = get_save_path(args.image_path, args.object_name, model.model)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        result = model.detect_object(args.image_path, args.object_name)
        print(f"\nResult for {model.model}: {result}")

        if result["confidence"] > 0:
            image = draw_box(args.image_path, result["bbox"])
            image.save(save_path, "JPEG")
            print(f"Saved image with bounding box to {save_path}")
        else:
            print(f"No {args.object_name} detected")


if __name__ == "__main__":
    main()
