import argparse
import base64
import os
from typing import List

from dotenv import load_dotenv
from PIL import Image, ImageDraw

from wimmelbench.models import AnthropicModel, OpenAIModel

load_dotenv()


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def draw_box(image_path: str, bbox: List[float]):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    w, h = image.size
    x1, y1, x2, y2 = bbox

    # Convert normalized coordinates to pixel coordinates
    box = (x1 * w, y1 * h, x2 * w, y2 * h)

    # Draw rectangle
    draw.rectangle(box, outline="red", width=6)
    return image


def get_save_path(image_path: str, object_name: str, model: str) -> str:
    filename = os.path.basename(image_path)
    new_filename = filename.replace(".", f"_{model}_{object_name}.")
    return os.path.join("results", new_filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("object_name", help="Object to detect")
    args = parser.parse_args()

    img = encode_image(args.image_path)
    models = [OpenAIModel(), AnthropicModel()]

    for model in models:
        result = model.detect_object(img, args.object_name)
        print(f"\nResult for {model.model}: {result}")

        if result["confidence"] > 0:
            image = draw_box(args.image_path, result["bbox"])
            save_path = get_save_path(args.image_path, args.object_name, model.model)
            image.save(save_path, "JPEG")
            print(f"Saved image with bounding box to {save_path}")
        else:
            print(f"No {args.object_name} detected")


if __name__ == "__main__":
    main()
