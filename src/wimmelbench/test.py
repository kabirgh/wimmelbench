from PIL import Image, ImageDraw
import json
import argparse


def draw_bbox(image_path, annotations_path, image_name):
    # Load annotations
    with open(annotations_path, "r") as f:
        annotations = json.load(f)

    # Check if image exists in annotations
    if image_name not in annotations:
        print(f"No annotations found for {image_name}")
        return

    # Load image
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"Image not found: {image_path}")
        return

    # Get image dimensions
    width, height = img.size

    # Create drawing object
    draw = ImageDraw.Draw(img)

    # Draw each bbox
    for annotation in annotations[image_name]:
        # Convert relative coordinates to absolute pixels
        x1 = annotation["bbox"][0] * width
        y1 = annotation["bbox"][1] * height
        x2 = annotation["bbox"][2] * width
        y2 = annotation["bbox"][3] * height

        # Draw rectangle with red outline (width=2)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

    # Save or show the image
    img.show()
    img.save(f"annotated_{image_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Draw bounding boxes on images from annotations"
    )
    parser.add_argument("annotations", help="Path to annotations JSON file")
    parser.add_argument("image", help="Name of the image file")
    parser.add_argument("--img_dir", default="img", help="Directory containing images")

    args = parser.parse_args()

    image_path = f"{args.img_dir}/{args.image}"
    draw_bbox(image_path, args.annotations, args.image)


if __name__ == "__main__":
    main()
