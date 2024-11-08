import os
import json
import time
from PIL import Image
import google.generativeai as genai
from tqdm import tqdm  # For progress bar
import argparse

# Add argument parsing
parser = argparse.ArgumentParser(description="Process images and generate descriptions")
parser.add_argument(
    "--filter",
    type=str,
    help="Only process images with this text in filename",
    default="",
)
parser.add_argument(
    "--skip-existing",
    action="store_true",
    help="Skip processing if descriptions.json already exists",
)
args = parser.parse_args()

# Try to load existing descriptions
descriptions = {}
if args.skip_existing and os.path.exists("descriptions.json"):
    with open("descriptions.json", "r") as f:
        descriptions = json.load(f)

# Configure Gemini
genai.configure(
    api_key=os.environ.get("GOOGLE_AISTUDIO_API_KEY", "could-not-find-google-api-key")
)
model = genai.GenerativeModel("gemini-1.5-flash-002")

# Load annotations
with open("annotations.json", "r") as f:
    annotations = json.load(f)

items = annotations.items()
if args.filter:
    items = [i for i in items if args.filter in i[0]]

# Process each image
for image_file, objects in tqdm(items):
    # Skip if image already has descriptions in existing file AND none are empty
    if args.skip_existing and image_file in descriptions:
        has_empty_descriptions = any(
            not obj.get("description") for obj in descriptions[image_file]
        )
        if not has_empty_descriptions:
            annotations[image_file] = descriptions[image_file]
            print(
                f"\nSkipping {image_file} because it already has complete descriptions"
            )
            continue
        else:
            print(f"\nProcessing {image_file} to fill in missing descriptions")

    # Load the image once per file
    image_path = os.path.join("img", image_file)
    try:
        pil_image = Image.open(image_path)
    except FileNotFoundError:
        print(f"Warning: Image {image_file} not found, skipping")
        continue

    # Get description for each object without one
    for obj in objects:
        if not obj.get("description"):
            try:
                response = model.generate_content(
                    [
                        pil_image,
                        f"""Describe the {obj['object']}. Note specifically where it is located in the image. Also describe its colour, pose, activity, nearby prominent features, and any other relevant details.
                        Here are examples of good descriptions:
                        - (lighthouse) The lighthouse stands at the far right of the image, covering the middle third of the picture vertically. It is a tall white cylindrical structure with a viewing deck near its top. It's situated on a small outcropping in the lake, with people fishing or observing from an attached dock.
                        - (sedan) A red sedan car is located near the bottom right of the image. It is facing right, parked next to a blue house and behind a decorated christmas tree. A snow thrower is launching snow on to the car and startling the woman standing behind the car.
                        - (slide) A bright green slide extends into the lake in the mid-right portion of the image. People are sliding down it into the water, creating splashes where it meets the lake.

                        Provide only the description.
                        """,
                    ]
                )
                obj["description"] = response.text.strip()
                print(f"{obj['object']}\n{obj['description']}\n")
                # Small delay to avoid rate limiting
                time.sleep(0.5)
            except Exception as e:
                print(f"Error processing {image_file} - {obj['object']}: {str(e)}")
                obj["description"] = ""

# Save updated annotations
with open("descriptions.json", "w+") as f:
    json.dump(annotations, f, indent=2)
