import json
from pathlib import Path
import argparse


def verify_telescope(results_file: Path) -> None:
    with open(results_file) as f:
        data = json.load(f)

    for image_name, image_data in data.items():
        bbox = image_data["telescope"]["bbox"]
        if bbox != [0, 0, 0, 0]:
            print(image_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify telescope bounding boxes in JSON file"
    )
    parser.add_argument("results_file", type=Path, help="Path to the JSON results file")
    args = parser.parse_args()

    verify_telescope(args.results_file)
