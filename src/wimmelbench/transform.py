import json
import argparse


def transform_json(input_file, output_file):
    # Read the input JSON file
    with open(input_file, "r") as f:
        data = json.load(f)

    # Create new dictionary with transformed structure
    transformed = {}

    # Iterate through each image
    for image_name, objects_list in data.items():
        # Create dictionary for this image
        transformed[image_name] = {}

        # Add each object to the image dictionary
        for obj in objects_list:
            object_name = obj["object"]
            # Copy all other properties except 'object'
            obj_data = {k: v for k, v in obj.items() if k != "object"}
            transformed[image_name][object_name] = obj_data

    # Write the transformed data to output file
    with open(output_file, "w") as f:
        json.dump(transformed, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Transform JSON structure from list to dictionary format"
    )
    parser.add_argument("input", help="Input JSON file path")
    parser.add_argument("output", help="Output JSON file path")

    args = parser.parse_args()

    transform_json(args.input, args.output)


if __name__ == "__main__":
    main()
