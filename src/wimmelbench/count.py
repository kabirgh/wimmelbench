import json


def main():
    # Read the annotations file
    with open("annotations.json", "r") as f:
        annotations = json.load(f)

    # Count images and objects
    num_images = len(annotations)
    num_objects = sum(len(objects) for objects in annotations.values())

    # Print results
    print(f"Number of images: {num_images}")
    print(f"Number of objects: {num_objects}")


if __name__ == "__main__":
    main()
