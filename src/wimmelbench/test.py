import json
import argparse


def find_objects(grading_path):
    # Read the JSON file
    with open(grading_path, "r") as f:
        data = json.load(f)

    # List to store results
    results = []

    # Iterate through all images and objects
    for image_name, objects in data.items():
        for obj_name, details in objects.items():
            # Skip if object wasn't predicted
            if details.get("status") != "predicted":
                continue

            # Check if meets criteria
            grade = details.get("description_grade", -1)
            giou = details.get("giou", 1)

            if grade >= 2 and giou <= -0.5:
                results.append(
                    {
                        "image": image_name,
                        "object": obj_name,
                        "grade": grade,
                        "giou": giou,
                    }
                )

    return results


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Find objects with grade >= 2 and giou <= 0"
    )
    parser.add_argument("grading_path", help="Path to grading.json file")

    # Parse arguments
    args = parser.parse_args()

    # Find matching objects
    results = find_objects(args.grading_path)

    # Print results
    if results:
        print("\nFound objects meeting criteria (grade >= 2 and giou <= 0):")
        print("\nimage | object | grade | giou")
        print("-" * 50)
        for r in results:
            print(f"{r['image']} | {r['object']} | {r['grade']} | {r['giou']:.3f}")
    else:
        print("No objects found meeting the criteria.")


if __name__ == "__main__":
    main()
