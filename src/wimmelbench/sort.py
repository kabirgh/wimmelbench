import json
import sys
from pathlib import Path


def sort_json_file(file_path: str | Path) -> None:
    """Load a JSON file, sort it, and overwrite the original file.

    Args:
        file_path: Path to the JSON file to sort
    """
    # Load JSON file
    with open(file_path, "r") as f:
        data = json.load(f)

    # Sort the data (for nested dicts, sort at each level)
    sorted_data = {
        k: {
            inner_k: dict(sorted(inner_v.items()))
            for inner_k, inner_v in sorted(v.items())
        }
        for k, v in sorted(data.items())
    }

    # Write back to file with consistent formatting
    with open(file_path, "w") as f:
        json.dump(sorted_data, f, indent=2)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python sort.py <path_to_json_file>")
        sys.exit(1)

    sort_json_file(sys.argv[1])
