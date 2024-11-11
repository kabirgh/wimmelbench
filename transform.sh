#!/bin/bash
set -euxo pipefail

# Use find to locate all JSON files in current directory and subdirectories
find . -type f -name "*.json" -not -path "./.venv/*" | while read file; do
    # Skip files ending with 2.json or _old.json
    [[ $file == *2.json ]] && continue
    [[ $file == *_old.json ]] && continue

    # Get the base name without .json extension and directory path
    dir_path=$(dirname "$file")
    base_name=$(basename "$file" .json)

    # Rename original file to _old.json
    mv "$file" "${dir_path}/${base_name}_old.json"

    # Run the transform script
    python src/wimmelbench/transform.py "${dir_path}/${base_name}_old.json" "${dir_path}/${base_name}.json"
done
