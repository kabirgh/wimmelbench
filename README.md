First, set up the virtual environment:
```sh
❯ uv venv
❯ source .venv/bin/activate
# Install packages from pyproject.toml
❯ uv pip install .
```

To run web server to label objects and desribe bounding boxes:
```sh
❯ fastapi dev server.py
# Go to http://localhost:8000 to see the UI
```

Evaluate models:
```sh
❯ python src/wimmelbench/eval.py --help
usage: eval.py [-h] [--filter FILTER] [--skip-existing] [--models MODELS] annotations_file

positional arguments:
  annotations_file  Path to JSON annotations file (annotations.json)

options:
  -h, --help        show this help message and exit
  --filter FILTER   Only process images containing this string
  --skip-existing   Skip images that already have results
  --models MODELS   Comma-separated list of models to use (claude,gemini,gpt4o)
```

Grade responses:
```sh
❯ python src/wimmelbench/grade.py --help
usage: grade.py [-h] [--filter FILTER] [--skip-existing] [--all] annotations [results]

Grade object detection results against ground truth.

positional arguments:
  annotations      Path to ground truth annotations JSON file (annotations.json)
  results          Path to model results JSON file

options:
  -h, --help       show this help message and exit
  --filter FILTER  Optional string to filter image names
  --skip-existing  Skip grading objects that exist in the output grading.json
  --all            Process all hardcoded model result files
```

```sh
# Plot statistics
❯ python src/wimmelbench/stats.py
```
