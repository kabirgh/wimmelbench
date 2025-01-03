from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import os
import json
from pathlib import Path

app = FastAPI()

# Mount img files
app.mount("/img", StaticFiles(directory="img"), name="img")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Create annotations.json if it doesn't exist
if not Path("annotations.json").exists():
    with open("annotations.json", "w") as f:
        json.dump({}, f)


@app.get("/")
async def home(request: Request):
    # Get list of images from img directory
    img_dir = Path("img")
    images = sorted(
        [
            str(f.relative_to("img"))
            for f in img_dir.glob("*")
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]
    )

    # Load existing annotations
    with open("annotations.json", "r") as f:
        annotations = json.load(f)

    return templates.TemplateResponse(
        "index.html", {"request": request, "images": images, "annotations": annotations}
    )


@app.post("/save_annotation")
async def save_annotation(request: Request):
    data = await request.json()
    print(data)

    with open("annotations.json", "r") as f:
        annotations = json.load(f)

    # Update annotations for the image
    for image_name, details in data.items():
        annotations[image_name] = details

    # Sort annotations by image name
    sorted_annotations = dict(sorted(annotations.items()))

    with open("annotations.json", "w") as f:
        json.dump(sorted_annotations, f, indent=2)

    return JSONResponse({"status": "success"})


@app.get("/get_annotations")
def get_annotations():
    # Return the current annotations from your storage
    with open("annotations.json", "r") as f:
        annotations = json.load(f)
    return annotations
