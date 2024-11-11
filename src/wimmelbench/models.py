import base64
import json

from PIL import Image

from anthropic import Anthropic
from openai import OpenAI
import google.generativeai as genai

SYSTEM_PROMPT = """You are an expert computer vision system. You will be given an image and asked find a specific object within it. The object may not be present in the image.

When asked to describe the object, return a detailed description of the object you identified. Describe specifically where it is located in the image. Also describe its colour, pose, activity, spatial relationships with other objects, prominent features, and any other relevant details.

Provide your result as a JSON object with the following structure:
{json_prompt}
"""

DEFAULT_JSON_PROMPT = """
{
  "bbox": [x1, y1, x2, y2],
  "description": string
}

- bbox: The bounding box around the object you identified.
- All coordinates should be normalized to be between 0 and 1, where (0,0) is the top-left corner of the image and (1,1) is the bottom-right corner.
- If you do not see the object, provide bounding box coordinates of [0, 0, 0, 0]."""

# Gemini wants to return coordinates from 0 to 1000 in the format [ymin, xmin, ymax, xmax]
# https://ai.google.dev/gemini-api/docs/vision?lang=python#bbox
GEMINI_JSON_PROMPT = """
{
  "bbox": [ymin, xmin, ymax, xmax],
  "description": string
}

- bbox: The bounding box around the object you identified.
- If you do not see the object, provide bounding box coordinates of [0, 0, 0, 0]."""


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class AnthropicModel:
    def __init__(self, api_key, model):
        self.client = Anthropic(
            api_key=api_key,
        )
        self.model = model

    def detect_object(self, image_path: str, object_name: str, max_tokens=1024) -> dict:
        base64_image = encode_image(image_path)

        response = self.client.messages.create(
            model=self.model,
            system=SYSTEM_PROMPT.format(json_prompt=DEFAULT_JSON_PROMPT),
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_image,
                            },
                        },
                        {
                            "type": "text",
                            "text": f"Find the {object_name}",
                        },
                    ],
                },
            ],
        )

        return json.loads(response.content[0].text)  # type: ignore


class OpenAIModel:
    """
    Can also be used for OpenAI-compatible APIs like OpenRouter
    """

    def __init__(self, api_key, model, base_url=None):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.model = model

    def detect_object(self, image_path: str, object_name: str, max_tokens=1024) -> dict:
        base64_image = encode_image(image_path)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT.format(json_prompt=DEFAULT_JSON_PROMPT),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                        {
                            "type": "text",
                            "text": f"Find the {object_name}",
                        },
                    ],
                },
            ],
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )

        # OpenRouter error handling
        if (
            hasattr(response, "error")
            and response.error  # type: ignore
            and response.error.get("code") == 429  # type: ignore
        ):
            raise Exception("Rate limit exceeded")

        data = response.choices[0].message.content
        return json.loads(data)  # type: ignore


class GoogleModel:
    def __init__(self, api_key, model):
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(
            model,
            system_instruction=SYSTEM_PROMPT.format(json_prompt=GEMINI_JSON_PROMPT),
        )
        self.model = model

    def detect_object(self, image_path: str, object_name: str):
        pil_image = Image.open(image_path)

        response = self.client.generate_content([pil_image, f"Find the {object_name}"])

        # There's usually a code fence
        data = json.loads(response.text.replace("```json\n", "").replace("\n```", ""))
        bbox = data["bbox"]

        bbox = [
            bbox[1] / 1000,  # xmin
            bbox[0] / 1000,  # ymin
            bbox[3] / 1000,  # xmax
            bbox[2] / 1000,  # ymax
        ]

        return {
            "description": data["description"],
            "bbox": bbox,
        }
