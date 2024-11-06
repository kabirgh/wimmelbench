import base64
import json

from PIL import Image

from anthropic import Anthropic
from openai import OpenAI
import google.generativeai as genai


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class AnthropicModel:
    SYSTEM_PROMPT = """You are an expert computer vision system. You will be given an image and asked to find a specific object within it. The object may not be present in the image.

Provide your result as a JSON object with the following structure:
{
  "bbox": [x1, y1, x2, y2],
  "description": string,
  "confidence": number
}
Where:
- bbox: The bounding box around the object you identified. All coordinates should be normalized to be between 0 and 1, where (0,0) is the top-left corner of the image and (1,1) is the bottom-right corner.
- description: A detailed description of the object you identified. Describe specifically where it is located in the image. For example: "it is in the top third of the image, slightly left of center. It is above a red beach ball." Also describe its colour, pose, activity, and any other relevant details.
- confidence: Your confidence score for the identification, between 0 and 1. For example, if you are 50% sure the object is where you have described, provide a confidence of 0.5.

If you do not see the object, provide a confidence score of 0 and the bounding box coordinates as [0, 0, 0, 0].

Provide your final output ONLY in the JSON format described above.
"""

    def __init__(self, api_key, model):
        self.client = Anthropic(
            api_key=api_key,
        )
        self.model = model

    def detect_object(self, image_path: str, object_name: str, max_tokens=1024) -> dict:
        base64_image = encode_image(image_path)

        response = self.client.messages.create(
            model=self.model,
            system=self.SYSTEM_PROMPT,
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
                {
                    "role": "assistant",
                    "content": "{",
                },
            ],
        )

        data = json.loads("{" + response.content[0].text)  # type: ignore
        return data


class OpenAIModel:
    """
    Can also be used for OpenAI-compatible APIs like OpenRouter
    """

    SYSTEM_PROMPT = """You are an expert computer vision system. You will be given an image and asked to find a specific object within it. The object may not be present in the image.

Provide your result as a JSON object with the following structure:
{
  "bbox": [x1, y1, x2, y2],
  "description": string,
  "confidence": number
}
Where:
- bbox: The bounding box around the object you identified. All coordinates should be normalized to be between 0 and 1, where (0,0) is the top-left corner of the image and (1,1) is the bottom-right corner.
- description: A detailed description of the object you identified. Describe specifically where it is located in the image. For example: "it is in the top third of the image, slightly left of center. It is above a red beach ball." Also describe its colour, pose, activity, and any other relevant details.
- confidence: Your confidence score for the identification, between 0 and 1. For example, if you are 50% sure the object is where you have described, provide a confidence of 0.5.

If you do not see the object, provide a confidence score of 0 and the bounding box coordinates as [0, 0, 0, 0].

Provide your final output ONLY in the JSON format described above.
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
                {"role": "system", "content": self.SYSTEM_PROMPT},
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
    # Gemini wants to return coordinates from 0 to 1000 in the format [ymin, xmin, ymax, xmax]
    # https://ai.google.dev/gemini-api/docs/vision?lang=python#bbox
    SYSTEM_PROMPT = """You are an expert computer vision system. You will be given an image and asked to find a specific object within it. The object may not be present in the image.

Provide your result as a JSON object with the following structure:
{
  "bbox": [ymin, xmin, ymax, xmax],
  "description": string,
  "confidence": number
}
Where:
- bbox: The bounding box around the object you identified. All coordinates should be normalized to be between 0 and 1000, where (0,0) is the top-left corner of the image and (1000,1000) is the bottom-right corner.
- description: A detailed description of the object you identified. Describe specifically where it is located in the image. For example: "it is in the top third of the image, slightly left of center. It is above a red beach ball." Also describe its colour, pose, activity, and any other relevant details.
- confidence: Your confidence score for the identification, between 0 and 1. For example, if you are 50% sure the object is where you have described, provide a confidence of 0.5.

If you do not see the object, provide a confidence score of 0 and the bounding box coordinates as [0, 0, 0, 0].

Provide your final output ONLY in the JSON format described above.
"""

    def __init__(self, api_key, model):
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(
            model, system_instruction=self.SYSTEM_PROMPT
        )
        self.model = model

    def detect_object(self, image_path: str, object_name: str):
        pil_image = Image.open(image_path)

        response = self.client.generate_content([pil_image, "Find the {object_name}"])

        # Gemini likes adding a code fence
        data = json.loads(response.text.replace("```json\n", "").replace("\n```", ""))

        # Scale bounding box coordinates to 0-1 for drawing
        data["bbox"] = [coord / 1000 for coord in data["bbox"]]
        # Move around x and y coordinates
        data["bbox"] = [
            data["bbox"][1],
            data["bbox"][0],
            data["bbox"][3],
            data["bbox"][2],
        ]
        return data
