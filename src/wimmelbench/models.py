import json
import os

from anthropic import Anthropic
from openai import OpenAI

SYSTEM_PROMPT = """You are an expert computer vision system. You will be given an image and asked to find a specific object within it.

Provide your result as a JSON object with the following structure:
{
  "bbox": [x1, y1, x2, y2],
  "description": string,
  "confidence": number
}
Where:
- bbox: The bounding box around the object you identified.
    - All coordinates should be normalized to be between 0 and 1, where (0,0) is the top-left corner of the image and (1,1) is the bottom-right corner.
    - x1, y1: The coordinates of the top-left corner of the bounding box.
    - x2, y2: The coordinates of the bottom-right corner of the bounding box.
- description: A detailed description of the object you identified. Describe specifically where it is located in the image. For eg. "it is in the top third of the image, slightly left of center. It is above a red beach ball." Also describe its colour, pose, activity, and any other relevant details.
- confidence: Your confidence score for the identification, between 0 and 1. For eg, if you are 50% sure the object is where you have described, provide a confidence of 0.5.
- If you do not see the object, provide a confidence score of 0 and the bounding box coordinates as [0, 0, 0, 0].

Provide your final output ONLY in the JSON format described above.
"""


class AnthropicModel:
    def __init__(self, model="claude-3-5-sonnet-20241022"):
        self.client = Anthropic(
            api_key=os.environ.get(
                "ANTHROPIC_API_KEY", "could-not-find-anthropic-api-key"
            ),
        )
        self.model = model

    def detect_object(
        self, base64_image: str, object_name: str, max_tokens=1024
    ) -> dict:
        response = self.client.messages.create(
            model=self.model,
            system=SYSTEM_PROMPT,
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
                            "text": f"The object you should locate is: {object_name}",
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": "{",
                },
            ],
        )

        data = "{" + response.content[0].text  # type: ignore
        return json.loads(data)


class OpenAIModel:
    def __init__(self, model="gpt-4o-2024-08-06"):
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", "could-not-find-openai-api-key")
        )
        self.model = model

    def detect_object(
        self, base64_image: str, object_name: str, max_tokens=1024
    ) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
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
                            "text": f"The object you should locate is: {object_name}",
                        },
                    ],
                },
            ],
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )

        data = response.choices[0].message.content
        return json.loads(data)  # type: ignore
