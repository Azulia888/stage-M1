#!/usr/bin/env python3
"""
Describe an image using Ollama with a Qwen vision model.
Usage: python describe_image.py <image_path> [prompt]
"""

import sys
import ollama
from pathlib import Path


def describe_image(image_path: str, prompt: str = "Describe this image in detail.") -> str:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    response = ollama.chat(
        model="qwen3.5:2b",  
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [str(path)],  # pass path directly; ollama handles base64 encoding
            }
        ],
    )

    return response.message.content


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python describe_image.py <image_path> [prompt]")
        sys.exit(1)

    image_path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Recognize the faces in this picture. "

    print(describe_image(image_path, prompt))