"""
Reverse Image Search using SerpApi with a local image (base64 upload).

Requirements:
    pip install requests

Usage:
    python reverse_image_search.py --image /path/to/image.jpg --api-key YOUR_KEY
    python reverse_image_search.py --image /path/to/image.jpg --json
    python reverse_image_search.py --image /path/to/image.jpg --output results.txt
"""

import argparse
import base64
import json
import os

import requests

SERPAPI_ENDPOINT = "https://serpapi.com/search"


IMGBB_ENDPOINT = "https://api.imgbb.com/1/upload"
IMGBB_API_KEY  = "e752cda8cbbaf773e0a62d7c6844784a"   

def upload_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    r = requests.post(IMGBB_ENDPOINT, data={"key": IMGBB_API_KEY, "image": encoded})
    r.raise_for_status()
    return r.json()["data"]["url"]

def reverse_image_search(image_path: str, api_key: str) -> dict:
    image_url = upload_image(image_path)
    print(f"[upload] Image hosted at: {image_url}")

    params = {
        "engine": "google_reverse_image",
        "image_url": image_url,
        "api_key": api_key,
    }
    r = requests.get(SERPAPI_ENDPOINT, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def format_results(data: dict) -> str:
    lines = []

    knowledge = data.get("knowledge_graph", {})
    if knowledge:
        lines.append("--- Knowledge Graph ---")
        lines.append(f"  Title : {knowledge.get('title', 'N/A')}")
        lines.append(f"  Type  : {knowledge.get('type', 'N/A')}")
        lines.append(f"  URL   : {knowledge.get('header_images', [{}])[0].get('source', 'N/A')}")

    similar = data.get("image_results", [])
    if similar:
        lines.append(f"\n--- Visually Similar Images ({len(similar)} results) ---")
        for i, item in enumerate(similar[:5], 1):
            lines.append(f"  [{i}] {item.get('title', 'No title')}")
            lines.append(f"       {item.get('link', '')}")

    pages = data.get("inline_images", []) or data.get("image_sources", [])
    if pages:
        lines.append(f"\n--- Pages Containing This Image ({len(pages)} results) ---")
        for i, item in enumerate(pages[:5], 1):
            lines.append(f"  [{i}] {item.get('source', item.get('link', 'N/A'))}")

    if not knowledge and not similar and not pages:
        lines.append("No results returned. Check your API key.")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Reverse image search via SerpApi.")
    parser.add_argument("--image", required=True, help="Path to the local image file.")
    parser.add_argument("--api-key", default=os.environ.get("SERPAPI_KEY"))
    parser.add_argument("--output", help="Write results to this file (default: print to stdout).")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of summary.")
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Error: provide --api-key or set SERPAPI_KEY.")

    print(f"[search] Searching with image: {args.image}")
    data = reverse_image_search(args.image, args.api_key)
    print(data)

    if args.output :
        with open(args.output, "w") as f:
            f.write(str(data))
        print(f"[done] Results written to: {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()