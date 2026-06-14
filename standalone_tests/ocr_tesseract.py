#!/usr/bin/env python3
"""
OCR all images in a folder using Tesseract, combine outputs, and remove duplicate lines.

Usage:
    python ocr_folder.py <image_folder> [output_file]

Dependencies:
    pip install pytesseract Pillow
    # Also requires Tesseract binary: https://github.com/tesseract-ocr/tesseract
    # Ubuntu/Debian: sudo apt install tesseract-ocr
    # macOS:         brew install tesseract
    # Windows:       https://github.com/UB-Mannheim/tesseract/wiki
"""

import sys
import argparse
from pathlib import Path
from PIL import Image
import pytesseract

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif", ".webp"}


def ocr_image(path: Path, lang: str) -> str:
    """Run Tesseract OCR on a single image and return the extracted text."""
    try:
        img = Image.open(path)
        text = pytesseract.image_to_string(img, lang=lang)
        return text
    except Exception as e:
        print(f"  Warning: failed to OCR {path.name}: {e}", file=sys.stderr)
        return ""


def normalize_line(line: str) -> str:
    """Normalize a line for deduplication (strip + collapse whitespace + lowercase)."""
    return " ".join(line.lower().split())


def deduplicate(text: str) -> str:
    """Remove duplicate lines while preserving original casing and order."""
    seen: set[str] = set()
    result: list[str] = []
    for line in text.splitlines():
        key = normalize_line(line)
        if not key:
            # Preserve blank lines only once between blocks
            if result and result[-1] != "":
                result.append("")
            continue
        if key not in seen:
            seen.add(key)
            result.append(line)
    # Strip leading/trailing blank lines
    while result and result[0] == "":
        result.pop(0)
    while result and result[-1] == "":
        result.pop()
    return "\n".join(result)


def main():
    parser = argparse.ArgumentParser(
        description="OCR all images in a folder, combine and deduplicate the text."
    )
    parser.add_argument("folder", help="Path to folder containing images")
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Output text file (default: print to stdout)",
    )
    parser.add_argument(
        "--lang",
        default="eng",
        help="Tesseract language code(s), e.g. 'eng', 'fra', 'eng+fra' (default: eng)",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable deduplication and output raw combined text",
    )
    parser.add_argument(
        "--separator",
        default="\n---\n",
        help="Separator inserted between each file's output (default: '\\n---\\n')",
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Error: '{folder}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    images = sorted(
        p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        print(f"No images found in '{folder}'.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(images)} image(s). Running OCR...", file=sys.stderr)

    parts: list[str] = []
    for img_path in images:
        print(f"  Processing: {img_path.name}", file=sys.stderr)
        text = ocr_image(img_path, args.lang)
        if text.strip():
            parts.append(text.strip())

    combined = args.separator.join(parts)

    if args.no_dedup:
        final = combined
    else:
        print("Deduplicating lines...", file=sys.stderr)
        final = deduplicate(combined)

    if args.output:
        Path(args.output).write_text(final, encoding="utf-8")
        print(f"Output written to: {args.output}", file=sys.stderr)
    else:
        print(final)


if __name__ == "__main__":
    main()