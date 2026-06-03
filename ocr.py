import os
import base64
import argparse
import ollama
from pathlib import Path

def extract_and_deduplicate_ocr(folder_path, model_name="qwen3.5:2b"):
    folder = Path(folder_path)
    if not folder.is_dir():
        raise ValueError(f"The path '{folder_path}' is not a valid directory.")

    # Supported image extensions
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff'}
    image_files = sorted([f for f in folder.iterdir() if f.suffix.lower() in image_extensions])

    if not image_files:
        print("No images found in the specified folder.")
        return ""

    all_extracted_texts = []
    
    prompt = (
        "Extract all the text from this image. If the text isn't in english, translate it first. "
        "Output ONLY the extracted text. Do not include any introductory phrases, "
        "commentary, or markdown formatting like ```text."
    )

    print(f"Starting OCR on {len(image_files)} images using model '{model_name}'...\n")

    for img_path in image_files:
        print(f"Processing: {img_path.name}...", end=" ", flush=True)
        try:
            with open(img_path, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

            response = ollama.generate(
                model=model_name,
                prompt=prompt,
                images=[img_base64],
                think=False
            )
            
            extracted_text = response.get('response', '').strip()
            
            # Fallback cleanup for markdown code blocks
            if extracted_text.startswith("```"):
                lines = extracted_text.split('\n')
                if len(lines) > 1 and lines[-1].strip() == '```':
                    lines = lines[1:-1]
                else:
                    lines = lines[1:]
                extracted_text = '\n'.join(lines).strip()

            if extracted_text:
                all_extracted_texts.append(extracted_text)
                print("Success.")
            else:
                print("Skipped (No text detected).")
                
        except Exception as e:
            print(f"Error: {e}")

    # 1. Concatenate all texts
    combined_text = "\n".join(all_extracted_texts)

    # 2. Remove duplicate lines while preserving order
    seen = set()
    unique_lines = []
    for line in combined_text.splitlines():
        clean_line = line.strip()
        if clean_line and clean_line not in seen:
            seen.add(clean_line)
            unique_lines.append(clean_line)

    return "\n".join(unique_lines)

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Perform OCR on a folder of images using an Ollama vision model and deduplicate the text."
    )
    
    # Positional argument for the folder path
    parser.add_argument(
        "folder", 
        type=str, 
        help="Path to the folder containing the images."
    )
    
    # Optional arguments
    parser.add_argument(
        "-m", "--model", 
        type=str, 
        default="qwen3.5:2b", 
        help="Name of the Ollama model to use (default: qwen3.5:2b)."
    )
    parser.add_argument(
        "-o", "--output", 
        type=str, 
        default="final_ocr_output.txt", 
        help="Name of the output text file (default: final_ocr_output.txt)."
    )
    parser.add_argument(
        "--no-save", 
        action="store_true", 
        help="Do not save the output to a file, only print to the console."
    )

    args = parser.parse_args()

    # Run the OCR function
    try:
        final_result = extract_and_deduplicate_ocr(args.folder, args.model)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Print the result to the console
    print("\n" + "="*50)
    print("FINAL CONCATENATED & DEDUPLICATED TEXT:")
    print("="*50 + "\n")
    print(final_result)
    
    # Save to file unless --no-save is passed
    if not args.no_save:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(final_result)
        print(f"\n✅ Successfully saved the final text to '{args.output}'")

if __name__ == "__main__":
    main()