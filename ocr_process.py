import os
import sys
import re
import gdown
import requests
from PIL import Image
from pdf2image import convert_from_path
from transformers import AutoModel, AutoTokenizer
import torch
import markdown
import pdfkit
from io import BytesIO

def extract_file_id(link):
    """Extract file ID from a Google Drive shareable link."""
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', link)
    if match:
        return match.group(1)
    raise ValueError("Could not extract file ID from Google Drive link")

def download_from_gdrive(link, output_path):
    file_id = extract_file_id(link)
    gdown.download(id=file_id, output=output_path, quiet=False)
    original_name = get_original_filename(file_id)
    return original_name

def convert_pdf_to_images(pdf_path, output_folder="pdf_images"):
    """Convert each page of a PDF to an image."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    images = convert_from_path(pdf_path)
    image_paths = []
    for i, image in enumerate(images):
        img_path = os.path.join(output_folder, f"page_{i+1}.png")
        image.save(img_path, "PNG")
        image_paths.append(img_path)
    return image_paths

def perform_ocr_on_image(image_path, model, processor):
    """Perform OCR on a single image using DeepSeek-OCR."""
    image = Image.open(image_path)
    # Prompt the model to convert the document to markdown
    prompt = "Convert the document to markdown."
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=2048)
    result = processor.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from the output
    if result.startswith(prompt):
        result = result[len(prompt):].lstrip()
    return result

def main():
    if len(sys.argv) != 2:
        print("Usage: python ocr_process.py <google_drive_link>")
        sys.exit(1)
    drive_link = sys.argv[1]
    
    # 1. Download the file
    print("Downloading file from Google Drive...")
    # Download and get original filename
original_name = download_from_gdrive(drive_link, "temp_download")
temp_file = "temp_download"
# Rename to original name if possible
if original_name:
    os.rename(temp_file, original_name)
    temp_file = original_name

# Determine file type by extension
image_paths = []
if temp_file.lower().endswith('.pdf'):
    print("PDF detected. Converting to images...")
    image_paths = convert_pdf_to_images(temp_file)
    base_name = os.path.splitext(os.path.basename(temp_file))[0]
    pdf_output_name = f"{base_name}_extracted_text.pdf"
else:
    print("Image detected. Processing directly...")
    image_paths = [temp_file]
    base_name = os.path.splitext(os.path.basename(temp_file))[0]
    pdf_output_name = f"{base_name}_ocr_output.pdf"
    
    # 2. Load DeepSeek-OCR model (only once, but for simplicity, we load it here)
    print("Loading DeepSeek-OCR model (this may take a while on first run)...")
    model_name = "deepseek-ai/DeepSeek-OCR"
    processor = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # 3. Perform OCR on all images and combine results
    print("Performing OCR...")
    all_markdown_text = []
    for img_path in image_paths:
        print(f"Processing {img_path}...")
        ocr_result = perform_ocr_on_image(img_path, model, processor)
        all_markdown_text.append(ocr_result)
        # Clean up temporary image file
        if os.path.exists(img_path):
            os.remove(img_path)
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    # 4. Convert Markdown to HTML and then to PDF
    combined_markdown = "\n\n".join(all_markdown_text)
    html = markdown.markdown(combined_markdown)
    pdfkit.from_string(html, pdf_output_name)
    
    print(f"OCR complete. PDF saved as {pdf_output_name}")
    return pdf_output_name

if __name__ == "__main__":
    main()
