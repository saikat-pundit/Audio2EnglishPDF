import os
import sys
import re
import gdown
from PIL import Image
from pdf2image import convert_from_path
from transformers import AutoModel, AutoTokenizer
import torch
import markdown
import pdfkit

def extract_file_id(link):
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', link)
    if match:
        return match.group(1)
    raise ValueError("Could not extract file ID from Google Drive link")

def get_original_filename(file_id):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    try:
        import requests
        response = requests.head(url, allow_redirects=True)
        if 'Content-Disposition' in response.headers:
            filename_match = re.search(r'filename="(.+?)"', response.headers['Content-Disposition'])
            if filename_match:
                return filename_match.group(1)
    except:
        pass
    return None

def download_from_gdrive(link, output_path):
    file_id = extract_file_id(link)
    gdown.download(id=file_id, output=output_path, quiet=False)
    return get_original_filename(file_id)

def convert_pdf_to_images(pdf_path, output_folder="pdf_images"):
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
    image = Image.open(image_path)
    prompt = "Convert the document to markdown."
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=2048)
    result = processor.decode(outputs[0], skip_special_tokens=True)
    if result.startswith(prompt):
        result = result[len(prompt):].lstrip()
    return result

def main():
    if len(sys.argv) != 2:
        print("Usage: python ocr_process.py <google_drive_link>")
        sys.exit(1)
    drive_link = sys.argv[1]

    print("Downloading file from Google Drive...")
    original_name = download_from_gdrive(drive_link, "temp_download")
    temp_file = "temp_download"
    if original_name:
        os.rename(temp_file, original_name)
        temp_file = original_name

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

    print("Loading DeepSeek-OCR model...")
    model_name = "deepseek-ai/DeepSeek-OCR"
    processor = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print("Performing OCR...")
    all_markdown_text = []
    for img_path in image_paths:
        print(f"Processing {img_path}...")
        ocr_result = perform_ocr_on_image(img_path, model, processor)
        all_markdown_text.append(ocr_result)
        if os.path.exists(img_path):
            os.remove(img_path)
    if os.path.exists(temp_file):
        os.remove(temp_file)

    combined_markdown = "\n\n".join(all_markdown_text)
    html = markdown.markdown(combined_markdown)
    pdfkit.from_string(html, pdf_output_name)

    print(f"OCR complete. PDF saved as {pdf_output_name}")

if __name__ == "__main__":
    main()
