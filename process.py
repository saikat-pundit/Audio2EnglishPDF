import os
import sys
import re
import gdown
import whisper
from pydub import AudioSegment
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def extract_file_id(link):
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', link)
    if match:
        return match.group(1)
    raise ValueError("Could not extract file ID from Google Drive link")

def download_from_gdrive(link, output_path):
    file_id = extract_file_id(link)
    gdown.download(id=file_id, output=output_path, quiet=False)

def convert_m4a_to_wav(m4a_path, wav_path):
    audio = AudioSegment.from_file(m4a_path, format="m4a")
    audio.export(wav_path, format="wav")

def transcribe_with_whisper(wav_path):
    model = whisper.load_model("base")
    result = model.transcribe(wav_path, language="hi", task="transcribe")
    return result["text"]

def translate_hindi_to_english(hindi_text):
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-hi-en")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-hi-en")
    inputs = tokenizer(hindi_text, return_tensors="pt", truncation=True, max_length=512)
    translated = model.generate(**inputs, max_length=512)
    english_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return english_text

def create_pdf(text, pdf_path):
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    y = height - 40
    for line in text.split("\n"):
        c.drawString(40, y, line)
        y -= 14
        if y < 40:
            c.showPage()
            y = height - 40
    c.save()

def main():
    if len(sys.argv) != 2:
        print("Usage: python process.py <google_drive_link>")
        sys.exit(1)
    drive_link = sys.argv[1]
    m4a_file = "input.m4a"
    wav_file = "input.wav"
    pdf_file = "output.pdf"
    print("Downloading from Google Drive...")
    download_from_gdrive(drive_link, m4a_file)
    print("Converting M4A to WAV...")
    convert_m4a_to_wav(m4a_file, wav_file)
    print("Transcribing with Whisper (Hindi+English)...")
    hindi_mixed_text = transcribe_with_whisper(wav_file)
    print("Translating Hindi portions to English...")
    english_text = translate_hindi_to_english(hindi_mixed_text)
    print("Creating PDF...")
    create_pdf(english_text, pdf_file)
    print(f"Done. PDF saved as {pdf_file}")

if __name__ == "__main__":
    main()
