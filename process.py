import os
import sys
import gdown
from pydub import AudioSegment
import torch
from seamless_communication import load_model, load_processor
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def download_from_gdrive(link, output_path):
    gdown.download(link, output_path, quiet=False, fuzzy=True)

def convert_m4a_to_wav(m4a_path, wav_path):
    audio = AudioSegment.from_file(m4a_path, format="m4a")
    audio.export(wav_path, format="wav")

def transcribe_and_translate(wav_path):
    model = load_model("seamlessM4T_large", device="cuda" if torch.cuda.is_available() else "cpu")
    processor = load_processor("seamlessM4T_large")
    audio_input, sample_rate = torchaudio.load(wav_path)
    audio_input = audio_input.mean(dim=0)  
    text_output = model.generate(audio_input, task="s2tt", tgt_lang="eng")
    return text_output[0]["text"]

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
    print("Running SeamlessM4T (this may take a while)...")
    english_text = transcribe_and_translate(wav_file)
    print("Creating PDF...")
    create_pdf(english_text, pdf_file)
    print(f"Done. PDF saved as {pdf_file}")

if __name__ == "__main__":
    main()
