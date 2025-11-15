import os
import tempfile
import time
import uuid
import base64
from io import BytesIO
from typing import List

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
import soundfile as sf
import numpy as np
import torch
import whisper
from sentence_transformers import SentenceTransformer
import tensorflow as tf
from tensorflow.keras.models import load_model

# PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from PIL import Image

# Import utils
from app.utils import (
    extract_mel_spectrogram,
    z_normalize,
    pad_to_length,
    segment_audio_by_seconds,
    plot_spectrogram_base64,
    SAMPLE_RATE,
    MAX_SEG_LEN
)

# ---------------------- Config ----------------------
THRESHOLD = 0.5
BATCH_SIZE = 16
REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)
LATEST_REPORT_FILENAME = os.path.join(REPORTS_DIR, "latest_report.pdf")

# ---------------------- GPU / Model Setup ----------------------
gpus = tf.config.list_physical_devices("GPU")
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

device = "cuda" if torch.cuda.is_available() else "cpu"
asr_model = whisper.load_model("small").to(device)
sent_model = SentenceTransformer("all-MiniLM-L6-v2")

AUDIO_ENCODER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "audio_encoder.keras")
FUSION_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "fusion_model.keras")
audio_encoder = load_model(AUDIO_ENCODER_PATH)
fusion_model = load_model(FUSION_MODEL_PATH)

app = FastAPI(
    title="Depression Detection API",
    description=(
        "Upload an audio file to analyze for depression. "
        "The system returns a prediction and generates a detailed graphical PDF report."
    ),
    version="2.1"
)

_last_report_path = None  # cleared on each new prediction


# ---------------------- Helpers ----------------------
def safe_decode_base64_image(b64: str) -> BytesIO:
    if b64.startswith("data:image"):
        _, b64data = b64.split(",", 1)
    else:
        b64data = b64
    img_bytes = base64.b64decode(b64data)
    return BytesIO(img_bytes)


def transcribe_audio_segment(segment: np.ndarray, sr: int = SAMPLE_RATE) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_name = tmp.name
    try:
        sf.write(tmp_name, segment.astype(np.float32), sr)
        result = asr_model.transcribe(tmp_name, fp16=False)
        return result.get("text", "").strip()
    finally:
        try:
            os.remove(tmp_name)
        except Exception:
            pass


def split_text_to_lines(text: str, max_chars: int = 80) -> List[str]:
    words = text.split()
    if not words:
        return ["(no transcription)"]
    lines, current = [], ""
    for w in words:
        if len(current) + 1 + len(w) <= max_chars:
            current += (" " if current else "") + w
        else:
            lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines


def build_pdf_report(report_path: str,
                     title: str,
                     segment_images_b64: List[str],
                     segment_texts: List[str],
                     segment_probs: List[float],
                     final_pred_str: str,
                     final_prob: float):
    PAGE_W, PAGE_H = A4
    margin = 50
    line_gap = 14
    current_y = PAGE_H - margin

    c = canvas.Canvas(report_path, pagesize=A4)
    c.setTitle(title)

    # Title
    c.setFont("Helvetica-Bold", 20)
    c.drawString(margin, current_y, title)
    current_y -= 30

    c.setFont("Helvetica", 11)
    c.drawString(margin, current_y, f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    current_y -= 20

    # Classification
    c.setFont("Helvetica-Bold", 13)
    c.drawString(margin, current_y, f"Final Classification: {final_pred_str}")
    c.setFont("Helvetica", 12)
    c.drawString(margin + 300, current_y, f"Final Probability: {final_prob:.4f}")
    current_y -= 40

    # Segments
    for idx, (img_b64, text, prob) in enumerate(zip(segment_images_b64, segment_texts, segment_probs), start=1):
        if current_y < 200:
            c.showPage()
            current_y = PAGE_H - margin

        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, current_y, f"Segment {idx}  |  Probability: {prob:.4f}")
        current_y -= 20

        img_data = safe_decode_base64_image(img_b64)
        img = Image.open(img_data).convert("RGB")
        max_width, max_height = PAGE_W - 2 * margin, 200
        w, h = img.size
        ratio = min(max_width / w, max_height / h)
        img = img.resize((int(w * ratio), int(h * ratio)))

        img_io = BytesIO()
        img.save(img_io, format="PNG")
        img_io.seek(0)
        c.drawImage(ImageReader(img_io), margin, current_y - img.height, width=img.width, height=img.height)
        current_y -= img.height + 15

        c.setFont("Helvetica", 10)
        for line in split_text_to_lines(text):
            if current_y - line_gap < margin:
                c.showPage()
                current_y = PAGE_H - margin
            c.drawString(margin, current_y, line)
            current_y -= line_gap

        current_y -= 20

    c.save()


# ---------------------- Endpoints ----------------------
@app.post("/predict", response_class=JSONResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Runs full analysis:
    - Processes audio and transcripts
    - Generates a graphical PDF report
    - Returns final prediction and a link to the report
    """
    global _last_report_path
    _last_report_path = None  # reset previous state on new upload

    audio_bytes = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        temp_path = tmp.name

    try:
        audio, sr = sf.read(temp_path)
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass

    if sr != SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE

    segments = segment_audio_by_seconds(audio, sr)
    if not segments:
        return JSONResponse({"final_prediction": "Not Depressed", "report_link": "/download_report"})

    mel_segments = np.array([
        pad_to_length(z_normalize(extract_mel_spectrogram(seg)))
        for seg in segments
    ])[..., np.newaxis]

    audio_emb = audio_encoder.predict(mel_segments, batch_size=BATCH_SIZE)
    texts = [transcribe_audio_segment(seg, sr) for seg in segments]
    text_emb = sent_model.encode(texts, convert_to_numpy=True)
    probs = fusion_model.predict([audio_emb, text_emb], batch_size=BATCH_SIZE).flatten()
    final_prob = float(np.mean(probs))
    final_pred = "Depressed" if final_prob >= THRESHOLD else "Not Depressed"
    segment_images_b64 = [plot_spectrogram_base64(seg, sr, title="") for seg in segments]

    ts = time.strftime("%Y%m%d_%H%M%S")
    filename = f"report_{ts}_{uuid.uuid4().hex[:8]}.pdf"
    report_path = os.path.join(REPORTS_DIR, filename)

    build_pdf_report(report_path, "Depression Detection Report",
                     segment_images_b64, texts, probs.tolist(),
                     final_pred, final_prob)

    _last_report_path = report_path
    with open(report_path, "rb") as rf, open(LATEST_REPORT_FILENAME, "wb") as lf:
        lf.write(rf.read())

    return JSONResponse({
        "final_prediction": final_pred,
        "final_probability": round(final_prob, 4),
        "report_link": "/download_report"
    })


@app.get("/download_report", response_class=HTMLResponse)
async def view_or_download_report():
    """
    Displays the latest report inline with a download button.
    """
    global _last_report_path
    path = _last_report_path if _last_report_path else (
        LATEST_REPORT_FILENAME if os.path.exists(LATEST_REPORT_FILENAME) else None
    )
    if not path or not os.path.exists(path):
        return JSONResponse({"error": "No report available. Please run /predict first."}, status_code=404)

    filename = os.path.basename(path)
    html = f"""
    <html>
    <head>
        <title>Depression Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #f9fafc;
                margin: 0;
                padding: 20px;
            }}
            .container {{
                text-align: center;
            }}
            iframe {{
                width: 80%;
                height: 90vh;
                border: 1px solid #ccc;
                border-radius: 10px;
            }}
            .download-btn {{
                background-color: #4CAF50;
                color: white;
                padding: 12px 20px;
                border: none;
                border-radius: 8px;
                text-decoration: none;
                font-size: 16px;
                display: inline-block;
                margin-top: 20px;
                transition: background-color 0.3s;
            }}
            .download-btn:hover {{
                background-color: #45a049;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Depression Analysis Report</h2>
            <iframe src="/view_pdf/{filename}" frameborder="0"></iframe><br/>
            <a href="/view_pdf/{filename}?download=true" class="download-btn">⬇️ Download PDF</a>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/view_pdf/{filename}")
async def serve_pdf(filename: str, download: bool = False):
    """
    Internal route to view or download PDF file.
    """
    path = os.path.join(REPORTS_DIR, filename)
    if not os.path.exists(path):
        return JSONResponse({"error": "Report not found."}, status_code=404)
    if download:
        return FileResponse(path, filename=filename, media_type="application/pdf")
    return FileResponse(path, media_type="application/pdf")
