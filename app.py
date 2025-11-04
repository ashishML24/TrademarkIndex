# app.py
import base64
import io
import json
import os
import time
import re
import logging
from typing import Dict, Any
from datetime import datetime

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError, ImageOps, ImageFilter

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

import easyocr
from prometheus_client import CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST, Counter, Histogram
import psutil

# --------------------------
# Config
# --------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/model")
OCR_LANGS = os.environ.get("OCR_LANGS", "en,ch_sim,ch_tra")  # Added Traditional Chinese
OCR_CONF_THRESHOLD = float(os.environ.get("OCR_CONF_THRESHOLD", "0.80"))
OCR_BORDERLINE_LOW = 0.10
OCR_BORDERLINE_HIGH = 0.40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_IMAGE_SIZE = int(os.environ.get("MAX_IMAGE_SIZE", "2048"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/app/outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------
# Logging setup
# --------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("trademark-indexing-api")

# --------------------------
# Monitoring metrics
# --------------------------
app = FastAPI(title="Trademark Indexing API", description="Generates structured trademark description from images")

registry = CollectorRegistry()
REQUEST_COUNT = Counter('invocations_total', 'Total inference requests', ['endpoint', 'status'], registry=registry)
REQUEST_TIME = Histogram('invocation_duration_seconds', 'Time per invocation', ['endpoint'], registry=registry)

# --------------------------
# Pydantic models
# --------------------------
class InvokeRequest(BaseModel):
    image: str  # base64 image
    image_filename: str = None


class InvokeResponse(BaseModel):
    wordsInMark: str
    chineseCharacter: str
    descrOfDevice: str
    metadata: Dict[str, Any]

# --------------------------
# Initialize models
# --------------------------
logger.info(f"Loading BLIP from {MODEL_PATH}")
try:
    processor = BlipProcessor.from_pretrained(MODEL_PATH)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    logger.info("BLIP model loaded successfully.")
except Exception as e:
    logger.exception("Failed to load BLIP model.")
    processor = model = None

langs = [l.strip() for l in OCR_LANGS.split(",") if l.strip()]
logger.info(f"Initializing EasyOCR for langs: {langs}")
ocr_reader = easyocr.Reader(lang_list=langs, gpu=(DEVICE.type == "cuda"))

# --------------------------
# Helper functions
# --------------------------
def base64_to_pil(b64str: str) -> Image.Image:
    """Decode base64 to PIL.Image with robust error handling."""
    try:
        img_bytes = base64.b64decode(b64str)
        if not img_bytes:
            raise ValueError("Empty decoded image bytes")

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        if max(img.size) > MAX_IMAGE_SIZE:
            scale = MAX_IMAGE_SIZE / max(img.size)
            new_size = (int(img.width * scale), int(img.height * scale))
            img = img.resize(new_size, Image.LANCZOS)
        return img

    except (UnidentifiedImageError, OSError, ValueError, base64.binascii.Error) as e:
        logger.error(f"Image decode failed: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid or corrupted image: {str(e)}")


def run_ocr(img: Image.Image, enhanced: bool = False) -> Dict[str, Any]:
    """Run OCR; enhanced=True applies preprocessing before OCR."""
    pil_img = img
    if enhanced:
        pil_img = pil_img.convert("L")
        pil_img = ImageOps.autocontrast(pil_img)
        pil_img = pil_img.filter(ImageFilter.SHARPEN)
        pil_img = ImageOps.invert(pil_img)

    results = ocr_reader.readtext(np.array(pil_img))
    english_texts, chinese_texts = [], []
    english_confs, chinese_confs = [], []
    raw_tokens = []

    for bbox, text, conf in results:
        try:
            conf = float(conf)
        except Exception:
            conf = 0.0
        raw_tokens.append({"text": text, "conf": conf})

        if re.search(r'[\u4e00-\u9fff]', text):
            chinese_texts.append(text)
            chinese_confs.append(conf)
        else:
            english_texts.append(text)
            english_confs.append(conf)

    def agg(txts, confs):
        return {
            "text": " ".join(txts).strip(),
            "conf_mean": float(np.mean(confs)) if confs else 0.0,
            "conf_max": float(np.max(confs)) if confs else 0.0,
            "tokens": [{"text": t, "conf": float(c)} for t, c in zip(txts, confs)]
        }

    return {
        "english": agg(english_texts, english_confs),
        "chinese": agg(chinese_texts, chinese_confs),
        "raw": raw_tokens
    }


def _log_model_processor_info_once():
    try:
        logger.info("Model config: %s", model.config.to_diff_dict())
    except Exception:
        logger.info("Model config unavailable")
    try:
        if hasattr(processor, "tokenizer"):
            tok = processor.tokenizer
            logger.info("Tokenizer class: %s", tok.__class__.__name__)
    except Exception as e:
        logger.warning(f"Tokenizer info unavailable: {e}")


_log_model_processor_info_once()


def generate_caption(img: Image.Image, max_new_tokens=40, num_beams=5, do_sample=False):
    """Generate descriptive caption using BLIP."""
    if model is None or processor is None:
        raise HTTPException(status_code=500, detail="BLIP model not loaded.")
    model.to(DEVICE)
    inputs = processor(images=img, return_tensors="pt").to(DEVICE)

    torch.manual_seed(42)
    np.random.seed(42)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            repetition_penalty=2.5,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True
        )

    sequences = out.sequences if hasattr(out, "sequences") else out[0]
    return processor.decode(sequences[0], skip_special_tokens=True).strip()


def extract_english(text: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9]+", text)
    return " ".join(tokens)


def extract_chinese(text: str) -> str:
    chars = re.findall(r'[\u4e00-\u9fff]+', text)
    return "".join(chars)


def format_wordsInMark_for_indexing(text: str) -> str:
    """Normalize OCR text for trademark names."""
    if not text:
        return ""
    text = text.strip()
    split_tokens = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    split_tokens = re.sub(r'([A-Za-z])(\d)', r'\1 \2', split_tokens)
    split_tokens = re.sub(r'(\d)([A-Za-z])', r'\1 \2', split_tokens)
    spaced = split_tokens.lower().strip()
    compact = re.sub(r'\s+', '', spaced)
    combined = f"{spaced} {compact}".strip()
    return " ".join(dict.fromkeys(combined.split()))


# --------------------------
# Endpoints
# --------------------------
@app.get("/ping")
async def ping():
    return "pong"


@app.get("/metrics")
async def metrics():
    data = generate_latest(registry)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.post("/invoke", response_model=InvokeResponse)
async def invoke(req: InvokeRequest):
    start = time.time()
    endpoint = "/invoke"
    proc = psutil.Process()

    try:
        img = base64_to_pil(req.image)
    except HTTPException as e:
        # Graceful failure for invalid image
        REQUEST_COUNT.labels(endpoint=endpoint, status="400").inc()
        raise e
    except Exception as e:
        logger.exception("Unexpected image decode error")
        REQUEST_COUNT.labels(endpoint=endpoint, status="500").inc()
        raise HTTPException(status_code=500, detail=f"Unexpected image decode error: {str(e)}")

    mem_before = proc.memory_info().rss

    # Run OCR
    ocr_result = run_ocr(img)
    eng_conf_mean = ocr_result["english"].get("conf_mean", 0.0)
    chi_conf_mean = ocr_result["chinese"].get("conf_mean", 0.0)

    # Enhanced OCR for borderline confidence
    if OCR_BORDERLINE_LOW < eng_conf_mean < OCR_BORDERLINE_HIGH or OCR_BORDERLINE_LOW < chi_conf_mean < OCR_BORDERLINE_HIGH:
        enhanced = run_ocr(img, enhanced=True)
        if enhanced["english"].get("conf_mean", 0.0) > eng_conf_mean:
            ocr_result["english"] = enhanced["english"]
        if enhanced["chinese"].get("conf_mean", 0.0) > chi_conf_mean:
            ocr_result["chinese"] = enhanced["chinese"]

    # Run BLIP caption
    try:
        blip_caption = generate_caption(img)
    except Exception as e:
        logger.error(f"BLIP caption generation failed: {e}")
        blip_caption = ""

    eng_conf_max = ocr_result["english"].get("conf_max", 0.0)
    chi_text_raw = ocr_result["chinese"].get("text", "").strip()

    # English logic
    if eng_conf_max >= OCR_CONF_THRESHOLD:
        wordsInMark = format_wordsInMark_for_indexing(ocr_result["english"]["text"])
        src_eng = "ocr"
    else:
        candidate = None
        m = re.search(
            r'\bword(?:s)?\s+(?:["\']?)([A-Za-z0-9\s\-&]{2,80}?)(?=\s+(?:and\s+chinese|and\s+a|and\s+no|showing|device|shape|symbol|logo|design|mark|$))',
            blip_caption, flags=re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
        if not candidate:
            m2 = re.search(r'["\']([^"\']{1,60})["\']', blip_caption)
            if m2:
                candidate = m2.group(1).strip()
        if not candidate:
            m3 = re.search(
                r'\b(?:showing|saying|reads|reads:)\s+(?:the\s+)?(?:word\s+)?([A-Za-z0-9\s\-&]{2,80})(?=\s+(?:and\s+chinese|and\s+a|and\s+no|device|character|shape|symbol|logo|design|mark|$))',
                blip_caption, flags=re.IGNORECASE)
            if m3:
                candidate = m3.group(1).strip()
        if not candidate:
            candidate = extract_english(blip_caption)
        wordsInMark = candidate.strip()
        src_eng = "blip"

    # Chinese logic (always OCR)
    chineseCharacter = chi_text_raw
    src_chi = "ocr"

    descrOfDevice = blip_caption

    total_time = time.time() - start
    mem_after = proc.memory_info().rss

    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "perf": {
            "duration_s": total_time,
            "mem_before_bytes": mem_before,
            "mem_after_bytes": mem_after,
        },
        "ocr_result": ocr_result,
        "blip_caption": blip_caption,
        "chosen_sources": {"english": src_eng, "chinese": src_chi},
    }

    # Skip saving if everything failed
    if not wordsInMark and not chineseCharacter and not descrOfDevice:
        logger.warning("Skipping JSON save: all outputs empty")
        raise HTTPException(status_code=422, detail="Inference failed for this image")

    # Save outputs
    image_filename = req.image_filename or f"inference_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    base_filename = os.path.splitext(os.path.basename(image_filename))[0]

    main_output = {
        "wordsInMark": wordsInMark,
        "chineseCharacter": chineseCharacter,
        "descrOfDevice": descrOfDevice
    }

    meta_output = {
        "image_filename": image_filename,
        "metadata": log_entry
    }

    main_path = os.path.join(OUTPUT_DIR, f"{base_filename}_main_output.json")
    meta_path = os.path.join(OUTPUT_DIR, f"{base_filename}_meta_data.json")

    with open(main_path, "w", encoding="utf-8") as f:
        json.dump(main_output, f, ensure_ascii=False, indent=2)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_output, f, ensure_ascii=False, indent=2)

    print(json.dumps(main_output, ensure_ascii=False, indent=2))
    logger.info(f"Saved main output: {main_path}")
    logger.info(f"Saved metadata: {meta_path}")

    REQUEST_COUNT.labels(endpoint=endpoint, status="200").inc()
    REQUEST_TIME.labels(endpoint=endpoint).observe(total_time)

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return InvokeResponse(
        wordsInMark=wordsInMark,
        chineseCharacter=chineseCharacter,
        descrOfDevice=descrOfDevice,
        metadata=log_entry
    )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
