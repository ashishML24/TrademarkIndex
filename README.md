# Trademark Indexing API

This project implements an **AI-powered Trademark Image-to-Text Description System**, designed to extract textual and visual information from trademark logos.  
It combines **Optical Character Recognition (OCR)** and **Vision-Language Modeling (VLM)** for structured description generation.

---

## Overview

The API performs:
- **English Text Extraction (wordsInMark)** using EasyOCR and BLIP fallback.
- **Chinese Character Recognition (chineseCharacter)** using EasyOCR for both Simplified and Traditional Chinese.
- **Visual Description Generation (descrOfDevice)** using a **fine-tuned BLIP model** trained on a subset of the provided dataset.

Output is returned as JSON and also saved locally in `/outputs/`.

---


## Architecture

```
                  +-----------------------+
                  |   Base64 Image Input  |
                  +-----------+-----------+
                              |
                              v
                      [Decode + Resize]
                              |
                              v
                   +----------+-----------+
                   |      EasyOCR         |
                   | (en, ch_sim, ch_tra) |
                   +----------+-----------+
                              |
          +-------------------+-------------------+
          |                                       |
          v                                       v
 [OCR English tokens]                   [OCR Chinese tokens]
          |                                       |
          |                                       v
          |                               Chinese Characters
          |                                (always OCR-based)
          |
          v
 +------------------------------------+
 | Confidence-based English selection |
 |  - OCR conf_max >= threshold → use  |
 |  - else extract via BLIP caption    |
 +------------------------------------+
          |
          v
 +-------------------------------+
 |        BLIP Model (VLM)       |
 |  a logo showing the word ...  |
 +-------------------------------+
          |
          v
 +------------------------------------+
 |   Structured JSON Output Writer     |
 |   - main_output.json                |
 |   - meta_data.json                  |
 +------------------------------------+

```

---
## BLIP Caption Generation
descrOfDevice is generated as Caption using a BLIP finetuned on Trademark dataset (10K, 50K, 300K).
To use the model:
1. Download the model from https://drive.google.com/drive/folders/1MvKQjdUnogc0BTCN3RLNAMNM8Si-THHI?usp=sharing
2. Copy the model in /model/
3. Provide the model path when running the container (see below)

## How to Run

### Build Docker Image
```
docker build --no-cache -t trademark-indexing:cpu .
```

### Run Container
```
docker run --rm -p 8080:8080 \
  -v /Users/ashu/Desktop/Interview_Prep/GovTech/TrademarkLogoIndexing/model/blip_trademark_ft_2K_Samples:/app/model \
  -v /Users/ashu/Desktop/Interview_Prep/GovTech/TrademarkLogoIndexing/outputs:/app/outputs \
  -v /Users/ashu/Desktop/Interview_Prep/GovTech/TrademarkLogoIndexing/app.py:/app/app.py \
  -e MODEL_PATH=/app/model \
  -e OCR_LANGS="en,ch_sim" \
  -e OCR_CONF_THRESHOLD=0.40 \
  -e OUTPUT_DIR=/app/outputs \
  trademark-indexing:cpu

```

### Send Inference Request
```
IMAGE_PATH="/Users/ashu/Desktop/Interview_Prep/GovTech/trademark_hybrid/data/10k/Images/2004-09-27_19.jpg"
curl -X POST http://localhost:8080/invoke \
  -H "Content-Type: application/json" \
  -d "{\"image\":\"$(base64 "$IMAGE_PATH" | tr -d '\n')\", \"image_filename\":\"$(basename "$IMAGE_PATH")\"}"
```

---

## Example

--> Input image

![Alt text](/outputs/2004-09-27_19.jpg)

--> Output

```
{
  "wordsInMark": "m my choice chinese cuisine",
  "chineseCharacter": "顺  意  莱  馆",
  "descrOfDevice": "a logo showing the word m my choice chinese cuisine and chinese characters and a circle shaped device."
}
```

---

## Outputs Saved
Each run creates two JSON files in `/app/outputs/`:
- `<filename>_main_output.json` → model’s textual and visual summary  
- `<filename>_meta_data.json` → performance metrics, OCR & BLIP raw results  

---

## Tech Stack
- **FastAPI** – REST API framework  
- **PyTorch** – model inference (BLIP)  
- **EasyOCR** – multilingual OCR  
- **Prometheus** – metrics logging  
- **Docker** – containerization  

---

## Author
**Ashish Saxena**  
---

## Documentation
See full technical report:  
**`Trademark_Indexing_API_Documentation.pdf`** (included in repo)
