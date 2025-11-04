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
## Folder structure

Trademark_Indexing_Assignment/
├── app.py
├── requirements.txt
├── Dockerfile
├── /model/          # optional placeholder for BLIP fine-tuned model
├── /outputs/        # auto-generated
├── Trademark_Indexing_API_Documentation.docx
└── README.md


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
3. Provide the model path to the inference API request call (see below)

## How to Run

### Build Docker Image
```bash
docker build -t trademark-api .
```

### Run Container
```bash
docker run -p 8080:8080 trademark-api
```

### Send Inference Request
```bash
curl -X POST http://localhost:8080/invoke \
  -H "Content-Type: application/json" \
  -d "{\"image\":\"$(base64 /path/to/image.jpg | tr -d '\n')\"}"
```

---

## Example Output
```json
{
  "wordsInMark": "silverstone",
  "chineseCharacter": "",
  "descrOfDevice": "a logo showing the word silverstone and a circle shaped device."
}
```

---

## 📦 Outputs Saved
Each run creates two JSON files in `/app/outputs/`:
- `<filename>_main_output.json` → model’s textual and visual summary  
- `<filename>_meta_data.json` → performance metrics, OCR & BLIP raw results  

---

## 🧰 Tech Stack
- **FastAPI** – REST API framework  
- **PyTorch** – model inference (BLIP)  
- **EasyOCR** – multilingual OCR  
- **Prometheus** – metrics logging  
- **Docker** – containerization  

---

## 🧑‍💼 Author
**Ashish Saxena**  
Candidate for Senior/Lead Data Scientist – Computer Vision  

---

## 📄 Documentation
See full technical report:  
**`Trademark_Indexing_API_Documentation.docx`** (included in repo)
