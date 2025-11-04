# Trademark Indexing API

This project implements an **AI-powered Trademark Image-to-Text Description System**, designed to extract textual and visual information from trademark logos.  
It combines **Optical Character Recognition (OCR)** and **Vision-Language Modeling (VLM)** for structured description generation.

---

## 🧠 Overview

The API performs:
- **English Text Extraction (wordsInMark)** using EasyOCR and BLIP fallback.
- **Chinese Character Recognition (chineseCharacter)** using EasyOCR for both Simplified and Traditional Chinese.
- **Visual Description Generation (descrOfDevice)** using a **fine-tuned BLIP model** trained on a subset of the provided dataset.

Output is returned as JSON and also saved locally in `/outputs/`.

---

## ⚙️ Architecture

```
Base64 Image Input 
    ↓
Decode & Resize
    ↓
EasyOCR (English + Chinese)
    ↓
Confidence-based Fusion
    ↓
BLIP Caption Generation
    ↓
Structured JSON Output
```

---

## 🚀 How to Run

### 1️⃣ Build Docker Image
```bash
docker build -t trademark-api .
```

### 2️⃣ Run Container
```bash
docker run -p 8080:8080 trademark-api
```

### 3️⃣ Send Inference Request
```bash
curl -X POST http://localhost:8080/invoke \
  -H "Content-Type: application/json" \
  -d "{\"image\":\"$(base64 /path/to/image.jpg | tr -d '\n')\"}"
```

---

## 🧾 Example Output
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

## ⚡ Performance
| Environment | Inference Time (per image) |
|--------------|----------------------------|
| CPU (Intel i5) | 30–50 sec |
| GPU (RTX 3060 / A100) | 2–3 sec |

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
Candidate for Senior/Lead Data Scientist – Computer Vision & Generative AI  

---

## 📄 Documentation
See full technical report:  
**`Trademark_Indexing_API_Documentation.docx`** (included in repo)
