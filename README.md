# 🧵 LabelSense

**Intelligent Defect Detection in Textile Labels**  
*Zero‑Shot YOLO Deployment with FastAPI & ONNX*

<p align="center">
  <img src="https://img.shields.io/badge/Model-YOLOv8-blue" />
  <img src="https://img.shields.io/badge/Server-FastAPI-green" />
  <img src="https://img.shields.io/badge/License-MIT-black" />
</p>

---

## 🌐 Overview

**LabelSense** is an AI solution that **automates quality‑control** by detecting defects in textile labels—printing errors, stains, structural flaws—directly from production‑line images.  
Unlike classic approaches that require thousands of images per defect, LabelSense employs **zero‑shot detection**: the YOLO model can localise **a defect class it never saw during training**.

> This project was built for the **DataQuest 2025** hackathon, where scoring is split 50‑50 between model accuracy and the interface / visualisation module.

---

## 🎯 Key Features

- **Zero‑Shot YOLO** — detect unseen defect classes with textual prompts.
- **ONNX Runtime** — lightweight inference with GPU/CPU fallback.
- **FastAPI** — one `POST /predict` endpoint for image uploads.
- **CORS enabled** — plug into any web / mobile frontend.
- **Plug‑and‑Play** — swap `DragNet.onnx` with any YOLO‑family model.

---

## 🖼️ Model Pipeline

```mermaid
graph TD
    A[Upload Image] --> B[Pre‑processing 640×640]
    B --> C[YOLOv8 ONNX (Zero‑Shot)]
    C --> D[Post‑processing]
    D --> E[JSON Response]
```

**Zero‑shot trick:** we attach a textual embedding (e.g. `"mis_print"`) to the detection head — no extra images required.

---

## ⚙️ Architecture

### 1 · Pre‑processing

- Resize or letterbox to **640 × 640**.
- Convert grayscale ➜ RGB if needed.
- Change layout **HWC → CHW**, add batch dim.

### 2 · Inference

- Load `DragNet.onnx` in **ONNX Runtime**.
- Output: bounding‑boxes, confidence, class‑ids.
- Map zero‑shot classes via `names.json`.

### 3 · FastAPI Service

```http
POST /predict
Content‑Type: multipart/form‑data
file=<image>
→ { "prediction": [...] }
```

---

## 🚀 Quickstart

```bash
# 1 · Clone + install deps
$ git clone https://github.com/your-org/LabelSense.git
$ cd LabelSense
$ pip install -r requirements.txt

# 2 · Run locally
$ uvicorn app:app --reload --host 0.0.0.0 --port 8000

# 3 · Test
$ curl -F "file=@examples/label.jpg" http://localhost:8000/predict
```

---

## 📊 Hackathon Scoring

| Criterion | Weight |
|-----------|-------:|
| **Model Accuracy**<br/>Precision · Recall · F1 · mAP | **89 %** |


LabelSense fully covers the accuracy component and exposes a REST API so any frontend team can build the analytics dashboard.

---

## 🤝 Contributing

1. Fork the project  
2. Create your feature branch: `git checkout -b feat/awesome`  
3. Commit changes: `git commit -m "feat: add awesome"`  
4. Push: `git push origin feat/awesome`  
5. Open a Pull Request

---

<p align="center"><strong>Made with ❤️ by the LabelSense Team · IEEE INSAT DataQuest 2025</strong></p>

