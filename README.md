---
title: ImgAuth AI
emoji: 🛡️
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# ImgAuth AI — Image Authenticity Detector

🛡️ **ImgAuth AI** is a state-of-the-art web application designed to detect AI-generated and manipulated images. Built with a "simple on the surface, powerful underneath" philosophy, it combines deep learning models with advanced digital forensics heuristics to deliver clear, binary verdicts: **Likely AI-Generated** or **Likely Authentic**.

Developed as a Major Project by **Team VisionGuard** (student team of 4).

---

## 🚀 Key Features

- **Binary Classification**: Simplified verdicts removing ambiguity ("Likely AI-Generated" or "Likely Authentic").
- **Deep Learning Ensemble**: Combined predictions from 3 Hugging Face model pipelines:
  - `umm-maybe/AI-image-detector`
  - `dima806/ai_vs_real_image_detection`
  - `Organika/sdxl-detector`
- **5 Forensic Heuristics**: Multi-layer analysis for technical validation:
  1. *Noise Kurtosis Analysis* (checks high-frequency noise distributions)
  2. *Deep Feature Inconsistency (DFI)* (checks patch-level consistency of Vision Transformer embeddings)
  3. *FFT Spectral Analysis* (identifies periodic artifacts in frequency domain)
  4. *Color Histogram Analysis* (detects synthetic pixel roughness/smoothness)
  5. *JPEG Ghost Analysis* (detects double compression artifacts in JPEG files)
- **AI Focus Areas (Explainability)**: Visual heatmaps showing ViT Attention Maps and Deep Feature Inconsistencies.
- **Collapsible Technical Drawer**: Advanced forensic signal logs, weights, and metrics available for researchers, while maintaining a clean, technical-jargon-free interface for everyday users.
- **Privacy First**: Fully stateless architecture; no images are stored permanently. Scanning history is saved only in local browser storage (`localStorage`).

---

## 👥 Meet Team VisionGuard

- **Vishal Chauhan** (Computer Science & Engineering, Project Lead)
- **Prince Mishra** (Computer Science & Engineering, Backend Developer)
- **Prince Dubey** (Computer Science & Engineering, Security & Testing)
- **Raksha** (Computer Science & Engineering, Frontend Developer)

---

## 🛠️ Technology Stack

- **Backend**: FastAPI, Uvicorn, PyTorch, Hugging Face Transformers, OpenCV, NumPy, SciPy
- **Frontend**: Vanilla HTML5, CSS3 (Modern dark-theme layout with purple gradients & glassmorphism), Vanilla JavaScript
- **Deployment**: Docker, Hugging Face Spaces

---

## 💻 Local Setup and Running

To run this application locally on your machine, follow these steps:

### Prerequisites
- Python 3.10+
- Pip package manager

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd imgauth-ai
   ```

2. **Create and activate a virtual environment**:
   - **Windows (PowerShell)**:
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the server**:
   ```bash
   python run.py
   ```
   *The app will start running at:* `http://localhost:5000`

---

## 🐳 Running with Docker

Alternatively, build and run via Docker:

1. **Build the image**:
   ```bash
   docker build -t imgauth-ai .
   ```

2. **Run the container**:
   ```bash
   docker run -p 7860:7860 imgauth-ai
   ```
   *Open browser to:* `http://localhost:7860`

---

## ⚖️ License & Attribution

- **Non-Commercial**: This project uses the `Organika/sdxl-detector` model, licensed under CC BY-NC 4.0. It is intended strictly for non-commercial educational and research purposes.
- **Model Attribution**: All deep learning classifications are handled by model weights published by the Hugging Face community.
