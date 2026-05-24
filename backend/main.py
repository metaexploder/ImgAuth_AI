from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime
from pathlib import Path

from backend.detectors.image_detector import analyze_image_models
from backend.detectors.metadata_detector import analyze_metadata
from backend.detectors.scoring_engine import calculate_final_score
from backend.detectors.style_detector import analyze_visual_style
from PIL import Image
import io

app = FastAPI(title="ImgAuth AI API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE_DIR = Path(__file__).parent
FRONTEND = BASE_DIR.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND / "static")), name="static")


@app.on_event("startup")
async def startup():
    print("ImgAuth AI started -> http://localhost:5000")
    try:
        print("[ImgAuth] Preloading models on startup...")
        from backend.detectors.image_detector import load_models
        load_models()
        print("[ImgAuth] Models preloaded successfully.")
    except Exception as e:
        print(f"[ImgAuth] Warning: failed to preload models: {e}")


@app.get("/")
async def root():
    return FileResponse(str(FRONTEND / "index.html"))


@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only image files are accepted.")
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(400, "File too large. Max 10 MB.")

    fn = {}
    md = analyze_metadata(contents)
    ml = analyze_image_models(contents)
    
    try:
        img_full = Image.open(io.BytesIO(contents)).convert("RGB")
        st = analyze_visual_style(img_full)
    except Exception as e:
        st = {"ai_points": 0, "real_points": 0, "signals": [f"[ERROR] Style loading: {str(e)[:50]}"]}

    ml["style"] = st
    if "signals" in ml and "signals" in st:
        ml["signals"].extend(st["signals"])

    res = calculate_final_score(fn, md, ml)

    return JSONResponse({
        "filename": file.filename,
        "verdict": res["verdict"],
        "ai_score": res["ai_score"],
        "real_score": res["real_score"],
        "confidence": res["confidence"],
        "color": res["color"],
        "layers": {"filename": fn, "metadata": md, "models": ml},
        "breakdown": res["breakdown"],
        "summary": res["summary"],
        "timestamp": datetime.now().isoformat()
    })
