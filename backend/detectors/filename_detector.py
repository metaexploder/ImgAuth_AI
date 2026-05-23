import re
import os

STRONG_AI_KEYWORDS = {
    "midjourney": 70, "dall-e": 70, "dalle": 70, "stable-diffusion": 70,
    "stablediffusion": 70, "sdxl": 70, "sd3": 70, "sd35": 70,
    "firefly": 70, "ideogram": 70, "imagen": 70, "kling": 70,
    "sora": 70, "runway": 70, "pika": 70,
    "chatgpt": 70, "openai": 70, "gpt4": 70, "gpt-4": 70, "gpt4o": 70,
    "gemini": 70, "claude": 70, "perplexity": 70, "copilot": 70,
    "leonardo": 70, "nanobanana": 70, "nano-banana": 70, "flux": 70,
    "pixart": 70, "playground": 70, "kolors": 70, "hunyuan": 70,
    "cogview": 70, "deepfloyd": 70, "kandinsky": 70, "wuerstchen": 70,
    "animagine": 70, "waifu": 60,
    "comfyui": 70, "automatic1111": 70, "a1111": 70, "fooocus": 70,
    "invokeai": 70, "forge": 60,
    "txt2img": 70, "img2img": 70, "t2i": 60, "i2i": 60,
}

MEDIUM_AI_KEYWORDS = {
    "generated": 35, "ai-generated": 35, "aigenerated": 35,
    "fake": 25, "synthetic": 25, "deepfake": 30,
    "aiimage": 30, "ai_image": 30, "ai-image": 30,
    "upscaled": 15, "inpainted": 20, "outpainted": 20,
    "dreamshaper": 25, "juggernaut": 25, "realisticvision": 25,
    "deepdream": 20, "artbreeder": 20, "nightcafe": 20,
    "craiyon": 20, "dreamstudio": 25, "prompthero": 20,
    "civitai": 25, "tensor-art": 20, "tensorart": 20,
    "diffusion": 20, "lora": 15, "checkpoint": 15,
}

REAL_KEYWORDS = {
    "photo": 8, "raw": 8, "dsc": 10, "dscn": 10, "dscf": 10,
    "img_": 8, "img-": 8, "canon": 10, "nikon": 10, "sony": 10,
    "iphone": 10, "samsung": 8, "pixel": 10, "oneplus": 8,
    "screenshot": 5, "scan": 8, "pxl_": 10,
    "snap": 6, "dcim": 10, "camera": 8,
    "fujifilm": 10, "olympus": 10, "panasonic": 10, "leica": 10,
    "gopro": 10, "dji_": 10, "mavic": 10,
}


def analyze_filename(filename: str) -> dict:
    name = filename.lower()
    name_no_ext = os.path.splitext(name)[0]

    strong_hits, medium_hits, real_hits = [], [], []
    strong_ai, medium_ai, real_pts = 0, 0, 0
    signals = []

    for kw, pts in STRONG_AI_KEYWORDS.items():
        if kw in name_no_ext:
            strong_ai = max(strong_ai, pts)
            strong_hits.append(kw)
            signals.append(f"[AI] Strong AI keyword: '{kw}'")

    for kw, pts in MEDIUM_AI_KEYWORDS.items():
        if kw in name_no_ext:
            medium_ai = max(medium_ai, pts)
            medium_hits.append(kw)
            signals.append(f"[INFO] Medium AI keyword: '{kw}'")

    for kw, pts in REAL_KEYWORDS.items():
        if kw in name_no_ext:
            real_pts += pts
            real_hits.append(kw)
            signals.append(f"[REAL] Camera keyword: '{kw}' (+{pts} Real pts)")

    if re.search(r'[a-f0-9]{16,}', name_no_ext):
        medium_ai = max(medium_ai, 30)
        signals.append("[INFO] Long hex string pattern (common AI export naming)")

    if re.search(r'[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[a-f0-9]{4}-[a-f0-9]{12}', name_no_ext):
        medium_ai = max(medium_ai, 25)
        signals.append("[INFO] UUID v4 pattern detected (common AI platform export)")

    if re.search(r'(seed|cfg|step|sampler|guidance|scheduler)[_\-]?\d+', name_no_ext):
        strong_ai = max(strong_ai, 70)
        signals.append("[AI] Generation parameter pattern (seed/cfg/step)")

    if re.search(r'^(0{2,}\d+[\-_]|grid[\-_]|tmp[\-_])', name_no_ext):
        medium_ai = max(medium_ai, 25)
        signals.append("[INFO] WebUI output naming pattern (sequential numbering)")

    if re.search(r'(1024x1024|512x512|768x768|1024x768|1216x832|832x1216)', name_no_ext):
        medium_ai = max(medium_ai, 20)
        signals.append("[INFO] Common AI resolution in filename")

    if strong_ai >= 60:
        bucket = "strong_ai_trigger"
        ai_pts = min(strong_ai, 70)
        real_pts = min(real_pts, 10)
        priority_note = "Filename flagged — but pixel analysis will confirm or override."
    elif medium_ai >= 25:
        bucket = "medium_ai_suspicion"
        ai_pts = min(medium_ai, 50)
        real_pts = min(real_pts, 15)
        priority_note = "Suspicious filename — models and forensics will verify."
    else:
        bucket = "neutral_or_real_hint"
        ai_pts = 0
        real_pts = max(real_pts, 5)
        priority_note = "No suspicious filename signals."

    if signals:
        signals.append("[NOTE] Filename can be easily changed — this layer has low reliability.")

    return {
        "bucket": bucket,
        "ai_points": ai_pts,
        "real_points": min(real_pts, 25),
        "strong_hits": strong_hits,
        "medium_hits": medium_hits,
        "real_hits": real_hits,
        "signals": signals if signals else ["No suspicious filename patterns detected."],
        "priority_note": priority_note,
    }
