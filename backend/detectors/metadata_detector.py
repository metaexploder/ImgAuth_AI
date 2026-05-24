import io


def analyze_metadata(image_bytes: bytes) -> dict:
    ai_pts, real_pts = 0, 0
    signals = []

    try:
        from PIL import Image
        from PIL.ExifTags import TAGS

        img = Image.open(io.BytesIO(image_bytes))

        exif_data = {}
        raw = None
        try:
            raw = img._getexif()
            if raw:
                exif_data = {
                    TAGS.get(k, k): v
                    for k, v in raw.items()
                    if isinstance(v, (str, int, float, bytes))
                }
        except Exception:
            pass

        camera_fields_found = 0
        for field in ["Make", "Model", "LensMake", "LensModel",
                      "FocalLength", "ExposureTime", "FNumber", "ISOSpeedRatings"]:
            if field in exif_data:
                camera_fields_found += 1

        if camera_fields_found >= 3:
            real_pts += 25
            signals.append(f"[REAL] {camera_fields_found} camera EXIF fields found (+25 Real)")
        elif camera_fields_found >= 1:
            real_pts += 10
            signals.append(f"[REAL] {camera_fields_found} camera EXIF field(s) found (+10 Real)")

        gps_tags = [k for k in (exif_data or {}) if isinstance(k, str) and "GPS" in k]
        if not gps_tags and raw:
            if 0x8825 in raw or 34853 in raw:
                gps_tags = ["GPSInfo"]
        if gps_tags:
            real_pts += 15
            signals.append("[REAL] GPS geolocation data present (+15 Real)")

        software = str(exif_data.get("Software", "")).lower()
        ai_software_keywords = [
            "stable diffusion", "midjourney", "comfyui", "diffusers",
            "drawthings", "invoke", "fooocus", "flux", "dall-e", "dalle",
            "nanobanana", "nano banana", "pixart", "playground",
            "automatic1111", "a1111", "novelai", "tensorrt",
            "chatgpt", "gemini", "copilot",
        ]
        for kw in ai_software_keywords:
            if kw in software:
                ai_pts += 35
                signals.append(f"[AI] EXIF Software = AI tool '{kw}' (+35 AI)")
                break

        photo_editors = ["photoshop", "lightroom", "gimp", "capture one",
                         "darktable", "rawtherapee", "snapseed"]
        for ed in photo_editors:
            if ed in software:
                real_pts += 5
                signals.append(f"[REAL] Photo editor '{ed}' in Software tag (+5 Real)")
                break

        if img.format == "PNG":
            png_keys = [str(k).lower() for k in img.info.keys()]
            ai_png_keys = ["prompt", "parameters", "workflow", "invokeai",
                           "comfy", "negative_prompt", "steps", "sampler",
                           "cfg_scale", "seed", "model", "generation_data"]
            for k in ai_png_keys:
                if any(k in pk for pk in png_keys):
                    ai_pts += 35
                    signals.append(f"[AI] PNG metadata key '{k}' found (AI generator output) (+35 AI)")
                    break

            for key, value in img.info.items():
                if isinstance(value, str) and len(value) > 50:
                    val_lower = value.lower()
                    if any(w in val_lower for w in ["negative prompt", "cfg scale",
                                                     "seed:", "steps:", "sampler"]):
                        ai_pts += 30
                        signals.append("[AI] PNG text contains generation parameters (+30 AI)")
                        break

        if img.format in ("JPEG", "JPG") and not exif_data:
            ai_pts += 5
            signals.append("[INFO] JPEG has no EXIF — weak signal, common in shared photos (+5 AI)")

        if img.format in ("JPEG", "JPG") and not exif_data and not raw:
            try:
                data = image_bytes
                if data[0:2] == b'\xff\xd8':
                    has_app1 = b'\xff\xe1' in data[:20]
                    if not has_app1:
                        ai_pts += 3
                        signals.append("[INFO] JPEG missing APP1 marker — metadata may have been stripped (+3 AI)")
            except Exception:
                pass

        xmp = img.info.get("xmp", b"")
        if isinstance(xmp, bytes):
            xmp = xmp.decode("utf-8", errors="ignore")

        if xmp:
            if "c2pa" in xmp.lower() or "contentcredentials" in xmp.lower():
                real_pts += 20
                signals.append("[REAL] C2PA content credentials found (+20 Real)")

            if "ai-generated" in xmp.lower() or "ai_generated" in xmp.lower():
                ai_pts += 25
                signals.append("[AI] XMP marks image as AI-generated (+25 AI)")

            if "photoshop" in xmp.lower() or "lightroom" in xmp.lower():
                real_pts += 5
                signals.append("[REAL] Adobe editing history in XMP (+5 Real)")

        w, h = img.size
        common_ai_sizes = [
            (512, 512), (768, 768), (1024, 1024), (2048, 2048),
            (1024, 768), (768, 1024),
            (1216, 832), (832, 1216),
            (1344, 768), (768, 1344),
            (1152, 896), (896, 1152),
            (1536, 1024), (1024, 1536),
            (1024, 576), (576, 1024),
        ]
        if (w, h) in common_ai_sizes:
            ai_pts += 8
            signals.append(f"[INFO] Resolution {w}x{h} is common AI generation size (+8 AI)")

        if w == h and w >= 512:
            ai_pts += 5
            signals.append(f"[INFO] Perfect square ({w}x{h}) — uncommon for cameras (+5 AI)")

        if img.format in ("JPEG", "JPG"):
            try:
                qtables = img.quantization
                if qtables:
                    avg_quant = sum(sum(t) for t in qtables.values()) / sum(len(t) for t in qtables.values())
                    if avg_quant < 3:
                        ai_pts += 5
                        signals.append("[INFO] Extremely high JPEG quality (typical of AI output) (+5 AI)")
            except Exception:
                pass

        if img.mode == "RGBA":
            if img.format in ("JPEG", "JPG"):
                signals.append("[INFO] Image has alpha channel in JPEG (unusual)")

    except Exception as e:
        signals.append(f"[ERROR] Metadata error: {str(e)[:60]}")

    return {
        "ai_points": min(ai_pts, 65),
        "real_points": min(real_pts, 45),
        "signals": signals or ["No strong metadata signals found."],
        "weight": 0.25,
    }
