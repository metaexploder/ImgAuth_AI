import io
import numpy as np
import cv2
from PIL import Image
from scipy.stats import kurtosis as sp_kurtosis

MODEL_CACHE = {}


def load_models():
    from transformers import pipeline
    import torch

    device_id = 0 if torch.cuda.is_available() else -1
    print(f"[ImgAuth] Using device: {'GPU (cuda:0)' if device_id == 0 else 'CPU'}")

    model_ids = [
        ("umm_maybe", "umm-maybe/AI-image-detector"),
        ("dima806",   "dima806/ai_vs_real_image_detection"),
        ("organika",  "Organika/sdxl-detector"),
    ]
    for key, model_id in model_ids:
        if key not in MODEL_CACHE:
            try:
                print(f"[ImgAuth] Loading model: {model_id}")
                MODEL_CACHE[key] = pipeline(
                    "image-classification", model=model_id, device=device_id, framework="pt"
                )
            except Exception as e:
                print(f"[ImgAuth] Failed to load {model_id}: {e}")
                MODEL_CACHE[key] = None
    return MODEL_CACHE


def run_model(pipe, img):
    try:
        preds = pipe(img)
        ai_s, real_s = 0.0, 0.0
        for p in preds:
            label = p["label"].lower()
            if any(k in label for k in ["ai", "fake", "artificial", "generated", "synthetic"]):
                ai_s = max(ai_s, p["score"])
            elif any(k in label for k in ["real", "human", "natural", "authentic"]):
                real_s = max(real_s, p["score"])
        if ai_s == 0 and real_s == 0 and len(preds) >= 2:
            ai_s = preds[0]["score"]
            real_s = preds[1]["score"]
        elif ai_s == 0 and real_s == 0:
            ai_s, real_s = 0.5, 0.5
        total = ai_s + real_s or 1
        return {
            "ai_prob": round(ai_s / total, 4),
            "real_prob": round(real_s / total, 4),
            "raw": preds,
        }
    except Exception as e:
        return {"ai_prob": 0.5, "real_prob": 0.5, "raw": [], "error": str(e)}


def run_model_multiscale(pipe, img_full):
    img_512 = img_full.copy()
    img_512.thumbnail((512, 512))
    result_512 = run_model(pipe, img_512)

    w, h = img_full.size
    import torch
    if w > 600 and h > 600 and torch.cuda.is_available():
        crop_size = min(w, h, 384)
        cx, cy = w // 2, h // 2
        half = crop_size // 2
        center_crop = img_full.crop((cx - half, cy - half, cx + half, cy + half))
        result_crop = run_model(pipe, center_crop)
        ai_prob = result_512["ai_prob"] * 0.65 + result_crop["ai_prob"] * 0.35
        real_prob = result_512["real_prob"] * 0.65 + result_crop["real_prob"] * 0.35
        return {
            "ai_prob": round(ai_prob, 4),
            "real_prob": round(real_prob, 4),
            "raw": result_512["raw"],
        }
    return result_512


def noise_kurtosis_analysis(img):
    try:
        gray = np.array(img.convert("L"), dtype=np.float64)
        residual = cv2.Laplacian(gray, cv2.CV_64F)
        flat = residual.flatten()
        flat = flat[np.abs(flat) > 0.5]
        if len(flat) < 100:
            return {"ai_prob": 0.5, "real_prob": 0.5, "kurtosis": 0.0,
                    "detail": "Insufficient noise data"}

        k = float(sp_kurtosis(flat, fisher=True))

        if k > 5.0:
            ai_prob = 0.12
        elif k > 2.5:
            ai_prob = 0.28
        elif k > 1.0:
            ai_prob = 0.42
        elif k > 0.0:
            ai_prob = 0.55
        elif k > -0.5:
            ai_prob = 0.65
        else:
            ai_prob = 0.78

        label = "leptokurtic (real-like)" if k > 1.5 else "platykurtic (AI-like)" if k < 0 else "borderline"
        return {
            "ai_prob": round(ai_prob, 4),
            "real_prob": round(1 - ai_prob, 4),
            "kurtosis": round(k, 3),
            "detail": f"Excess kurtosis={k:.3f} -> {label}",
        }
    except Exception as e:
        return {"ai_prob": 0.5, "real_prob": 0.5, "kurtosis": 0.0,
                "detail": f"Error: {str(e)[:60]}"}


import torch


def extract_vit_features_and_attentions(img):
    try:
        pipe = MODEL_CACHE.get("umm_maybe")
        if not pipe or not hasattr(pipe, "model") or not hasattr(pipe, "image_processor"):
            return None

        model = pipe.model
        processor = pipe.image_processor

        inputs = processor(images=img, return_tensors="pt")
        # Match device of inputs to device of model (CPU or GPU)
        model_device = next(model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True, output_hidden_states=True)

        return {
            "logits": outputs.logits,
            "attentions": outputs.attentions,
            "hidden_states": outputs.hidden_states
        }
    except Exception as e:
        print(f"[ImgAuth] Feature/Attention extraction error: {e}")
        return None


def deep_feature_inconsistency_analysis(vit_data):
    try:
        if not vit_data or "hidden_states" not in vit_data:
            return {"ai_prob": 0.5, "real_prob": 0.5, "variance": 0.0, "detail": "ViT data unavailable", "cos_dist": None, "grid_w": 0, "grid_h": 0, "patch_features": None}

        last_hidden = vit_data["hidden_states"][-1][0].cpu()
        N = last_hidden.shape[0]

        import math
        root_N = int(round(math.sqrt(N)))
        if root_N * root_N == N:
            patch_features = last_hidden
            grid_w = grid_h = root_N
        else:
            root_N_minus_1 = int(round(math.sqrt(N - 1)))
            if root_N_minus_1 * root_N_minus_1 == N - 1:
                patch_features = last_hidden[1:, :]
                grid_w = grid_h = root_N_minus_1
            else:
                patch_features = last_hidden[1:, :]
                N_patches = N - 1
                grid_w = int(math.sqrt(N_patches))
                grid_h = N_patches // grid_w
                patch_features = patch_features[:grid_w * grid_h, :]

        mean_feat = torch.mean(patch_features, dim=0, keepdim=True)
        cos_sim = torch.nn.functional.cosine_similarity(patch_features, mean_feat, dim=1)
        cos_dist = 1.0 - cos_sim

        dist_variance = float(torch.var(cos_dist).item())

        if dist_variance > 0.0035:
            ai_prob = 0.76
            detail = f"High deep feature inconsistency (variance={dist_variance:.6f})"
        elif dist_variance > 0.0018:
            ai_prob = 0.62
            detail = f"Moderate deep feature inconsistency (variance={dist_variance:.6f})"
        elif dist_variance > 0.0006:
            ai_prob = 0.44
            detail = f"Normal feature consistency (variance={dist_variance:.6f})"
        else:
            ai_prob = 0.28
            detail = f"High feature uniformity (variance={dist_variance:.6f})"

        return {
            "ai_prob": round(ai_prob, 4),
            "real_prob": round(1.0 - ai_prob, 4),
            "variance": round(dist_variance, 6),
            "cos_dist": cos_dist,
            "grid_w": grid_w,
            "grid_h": grid_h,
            "patch_features": patch_features,
            "detail": detail
        }
    except Exception as e:
        return {"ai_prob": 0.5, "real_prob": 0.5, "variance": 0.0, "detail": f"Error: {str(e)[:60]}", "cos_dist": None, "grid_w": 0, "grid_h": 0, "patch_features": None}


def _encode_overlay_to_base64(img_bgr):
    """Encode a BGR numpy array to a Base64 JPEG data URL string."""
    success, buffer = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        return None
    import base64
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def generate_heatmap_overlay(img, vit_data, cos_dist_tensor, grid_w, grid_h, patch_features):
    """Generate attention and DFI heatmap overlays entirely in memory.
    Returns Base64-encoded JPEG data URL strings (no files written to disk).
    """
    try:
        w, h = img.size
        img_np = np.array(img.convert("RGB"))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        att_b64 = None
        dfi_b64 = None

        # ── Attention Heatmap ────────────────────────────────────────────────
        attentions = vit_data.get("attentions") if vit_data else None
        att_grid = None
        if attentions:
            try:
                last_layer_att = attentions[-1][0].cpu()
                if last_layer_att.ndim == 3:
                    avg_att = torch.mean(last_layer_att, dim=0)
                    if avg_att.shape[0] == patch_features.shape[0]:
                        cls_attention = avg_att.mean(dim=0)
                    else:
                        cls_attention = avg_att[0, 1:]
                    if cls_attention.numel() == grid_w * grid_h:
                        att_grid = cls_attention.reshape(grid_w, grid_h).numpy()
            except Exception as e:
                print(f"[ImgAuth] Attention extraction error: {e}")

        # Fallback to feature norm if attention parsing fails
        if att_grid is None and patch_features is not None and grid_w > 0 and grid_h > 0:
            try:
                norm_feat = torch.norm(patch_features, dim=1)
                att_grid = norm_feat.reshape(grid_w, grid_h).numpy()
            except Exception as e:
                print(f"[ImgAuth] Feature norm fallback error: {e}")

        if att_grid is not None:
            g_min, g_max = att_grid.min(), att_grid.max()
            att_grid = (att_grid - g_min) / (g_max - g_min + 1e-8)
            att_resized = cv2.resize((att_grid * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
            heatmap_att = cv2.applyColorMap(att_resized, cv2.COLORMAP_JET)
            overlay_att = cv2.addWeighted(img_bgr, 0.6, heatmap_att, 0.4, 0)
            att_b64 = _encode_overlay_to_base64(overlay_att)

        # ── DFI Heatmap ──────────────────────────────────────────────────────
        if cos_dist_tensor is not None and grid_w > 0 and grid_h > 0:
            dist_grid = cos_dist_tensor.reshape(grid_w, grid_h).numpy()
            g_min, g_max = dist_grid.min(), dist_grid.max()
            dist_grid = (dist_grid - g_min) / (g_max - g_min + 1e-8)
            dist_resized = cv2.resize((dist_grid * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
            heatmap_dfi = cv2.applyColorMap(dist_resized, cv2.COLORMAP_JET)
            overlay_dfi = cv2.addWeighted(img_bgr, 0.6, heatmap_dfi, 0.4, 0)
            dfi_b64 = _encode_overlay_to_base64(overlay_dfi)

        return att_b64, dfi_b64
    except Exception as e:
        print(f"[ImgAuth] Error generating heatmaps: {e}")
        return None, None


def fft_spectral_analysis(img):
    try:
        gray = np.array(img.convert("L"), dtype=np.float64)
        size = min(gray.shape[0], gray.shape[1], 512)
        gray = cv2.resize(gray, (size, size))

        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.log1p(np.abs(f_shift))

        center = size // 2
        mask = np.ones_like(magnitude, dtype=bool)
        mask[center - 5:center + 5, center - 5:center + 5] = False
        outer_mag = magnitude[mask]

        mean_mag = np.mean(outer_mag)
        std_mag = np.std(outer_mag)
        spike_threshold = mean_mag + 5.0 * std_mag
        spike_count = np.sum(magnitude[mask] > spike_threshold)
        spike_ratio = spike_count / len(outer_mag) if len(outer_mag) > 0 else 0
        sr = float(spike_ratio)

        if sr > 0.01:
            ai_prob = 0.72
            detail = f"Periodic artifacts detected (spike ratio={sr:.4f})"
        elif sr > 0.005:
            ai_prob = 0.58
            detail = f"Minor spectral anomalies ({sr:.4f})"
        elif sr > 0.002:
            ai_prob = 0.48
            detail = f"Faint spectral patterns ({sr:.4f})"
        else:
            ai_prob = 0.38
            detail = f"Clean spectrum (no periodic artifacts)"

        return {
            "ai_prob": round(ai_prob, 4),
            "real_prob": round(1 - ai_prob, 4),
            "spike_ratio": round(sr, 5),
            "detail": detail,
        }
    except Exception as e:
        return {"ai_prob": 0.5, "real_prob": 0.5, "spike_ratio": 0.0,
                "detail": f"Error: {str(e)[:60]}"}


def color_histogram_analysis(img):
    try:
        rgb = np.array(img.convert("RGB"))
        roughness_scores = []
        for channel in range(3):
            hist, _ = np.histogram(rgb[:, :, channel], bins=64, range=(0, 256))
            hist = hist.astype(np.float64)
            hist /= (hist.sum() + 1e-10)
            diffs = np.diff(hist)
            roughness_scores.append(float(np.std(diffs)))

        avg_roughness = np.mean(roughness_scores)

        if avg_roughness > 0.010:
            ai_prob = 0.25
            detail = f"Natural histogram roughness ({avg_roughness:.5f})"
        elif avg_roughness > 0.006:
            ai_prob = 0.38
            detail = f"Moderate histogram roughness ({avg_roughness:.5f})"
        elif avg_roughness > 0.003:
            ai_prob = 0.52
            detail = f"Smooth histogram ({avg_roughness:.5f}) — possibly synthetic"
        else:
            ai_prob = 0.68
            detail = f"Very smooth histogram ({avg_roughness:.5f}) — likely AI"

        return {
            "ai_prob": round(ai_prob, 4),
            "real_prob": round(1 - ai_prob, 4),
            "roughness": round(avg_roughness, 6),
            "detail": detail,
        }
    except Exception as e:
        return {"ai_prob": 0.5, "real_prob": 0.5, "roughness": 0.0,
                "detail": f"Error: {str(e)[:60]}"}


def jpeg_ghost_analysis(img):
    try:
        if img.format not in ("JPEG", "JPG", None):
            return {"ai_prob": 0.5, "real_prob": 0.5, "detail": "Not a JPEG"}

        rgb = np.array(img.convert("RGB"), dtype=np.float64)
        ghost_scores = []
        for q in [60, 70, 80]:
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=q)
            buf.seek(0)
            recomp = np.array(Image.open(buf).convert("RGB"), dtype=np.float64)
            diff = np.abs(rgb - recomp)
            ghost_scores.append(float(np.mean(diff)))

        min_ghost = min(ghost_scores)
        max_ghost = max(ghost_scores)
        spread = max_ghost - min_ghost

        if spread < 1.0:
            ai_prob = 0.58
            detail = f"Low ghost spread ({spread:.2f}) — uniform compression (possibly synthetic)"
        elif spread < 3.0:
            ai_prob = 0.45
            detail = f"Normal ghost spread ({spread:.2f})"
        else:
            ai_prob = 0.38
            detail = f"High ghost spread ({spread:.2f}) — natural re-compression history"

        return {
            "ai_prob": round(ai_prob, 4),
            "real_prob": round(1 - ai_prob, 4),
            "ghost_spread": round(spread, 3),
            "detail": detail,
        }
    except Exception as e:
        return {"ai_prob": 0.5, "real_prob": 0.5, "ghost_spread": 0.0,
                "detail": f"Error: {str(e)[:60]}"}


def analyze_image_models(image_bytes: bytes) -> dict:
    img_full = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    models = load_models()
    votes = []
    signals = []

    # 1. Run ViT Feature Extraction and Attentions
    vit_data = extract_vit_features_and_attentions(img_full)

    model_info = [
        ("umm_maybe", "umm-maybe ViT",          0.25),
        ("dima806",   "dima806 CNN",            0.25),
        ("organika",  "Organika SDXL-Detector",  0.25),
    ]
    active_dl_weight = 0.0
    for key, name, w in model_info:
        if models.get(key):
            r = run_model_multiscale(models[key], img_full)
            votes.append({
                "detector": name, "type": "deep_learning",
                "ai_prob": r["ai_prob"], "real_prob": r["real_prob"], "weight": w,
            })
            active_dl_weight += w
            signals.append(f"{name}: {int(r['ai_prob'] * 100)}% AI probability")
        else:
            signals.append(f"{name}: unavailable")

    if active_dl_weight == 0:
        pass
    else:
        for v in votes:
            if v["type"] == "deep_learning":
                v["weight"] = v["weight"] * (0.75 / active_dl_weight)

    # 2. Run Noise Kurtosis
    nk = noise_kurtosis_analysis(img_full)
    votes.append({
        "detector": "Noise Kurtosis Analysis", "type": "forensic",
        "ai_prob": nk["ai_prob"], "real_prob": nk["real_prob"], "weight": 0.08,
        "detail": nk.get("detail", ""), "kurtosis": nk.get("kurtosis", 0),
    })
    signals.append(f"Noise Kurtosis: {nk['detail']}")

    # 3. Run Deep Feature Inconsistency (DFI) instead of ELA
    dfi = deep_feature_inconsistency_analysis(vit_data)
    votes.append({
        "detector": "Deep Feature Inconsistency (DFI)", "type": "forensic",
        "ai_prob": dfi["ai_prob"], "real_prob": dfi["real_prob"], "weight": 0.07,
        "detail": dfi.get("detail", ""), "variance": dfi.get("variance", 0),
    })
    signals.append(f"DFI: {dfi['detail']}")

    # 4. Run FFT Spectral
    fft = fft_spectral_analysis(img_full)
    votes.append({
        "detector": "FFT Spectral Analysis", "type": "forensic",
        "ai_prob": fft["ai_prob"], "real_prob": fft["real_prob"], "weight": 0.05,
        "detail": fft.get("detail", ""), "spike_ratio": fft.get("spike_ratio", 0),
    })
    signals.append(f"FFT: {fft['detail']}")

    # 5. Run Color Histogram
    ch = color_histogram_analysis(img_full)
    votes.append({
        "detector": "Color Histogram Analysis", "type": "forensic",
        "ai_prob": ch["ai_prob"], "real_prob": ch["real_prob"], "weight": 0.03,
        "detail": ch.get("detail", ""), "roughness": ch.get("roughness", 0),
    })
    signals.append(f"Color Histogram: {ch['detail']}")

    # 6. Run JPEG Ghost
    jg = jpeg_ghost_analysis(img_full)
    votes.append({
        "detector": "JPEG Ghost Analysis", "type": "forensic",
        "ai_prob": jg["ai_prob"], "real_prob": jg["real_prob"], "weight": 0.02,
        "detail": jg.get("detail", ""), "ghost_spread": jg.get("ghost_spread", 0),
    })
    signals.append(f"JPEG Ghost: {jg['detail']}")

    # Generate explainability overlays (in-memory Base64, no disk writes)
    att_path, dfi_path = generate_heatmap_overlay(
        img_full,
        vit_data,
        dfi.get("cos_dist"),
        dfi.get("grid_w", 0),
        dfi.get("grid_h", 0),
        dfi.get("patch_features"),
    )

    # Delete non-serializable PyTorch tensors from returned dictionary
    for k in ["cos_dist", "patch_features"]:
        if k in dfi:
            del dfi[k]

    total_w = sum(v["weight"] for v in votes) or 1
    w_ai = sum(v["ai_prob"] * v["weight"] for v in votes) / total_w
    w_real = sum(v["real_prob"] * v["weight"] for v in votes) / total_w
    tot = w_ai + w_real or 1
    w_ai /= tot
    w_real /= tot

    dl_pct = int(min(active_dl_weight / total_w * 100, 100))
    forensic_pct = 100 - dl_pct

    return {
        "ai_points": int(w_ai * 100),
        "real_points": int(w_real * 100),
        "weighted_ai_prob": round(w_ai, 4),
        "votes": votes,
        "signals": signals,
        "forensics": {
            "kurtosis": nk, "dfi": dfi, "fft": fft,
            "color_histogram": ch, "jpeg_ghost": jg,
        },
        "attention_heatmap": att_path,
        "dfi_heatmap": dfi_path,
        "priority_note": f"DL-dominant ensemble: 3 models ({dl_pct}%) + 5 forensics ({forensic_pct}%).",
    }



