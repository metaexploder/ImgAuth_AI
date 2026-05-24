import cv2
import numpy as np
from PIL import Image

def analyze_visual_style(img: Image.Image) -> dict:
    signals = []
    ai_pts = 0
    real_pts = 0
    correlation = 0.5
    lap_var = 100.0

    try:
        # Convert PIL Image to grayscale numpy array
        img_rgb = np.array(img.convert("RGB"))
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # 1. Symmetry Analysis
        # AI images (especially portraits or generated objects) often exhibit unnatural bilateral symmetry
        h, w = gray.shape
        size = min(h, w, 256)
        resized = cv2.resize(gray, (size, size))
        flipped = cv2.flip(resized, 1)  # Horizontal flip

        # Calculate Pearson correlation coefficient between original and flipped halves, checking std first
        std_val = np.std(resized)
        if std_val > 1e-4:
            corr_matrix = np.corrcoef(resized.flat, flipped.flat)
            if corr_matrix.shape == (2, 2):
                correlation = float(corr_matrix[0, 1])
                if np.isnan(correlation):
                    correlation = 0.5

        if correlation > 0.94:
            ai_pts += 15
            signals.append(f"[STYLE] Unnaturally high horizontal symmetry (corr={correlation:.3f}) (+15 AI)")
        elif correlation > 0.88:
            ai_pts += 5
            signals.append(f"[STYLE] Moderate horizontal symmetry (corr={correlation:.3f}) (+5 AI)")
        elif correlation < 0.35:
            real_pts += 10
            signals.append(f"[STYLE] Natural asymmetry (corr={correlation:.3f}) (+10 Real)")

        # 2. Over-smoothing / Blur Detection
        # AI textures are often unnaturally smooth or lack fine, high-frequency camera noise
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        if lap_var < 50.0:
            ai_pts += 12
            signals.append(f"[STYLE] Extremely smooth texture / low local contrast (var={lap_var:.2f}) (+12 AI)")
        elif lap_var < 150.0:
            ai_pts += 6
            signals.append(f"[STYLE] Soft textures / over-smoothing (var={lap_var:.2f}) (+6 AI)")
        elif lap_var > 600.0:
            real_pts += 10
            signals.append(f"[STYLE] High-frequency natural textures (var={lap_var:.2f}) (+10 Real)")

    except Exception as e:
        signals.append(f"[ERROR] Style analysis error: {str(e)[:60]}")

    return {
        "ai_points": min(ai_pts, 30),
        "real_points": min(real_pts, 20),
        "signals": signals if signals else ["No style anomalies detected."],
        "symmetry_correlation": round(correlation, 4),
        "texture_variance": round(lap_var, 2),
    }
