import cv2
import numpy as np
import os
from PIL import Image, ImageChops, ImageEnhance
import io

DATA_DIR = "data"

# ─────────────────────────────────────────────────
# ELA — Error Level Analysis
# ─────────────────────────────────────────────────
def compute_ela(image_path, quality=90):
    try:
        original = Image.open(image_path).convert("RGB")
        buffer = io.BytesIO()
        original.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer).convert("RGB")
        ela = ImageChops.difference(original, compressed)
        ela = ImageEnhance.Brightness(ela).enhance(20)
        ela_arr = np.array(ela)
        return np.mean(ela_arr), np.std(ela_arr)
    except:
        return 0, 0

# ─────────────────────────────────────────────────
# FFT — Frequency Domain Analysis
# ─────────────────────────────────────────────────
def compute_fft(gray):
    fft = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = 20 * np.log(np.abs(fft) + 1)
    return np.mean(magnitude), np.std(magnitude)

# ─────────────────────────────────────────────────
# Color Channel Analysis
# ─────────────────────────────────────────────────
def compute_color_stats(img):
    b, g, r = cv2.split(img)
    stats = {}
    for name, channel in zip(["R", "G", "B"], [r, g, b]):
        stats[f"{name}_mean"] = np.mean(channel)
        stats[f"{name}_std"]  = np.std(channel)
    # AI images tend to have very balanced RGB channels (too perfect)
    channel_diff = np.std([stats["R_mean"], stats["G_mean"], stats["B_mean"]])
    stats["channel_balance"] = channel_diff
    return stats

# ─────────────────────────────────────────────────
# EXIF Metadata Check
# ─────────────────────────────────────────────────
def check_exif(image_path):
    try:
        exif = Image.open(image_path)._getexif()
        return "Has EXIF" if exif else "No EXIF"
    except:
        return "No EXIF"

# ─────────────────────────────────────────────────
# Texture — Local Binary Pattern (simple version)
# ─────────────────────────────────────────────────
def compute_texture_score(gray):
    # Measures local variation — AI images are often too uniform
    local_std = []
    step = 16
    for y in range(0, gray.shape[0] - step, step):
        for x in range(0, gray.shape[1] - step, step):
            patch = gray[y:y+step, x:x+step]
            local_std.append(np.std(patch))
    return np.mean(local_std), np.std(local_std)

# ─────────────────────────────────────────────────
# Master Analyze Function
# ─────────────────────────────────────────────────
def analyze_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Basic stats
    mean_val        = np.mean(gray)
    std_val         = np.std(gray)

    # Edge detection
    edges           = cv2.Canny(gray, 100, 200)
    edge_density    = np.sum(edges > 0) / edges.size

    # Noise
    laplacian_var   = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Advanced features
    ela_mean, ela_std           = compute_ela(image_path)
    fft_mean, fft_std           = compute_fft(gray)
    color_stats                 = compute_color_stats(img)
    texture_mean, texture_std   = compute_texture_score(gray)
    exif_status                 = check_exif(image_path)

    return {
        "mean":            mean_val,
        "std":             std_val,
        "edge_density":    edge_density,
        "noise":           laplacian_var,
        "ela_mean":        ela_mean,
        "fft_mean":        fft_mean,
        "texture_mean":    texture_mean,
        "channel_balance": color_stats["channel_balance"],
        "exif":            exif_status,
        "color_stats":     color_stats
    }

# ─────────────────────────────────────────────────
# Scoring Engine — Weighted Decision
# ─────────────────────────────────────────────────
def score_image(f):
    score = 0
    reasons = []

    if f["noise"] < 80:
        score += 2
        reasons.append("Low noise (too smooth)")

    if f["edge_density"] < 0.05:
        score += 2
        reasons.append("Low edge density")

    if f["ela_mean"] < 8:
        score += 2
        reasons.append("Low ELA response")

    if f["texture_mean"] < 20:
        score += 1
        reasons.append("Uniform texture (low local variation)")

    if f["channel_balance"] < 5:
        score += 1
        reasons.append("Suspiciously balanced RGB channels")

    if f["exif"] == "No EXIF":
        score += 1
        reasons.append("No camera metadata (EXIF missing)")

    if f["fft_mean"] > 120:
        score += 1
        reasons.append("Unusual frequency pattern (FFT)")

    max_score   = 10
    confidence  = round((score / max_score) * 100)
    label       = "AI Generated" if score >= 5 else "Real / Natural"

    return label, confidence, reasons

# ─────────────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────────────
print("\n" + "="*55)
print("        🔍 SecureLens — Image Forensics Engine")
print("="*55)

for file in sorted(os.listdir(DATA_DIR)):
    if not file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    path     = os.path.join(DATA_DIR, file)
    features = analyze_image(path)

    if features is None:
        print(f"⛔ Could not read: {file}\n")
        continue

    label, confidence, reasons = score_image(features)
    icon = "🤖" if "AI" in label else "✅"

    print(f"\n📸 {file}")
    print(f"   Mean Pixel      : {features['mean']:.2f}")
    print(f"   Std Deviation   : {features['std']:.2f}")
    print(f"   Edge Density    : {features['edge_density']:.4f}")
    print(f"   Noise Estimate  : {features['noise']:.2f}")
    print(f"   ELA Mean        : {features['ela_mean']:.2f}")
    print(f"   FFT Mean        : {features['fft_mean']:.2f}")
    print(f"   Texture Mean    : {features['texture_mean']:.2f}")
    print(f"   RGB Balance     : {features['channel_balance']:.2f}")
    print(f"   EXIF Status     : {features['exif']}")
    print(f"   ── Verdict ──────────────────────────")
    print(f"   {icon}  {label}  |  Confidence: {confidence}%")
    if reasons:
        print(f"   📋 Reasons:")
        for r in reasons:
            print(f"      • {r}")
    print()

print("="*55)
print("✅ SecureLens scan complete.")
print("="*55)
```