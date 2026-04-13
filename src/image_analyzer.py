import cv2
import numpy as np
import os
import sys
from PIL import Image, ImageChops, ImageEnhance
import io

# ─────────────────────────────────────────────────
# PATH FIX — works regardless of where you run from
# ─────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# NEW — points directly to SecureLens/data/
DATA_DIR = os.path.join(BASE_DIR, "..", "data")


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
# FFT — Frequency Analysis
# ─────────────────────────────────────────────────
def compute_fft(gray):
    try:
        fft = np.fft.fftshift(np.fft.fft2(gray))
        magnitude = 20 * np.log(np.abs(fft) + 1)
        return np.mean(magnitude), np.std(magnitude)
    except:
        return 0, 0


# ─────────────────────────────────────────────────
# Color Channel Analysis
# ─────────────────────────────────────────────────
def compute_color_stats(img):
    try:
        b, g, r = cv2.split(img)
        stats = {}
        for name, channel in zip(["R", "G", "B"], [r, g, b]):
            stats[f"{name}_mean"] = np.mean(channel)
            stats[f"{name}_std"] = np.std(channel)
        stats["channel_balance"] = np.std(
            [stats["R_mean"], stats["G_mean"], stats["B_mean"]]
        )
        return stats
    except:
        return {"channel_balance": 0}


# ─────────────────────────────────────────────────
# EXIF Metadata Check — fixed for PNG support
# ─────────────────────────────────────────────────
def check_exif(image_path):
    try:
        img = Image.open(image_path)
        # PNG files don't have _getexif() — this was crashing before
        if hasattr(img, "_getexif") and img._getexif() is not None:
            return "Has EXIF"
        else:
            return "No EXIF"
    except:
        return "No EXIF"


# ─────────────────────────────────────────────────
# Texture Analysis
# ─────────────────────────────────────────────────
def compute_texture_score(gray):
    try:
        local_std = []
        step = 16
        for y in range(0, gray.shape[0] - step, step):
            for x in range(0, gray.shape[1] - step, step):
                patch = gray[y : y + step, x : x + step]
                local_std.append(np.std(patch))
        if not local_std:
            return 0, 0
        return np.mean(local_std), np.std(local_std)
    except:
        return 0, 0


# ─────────────────────────────────────────────────
# Master Analyze Function
# ─────────────────────────────────────────────────
def analyze_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"   ⚠️  cv2 could not read image: {image_path}")
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        mean_val = np.mean(gray)
        std_val = np.std(gray)

        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size

        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        ela_mean, _ = compute_ela(image_path)
        fft_mean, _ = compute_fft(gray)
        color_stats = compute_color_stats(img)
        texture_mean, _ = compute_texture_score(gray)
        exif_status = check_exif(image_path)

        return {
            "mean": mean_val,
            "std": std_val,
            "edge_density": edge_density,
            "noise": laplacian_var,
            "ela_mean": ela_mean,
            "fft_mean": fft_mean,
            "texture_mean": texture_mean,
            "channel_balance": color_stats.get("channel_balance", 0),
            "exif": exif_status,
        }

    except Exception as e:
        print(f"   ❌ Error analyzing image: {e}")
        return None


# ─────────────────────────────────────────────────
# Scoring System
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
        reasons.append("Uniform texture")

    if f["channel_balance"] < 5:
        score += 1
        reasons.append("Balanced RGB channels")

    if f["exif"] == "No EXIF":
        score += 1
        reasons.append("No EXIF metadata")

    if f["fft_mean"] > 120:
        score += 1
        reasons.append("Unusual frequency pattern")

    confidence = round((score / 10) * 100)
    label = "AI Generated" if score >= 5 else "Real"

    return label, confidence, reasons


# ─────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────
print("\n" + "=" * 50)
print("   🔍 SecureLens — Image Forensics Engine")
print("=" * 50)

# Check data folder exists
if not os.path.exists(DATA_DIR):
    print(f"\n❌ 'data' folder not found at: {DATA_DIR}")
    print("👉 Create a folder named 'data' and put your images in it.")
    sys.exit()

# Get valid image files
all_files = os.listdir(DATA_DIR)
image_files = [f for f in all_files if f.lower().endswith((".jpg", ".jpeg", ".png"))]

if not image_files:
    print(f"\n❌ No images found in: {DATA_DIR}")
    print("👉 Add .jpg or .png images to the data folder.")
    sys.exit()

print(f"\n📁 Found {len(image_files)} image(s) in: {DATA_DIR}\n")

# Process each image
for file in sorted(image_files):
    path = os.path.join(DATA_DIR, file)
    features = analyze_image(path)

    if features is None:
        print(f"⛔ Skipped: {file}\n")
        continue

    label, confidence, reasons = score_image(features)
    icon = "🤖" if label == "AI Generated" else "✅"

    print(f"📸 {file}")
    print(f"   Mean Pixel    : {features['mean']:.2f}")
    print(f"   Std Deviation : {features['std']:.2f}")
    print(f"   Edge Density  : {features['edge_density']:.4f}")
    print(f"   Noise         : {features['noise']:.2f}")
    print(f"   ELA Mean      : {features['ela_mean']:.2f}")
    print(f"   FFT Mean      : {features['fft_mean']:.2f}")
    print(f"   Texture       : {features['texture_mean']:.2f}")
    print(f"   RGB Balance   : {features['channel_balance']:.2f}")
    print(f"   EXIF          : {features['exif']}")
    print(f"   {'─'*35}")
    print(f"   {icon} {label}  |  Confidence: {confidence}%")

    if reasons:
        print(f"   📋 Reasons:")
        for r in reasons:
            print(f"      • {r}")
    print()

print("=" * 50)
print("✅ SecureLens scan complete.")
print("=" * 50)
