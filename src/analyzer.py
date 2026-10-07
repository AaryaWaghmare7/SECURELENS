"""Structured forensic evidence with no model downloads or framework imports."""

import cv2
import numpy as np

from .ela import perform_ela
from .fft import perform_fft
from .image_utils import ImageAnalysisError, load_image

# Retained from the original CLI scorer, with its gain-20 ELA convention.
# These are exploratory heuristics, never calibrated ML probabilities.
HEURISTIC_THRESHOLDS = {
    "ela_mean_below": 8.0,
    "frequency_mean_above": 120.0,
    "ela_weight": 2,
    "frequency_weight": 1,
    "high_score_at": 3,
    "moderate_score_at": 1,
}


def calculate_image_statistics(image):
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    probabilities = hist[hist > 0] / gray.size
    channel_means = rgb.mean(axis=(0, 1))
    # Keep the original 16-pixel patch sampling convention.
    patch_std = [float(gray[y:y + 16, x:x + 16].std())
                 for y in range(0, gray.shape[0] - 16, 16)
                 for x in range(0, gray.shape[1] - 16, 16)]
    return {
        "brightness": float(gray.mean()), "mean": float(gray.mean()),
        "std": float(gray.std()), "edge_density": float(np.count_nonzero(edges) / edges.size),
        "noise": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "entropy": float(-np.sum(probabilities * np.log2(probabilities))),
        "texture_mean": float(np.mean(patch_std)) if patch_std else 0.0,
        "texture_std": float(np.std(patch_std)) if patch_std else 0.0,
        "channel_balance": float(channel_means.std()),
        "color_channels": {name: {"mean": float(rgb[:, :, i].mean()), "std": float(rgb[:, :, i].std())}
                           for i, name in enumerate(("R", "G", "B"))},
    }


def assess_signals(ela_mean, frequency_mean):
    thresholds = HEURISTIC_THRESHOLDS
    score = 0
    reasons = []
    if ela_mean < thresholds["ela_mean_below"]:
        score += thresholds["ela_weight"]
        reasons.append("Low recompression response matched the legacy ELA heuristic; smooth real images can also trigger it.")
    if frequency_mean > thresholds["frequency_mean_above"]:
        score += thresholds["frequency_weight"]
        reasons.append("Frequency mean matched the legacy FFT heuristic; resolution and texture affect this metric.")
    if score >= thresholds["high_score_at"]:
        level = "HIGH"
    elif score >= thresholds["moderate_score_at"]:
        level = "MODERATE"
    else:
        level = "LOW"
    return {
        "level": level, "score": score,
        "max_score": thresholds["ela_weight"] + thresholds["frequency_weight"],
        "summary": f"{level.title()} anomaly level",
        "reasons": reasons or ["Neither legacy ELA nor FFT heuristic was triggered."],
        "method": "Unvalidated heuristic categories; not an AI/real classification or confidence probability.",
        "thresholds": dict(thresholds),
    }


def analyze_image(source):
    """Combine metadata, pixel statistics, ELA, FFT, and measured JPEG loss."""
    try:
        loaded = load_image(source)
        image = loaded.image
        stats = calculate_image_statistics(image)
        ela = perform_ela(image)
        fft = perform_fft(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY))
        compression = []
        for quality in (90, 50, 20):
            response = ela if quality == 90 else perform_ela(image, quality=quality)
            compression.append({"quality": quality, "mean_abs_difference": response["raw_mean"],
                                "std_abs_difference": response["raw_std"]})
        return {
            **loaded.metadata, **stats,
            "ela_mean": ela["mean"], "ela_std": ela["std"],
            "ela_raw_mean": ela["raw_mean"], "ela_raw_std": ela["raw_std"],
            "frequency_mean": fft["mean"], "frequency_std": fft["std"],
            "fft_mean": fft["mean"], "fft_std": fft["std"],
            "high_frequency_ratio": fft["high_frequency_ratio"], "spectral_centroid": fft["spectral_centroid"],
            "exif": "Has EXIF" if loaded.metadata["has_exif"] else "No EXIF",
            "assessment": assess_signals(ela["mean"], fft["mean"]),
            "compression": compression, "notes": loaded.notes,
            "visualizations": {"original": image, "ela": ela["image"], "fft": fft["image"]},
        }
    except ImageAnalysisError:
        raise
    except (cv2.error, OSError, ValueError, MemoryError) as error:
        raise ImageAnalysisError("Analysis could not finish. Try a smaller image exported as JPG or PNG.") from error


def report_metrics(result):
    """JSON-friendly report seam for exports and future classifiers."""
    return {key: value for key, value in result.items() if key != "visualizations"}
