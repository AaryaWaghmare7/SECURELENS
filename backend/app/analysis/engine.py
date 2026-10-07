"""Adapter for the unchanged ELA/FFT engine, with the original comparison formulas."""
from io import BytesIO
import base64

import cv2
import numpy as np

from src.analyzer import analyze_image, report_metrics

DISCLAIMER = ("SecureLens currently combines digital-image forensic signals. Results are indicators, "
              "not definitive proof of AI generation or manipulation. No validated classifier is integrated.")


def classify(metrics):
    level = metrics["assessment"]["level"]
    thresholds = metrics["assessment"]["thresholds"]
    ela_flag = metrics["ela_mean"] < thresholds["ela_mean_below"]
    fft_flag = metrics["frequency_mean"] > thresholds["frequency_mean_above"]
    labels = {"LOW": "Likely Authentic", "MODERATE": "Inconclusive", "HIGH": "Potential AI-Generated"}
    explanations = [
        "Low recompression response triggered the ELA heuristic; smooth genuine photos can also trigger it."
        if ela_flag else "Recompression response did not trigger the low-ELA rule.",
        "Frequency mean crossed the legacy threshold; resolution and texture also affect this value."
        if fft_flag else "Frequency mean did not cross the legacy threshold.",
    ]
    return {"label": labels[level], "ai_indicators": level,
            # There is no validated localization/manipulation decision rule in this engine.
            "manipulation_indicators": "NOT_ESTABLISHED", "source": "Forensic heuristics",
            "probability": None, "explanations": explanations, "limitations": DISCLAIMER}


def image_data(image):
    preview = image.copy()
    preview.thumbnail((720, 720))
    with BytesIO() as buffer:
        preview.save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def run_analysis(data, filename):
    result = analyze_image(data)
    metrics = report_metrics(result)
    return {"filename": filename, "status": "complete", "metrics": metrics,
            "classification": classify(metrics),
            "visualizations": {key: image_data(image) for key, image in result["visualizations"].items()}}


def compare_results(a, b):
    # Same resized histogram/edge/pixel comparison as dashboard/views.py.
    arrays = []
    from PIL import Image
    for item in (a, b):
        encoded = item["visualizations"]["original"].split(",", 1)[1]
        with Image.open(BytesIO(base64.b64decode(encoded))) as image:
            arrays.append(np.asarray(image.convert("RGB").resize((256, 256))))
    gray = [cv2.cvtColor(array, cv2.COLOR_RGB2GRAY) for array in arrays]
    hist = [cv2.calcHist([image], [0], None, [64], [0, 256]) for image in gray]
    for histogram in hist:
        cv2.normalize(histogram, histogram)
    corr = float(cv2.compareHist(hist[0], hist[1], cv2.HISTCMP_CORREL))
    histogram_similarity = max(0.0, min(100.0, (corr + 1) * 50))
    edges = [float(np.count_nonzero(cv2.Canny(image, 80, 180)) / image.size) for image in gray]
    edge_similarity = max(0.0, 100 - abs(edges[0] - edges[1]) * 1000)
    delta = float(np.abs(arrays[0].astype(np.float32) - arrays[1].astype(np.float32)).mean())
    pixel_similarity = max(0.0, 100 - delta / 255 * 100)
    return {"similarity": round(histogram_similarity * .4 + edge_similarity * .25 + pixel_similarity * .35, 2),
            "histogram_similarity": round(histogram_similarity, 2),
            "edge_similarity": round(edge_similarity, 2), "pixel_similarity": round(pixel_similarity, 2),
            "histograms": [histogram.ravel().tolist() for histogram in hist],
            "method": "Resized histogram, edge density and pixel comparison; not authenticity confidence."}
