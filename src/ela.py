"""JPEG recompression error analysis using the original Pillow calculation."""

from io import BytesIO

import numpy as np
from PIL import Image, ImageChops, ImageEnhance


def perform_ela(image, quality=90, gain=20):
    """Return raw error and the legacy gain-enhanced ELA metrics/visualization.

    Gain 20 preserves src/image_analyzer.py; Django historically uses gain 15.
    Recompression lives in memory and leaves no temporary files on disk.
    """
    if not 1 <= quality <= 100 or gain <= 0:
        raise ValueError("JPEG quality must be 1-100 and ELA gain must be positive.")
    original = image.convert("RGB")
    with BytesIO() as buffer:
        original.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            difference = ImageChops.difference(original, compressed.convert("RGB"))
    raw = np.asarray(difference)
    visualization = ImageEnhance.Brightness(difference).enhance(gain)
    enhanced = np.asarray(visualization)
    return {
        "mean": float(enhanced.mean()), "std": float(enhanced.std()),
        "raw_mean": float(raw.mean()), "raw_std": float(raw.std()),
        "quality": quality, "gain": gain, "image": visualization,
    }
