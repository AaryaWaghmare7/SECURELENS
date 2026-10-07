"""Frequency features with the existing 20 * log(|FFT| + 1) convention."""

import numpy as np
from PIL import Image


def perform_fft(gray):
    gray = np.asarray(gray)
    if gray.ndim != 2 or gray.size == 0:
        raise ValueError("FFT needs a nonempty grayscale image.")
    magnitude = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    spectrum = 20 * np.log(magnitude + 1)
    low, high = float(spectrum.min()), float(spectrum.max())
    normalized = (spectrum - low) / (high - low) if high > low else np.zeros_like(spectrum)
    height, width = gray.shape
    y, x = np.ogrid[:height, :width]
    # Preserve the Django frequency-feature radius convention for comparability.
    radius = np.sqrt((x - width / 2.0) ** 2 + (y - height / 2.0) ** 2)
    radius /= max(float(radius.max()), 1.0)
    energy = float(magnitude.sum())
    return {
        "mean": float(spectrum.mean()), "std": float(spectrum.std()),
        "high_frequency_ratio": float(magnitude[radius > 0.35].sum() / energy) if energy else 0.0,
        "spectral_centroid": float((radius * magnitude).sum() / energy) if energy else 0.0,
        "image": Image.fromarray((normalized * 255).astype(np.uint8)),
    }
