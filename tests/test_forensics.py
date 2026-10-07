import contextlib
from io import BytesIO, StringIO
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance

from src.analyzer import analyze_image, assess_signals, report_metrics
from src.ela import perform_ela
from src.fft import perform_fft
from src.image_utils import ImageAnalysisError, load_image


def encode(image, format="PNG"):
    buffer = BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


class ForensicTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.image = Image.fromarray(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8))

    def test_ela_matches_original_pillow_formula(self):
        buffer = BytesIO()
        self.image.save(buffer, format="JPEG", quality=90)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            legacy = np.asarray(ImageEnhance.Brightness(
                ImageChops.difference(self.image, compressed.convert("RGB"))).enhance(20))
        result = perform_ela(self.image)
        self.assertAlmostEqual(result["mean"], float(legacy.mean()))
        self.assertAlmostEqual(result["std"], float(legacy.std()))

    def test_fft_matches_original_numpy_formula(self):
        gray = cv2.cvtColor(np.asarray(self.image), cv2.COLOR_RGB2GRAY)
        expected = 20 * np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1)
        result = perform_fft(gray)
        self.assertAlmostEqual(result["mean"], float(expected.mean()))
        self.assertAlmostEqual(result["std"], float(expected.std()))
        self.assertTrue(0 <= result["high_frequency_ratio"] <= 1)

    def test_grayscale_and_rgba_are_supported(self):
        gray = analyze_image(encode(Image.new("L", (30, 20), 100)))
        self.assertEqual(gray["channels"], 1)
        self.assertEqual(gray["brightness"], 100)
        rgba = analyze_image(encode(Image.new("RGBA", (30, 20), (10, 20, 30, 0))))
        self.assertEqual(rgba["channels"], 4)
        self.assertEqual(rgba["brightness"], 255)
        self.assertTrue(any("Transparency" in note for note in rgba["notes"]))

    def test_corrupt_empty_and_disguised_unsupported_uploads(self):
        for data in (b"not an image", b"", encode(self.image, "GIF"), encode(self.image)[:40]):
            with self.subTest(size=len(data)), self.assertRaises(ImageAnalysisError):
                analyze_image(data)

    def test_size_limits_before_decoding(self):
        with patch("src.image_utils.MAX_UPLOAD_BYTES", 10), self.assertRaises(ImageAnalysisError):
            load_image(encode(self.image))
        with patch("src.image_utils.MAX_INPUT_PIXELS", 10), self.assertRaises(ImageAnalysisError):
            load_image(encode(self.image))

    def test_missing_image_has_helpful_error(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ImageAnalysisError, "could not be read"):
            load_image(Path(directory) / "missing.png")

    def test_large_input_is_resized_and_reported(self):
        result = analyze_image(encode(Image.new("RGB", (1800, 900), "navy")))
        self.assertEqual((result["width"], result["height"]), (1800, 900))
        self.assertEqual((result["analysis_width"], result["analysis_height"]), (1600, 800))
        self.assertTrue(result["notes"])

    def test_exif_orientation_and_source_dimensions(self):
        image = Image.new("RGB", (60, 40), "green")
        exif = Image.Exif()
        exif[274] = 6
        buffer = BytesIO()
        image.save(buffer, format="JPEG", exif=exif)
        result = analyze_image(buffer.getvalue())
        self.assertEqual((result["width"], result["height"]), (60, 40))
        self.assertEqual((result["analysis_width"], result["analysis_height"]), (40, 60))

    def test_json_export_and_no_files_written(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.png"
            source.write_bytes(encode(self.image))
            result = analyze_image(source)
            self.assertEqual(list(Path(directory).iterdir()), [source])
        report = report_metrics(result)
        json.dumps(report, allow_nan=False)
        self.assertNotIn("visualizations", report)
        self.assertNotIn("confidence", report)
        self.assertEqual([row["quality"] for row in report["compression"]], [90, 50, 20])

    def test_heuristic_categories_and_boundaries(self):
        self.assertEqual(assess_signals(8, 120)["level"], "LOW")
        self.assertEqual(assess_signals(7, 120)["level"], "MODERATE")
        self.assertEqual(assess_signals(8, 121)["level"], "MODERATE")
        self.assertEqual(assess_signals(7, 121)["level"], "HIGH")

    def test_zero_fft_is_finite(self):
        result = perform_fft(np.zeros((1, 1), dtype=np.uint8))
        self.assertEqual(result["mean"], 0)
        self.assertEqual(result["high_frequency_ratio"], 0)

    def test_legacy_import_has_no_scanning_side_effects(self):
        output = StringIO()
        with contextlib.redirect_stdout(output):
            legacy = importlib.import_module("src.image_analyzer")
            importlib.reload(legacy)
        self.assertEqual(output.getvalue(), "")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.png"
            source.write_bytes(encode(self.image))
            old = legacy.analyze_image(str(source))
            new = analyze_image(source)
        for key in ("mean", "std", "edge_density", "noise", "ela_mean", "fft_mean", "texture_mean", "channel_balance"):
            self.assertAlmostEqual(old[key], new[key], places=6, msg=key)


if __name__ == "__main__":
    unittest.main()
