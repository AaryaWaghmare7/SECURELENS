from io import BytesIO
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from PIL import Image
from streamlit.testing.v1 import AppTest

from src.analyzer import analyze_image, assess_signals, report_metrics
from streamlit_app import download_report, result_presentation

ROOT = Path(__file__).resolve().parents[1]


class StreamlitTests(unittest.TestCase):
    def test_empty_app_renders_without_errors(self):
        app = AppTest.from_file(str(ROOT / "streamlit_app.py")).run(timeout=20)
        self.assertFalse(app.exception)
        self.assertIn("SecureLens", app.title[0].value)
        self.assertTrue(app.info)

    def test_upload_analysis_renders_results(self):
        buffer = BytesIO()
        Image.new("RGB", (40, 30), "teal").save(buffer, format="PNG")
        with patch("streamlit.file_uploader", return_value=buffer):
            app = AppTest.from_file(str(ROOT / "streamlit_app.py")).run(timeout=20)
            app.button[0].click().run(timeout=20)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 4)
        self.assertIn("Analysis results", [heading.value for heading in app.subheader])
        self.assertIn("analysis", app.session_state)

    def test_bad_upload_shows_helpful_error(self):
        with patch("streamlit.file_uploader", return_value=BytesIO(b"invalid")):
            app = AppTest.from_file(str(ROOT / "streamlit_app.py")).run(timeout=20)
        self.assertFalse(app.exception)
        self.assertTrue(app.error)
        self.assertIn("decoded", app.error[0].value)

    def test_new_upload_does_not_display_previous_result(self):
        first, second = BytesIO(), BytesIO()
        Image.new("RGB", (40, 30), "teal").save(first, format="PNG")
        Image.new("RGB", (40, 30), "red").save(second, format="PNG")
        with patch("streamlit.file_uploader", return_value=first):
            app = AppTest.from_file(str(ROOT / "streamlit_app.py")).run(timeout=20)
            app.button[0].click().run(timeout=20)
        with patch("streamlit.file_uploader", return_value=second):
            app.run(timeout=20)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 0)

    def test_product_results_follow_existing_heuristic_rules(self):
        cases = [(8, 120, "LOW", "Likely Real Image"),
                 (7, 120, "MODERATE", "Inconclusive"),
                 (8, 121, "MODERATE", "Inconclusive"),
                 (7, 121, "HIGH", "Potential AI-Generated Image")]
        for ela_mean, fft_mean, level, label in cases:
            with self.subTest(level=level, ela=ela_mean, fft=fft_mean):
                presentation = result_presentation({"assessment": assess_signals(ela_mean, fft_mean)})
                self.assertEqual(presentation["label"], label)
                self.assertEqual(presentation["indicator_level"], level)
                self.assertEqual(presentation["source"], "Forensic heuristics")
                self.assertIsNone(presentation["model_probability"])

    def test_report_keeps_backend_values_and_displayed_result(self):
        buffer = BytesIO()
        Image.new("RGB", (40, 30), "teal").save(buffer, format="PNG")
        result = analyze_image(buffer.getvalue())
        report = json.loads(download_report(result))
        for key, value in report_metrics(result).items():
            self.assertEqual(report[key], value)
        self.assertEqual(report["display_result"], result_presentation(result))
        self.assertNotIn("visualizations", report)

    def test_results_prioritize_verdict_and_keep_details_collapsed(self):
        buffer = BytesIO()
        Image.new("RGB", (40, 30), "teal").save(buffer, format="PNG")
        with patch("streamlit.file_uploader", return_value=buffer):
            app = AppTest.from_file(str(ROOT / "streamlit_app.py")).run(timeout=20)
            app.button[0].click().run(timeout=20)
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        markup = "\n".join(item.value for item in app.markdown)
        self.assertIn("IMAGE AUTHENTICITY RESULT", markup)
        self.assertIn("HEURISTIC ASSESSMENT", markup)
        self.assertLess(markup.index("IMAGE AUTHENTICITY RESULT"), markup.index("SUPPORTING EVIDENCE"))
        self.assertEqual(app.expander[0].label, "View Advanced Forensic Analysis")
        self.assertFalse(app.expander[0].proto.expanded)
        self.assertEqual(len(app.get("download_button")), 1)


if __name__ == "__main__":
    unittest.main()
