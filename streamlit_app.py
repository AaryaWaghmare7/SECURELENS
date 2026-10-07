"""SecureLens product interface; forensic calculations live in src/analyzer.py."""

from html import escape
import json
import logging
from pathlib import Path

import streamlit as st

from src.analyzer import analyze_image, report_metrics
from src.image_utils import ImageAnalysisError, load_image

LOGGER = logging.getLogger(__name__)
DISCLAIMER = ("SecureLens currently combines digital-image forensic signals. "
              "Results should be treated as indicators rather than definitive proof of AI generation.")


def result_presentation(result):
    """Translate existing heuristic categories into product labels, not probabilities."""
    level = result["assessment"]["level"]
    labels = {"LOW": "Likely Real Image", "MODERATE": "Inconclusive",
              "HIGH": "Potential AI-Generated Image"}
    descriptions = {
        "LOW": "The current ELA and frequency rules did not flag this image. This does not establish authenticity.",
        "MODERATE": "One of the two forensic rules was triggered. The available evidence does not settle AI versus real.",
        "HIGH": "Both forensic rules were triggered. Genuine images can also show these patterns; review the evidence below.",
    }
    return {"label": labels[level], "indicator_level": level,
            "description": descriptions[level], "source": "Forensic heuristics",
            "model_probability": None}


def download_report(result):
    report = report_metrics(result)
    report["display_result"] = result_presentation(result)
    return json.dumps(report, indent=2, allow_nan=False)


def evidence_card(label, value, detail):
    st.markdown(f'<div class="evidence-card"><div class="eyebrow">{escape(label)}</div>'
                f'<h3>{escape(value)}</h3><p>{escape(detail)}</p></div>', unsafe_allow_html=True)


def display_results(result):
    assessment = result["assessment"]
    presentation = result_presentation(result)
    level = presentation["indicator_level"]
    st.subheader("Analysis results")
    st.markdown(f"""<section class="verdict verdict-{level.lower()}" aria-label="Image authenticity result">
      <div class="verdict-top"><span class="eyebrow">IMAGE AUTHENTICITY RESULT</span>
        <span class="method-badge">HEURISTIC ASSESSMENT</span></div>
      <h2>{escape(presentation['label'])}</h2>
      <div class="indicator-badge"><span class="status-dot"></span>AI-generation indicators: {level.title()}</div>
      <p>{escape(presentation['description'])}</p>
      <div class="verdict-foot">Based on forensic heuristics until a validated trained classifier is integrated.
        No model confidence probability is available.</div>
    </section>""", unsafe_allow_html=True)
    for note in result["notes"]:
        st.caption(note)
    st.markdown('<div class="section-kicker">SUPPORTING EVIDENCE</div>', unsafe_allow_html=True)
    st.subheader("Look beneath the surface")
    st.caption("The image and its signal views support the result above. Bright areas are not proof of manipulation.")
    for column, key, title, description in zip(
        st.columns(3), ("original", "ela", "fft"),
        ("01 / Original image", "02 / ELA visualization", "03 / Frequency spectrum"),
        ("Your image at the processing resolution.",
         "JPEG recompression differences, enhanced for visibility.",
         "Log-magnitude FFT: texture and spatial frequency patterns."),
    ):
        with column, st.container(border=True):
            st.markdown(f"**{title}**")
            st.image(result["visualizations"][key], width="stretch")
            st.caption(description)
    st.subheader("Technical indicators")
    thresholds = assessment["thresholds"]
    ela_flag = result["ela_mean"] < thresholds["ela_mean_below"]
    fft_flag = result["frequency_mean"] > thresholds["frequency_mean_above"]
    ela_column, fft_column = st.columns(2)
    with ela_column:
        evidence_card("ELA anomaly rule", "Triggered" if ela_flag else "Not triggered",
                      "Low recompression response matched the legacy rule." if ela_flag else
                      "Recompression response did not match the legacy rule.")
    with fft_column:
        evidence_card("Frequency anomaly rule", "Triggered" if fft_flag else "Not triggered",
                      "Frequency mean exceeded the legacy threshold." if fft_flag else
                      "Frequency mean did not exceed the legacy threshold.")
    for column, label, value in zip(st.columns(4),
        ("Resolution", "Source channels", "Brightness / 255", "JPEG quality 90 loss"),
        (f'{result["width"]} x {result["height"]}', result["channels"],
         f'{result["brightness"]:.1f}', f'{result["ela_raw_mean"]:.2f} / 255')):
        column.metric(label, value)
    st.caption("JPEG loss is the measured mean absolute pixel difference after recompression, not the original compression quality.")
    with st.container(border=True):
        st.subheader("Why did SecureLens reach this result?")
        if ela_flag:
            st.write("• The image changed very little during JPEG recompression. This triggered the existing low-ELA rule; smooth genuine photos can trigger it too.")
        else:
            st.write("• JPEG recompression response stayed outside the existing low-ELA flag.")
        if fft_flag:
            st.write("• The measured frequency mean crossed the existing threshold. Image size and texture also affect this value; there is no validated natural-image baseline here.")
        else:
            st.write("• The frequency mean stayed below or at the existing threshold.")
        st.caption("Brightness, channels, and compression measurements provide context. They do not contribute extra points to this result.")
    with st.expander("View Advanced Forensic Analysis"):
        st.write("Exact values for research and reproducibility")
        st.dataframe([
            {"Metric": "ELA mean (gain 20)", "Value": result["ela_mean"]},
            {"Metric": "ELA standard deviation (gain 20)", "Value": result["ela_std"]},
            {"Metric": "Raw JPEG recompression mean error", "Value": result["ela_raw_mean"]},
            {"Metric": "Raw JPEG recompression standard deviation", "Value": result["ela_raw_std"]},
            {"Metric": "FFT mean (20 x log magnitude)", "Value": result["frequency_mean"]},
            {"Metric": "FFT standard deviation", "Value": result["frequency_std"]},
            {"Metric": "High frequency magnitude ratio", "Value": result["high_frequency_ratio"]},
            {"Metric": "Spectral centroid", "Value": result["spectral_centroid"]},
            {"Metric": "Entropy (bits)", "Value": result["entropy"]},
            {"Metric": "Edge density (fraction)", "Value": result["edge_density"]},
            {"Metric": "Laplacian variance", "Value": result["noise"]},
        ], hide_index=True, width="stretch")
        st.markdown("**Image metadata**")
        st.json({key: result[key] for key in ("width", "height", "channels", "mode", "format",
                 "file_size_bytes", "has_exif", "analysis_width", "analysis_height", "analysis_channels")})
        st.markdown("**JPEG compression experiment**")
        st.dataframe(result["compression"], hide_index=True, width="stretch")
        st.markdown("**Heuristic rules**")
        st.json(thresholds)
        st.write(f'{assessment["score"]}/{assessment["max_score"]} heuristic points. '
                 "0 = Low, 1–2 = Moderate, 3 = High. These cutoffs are not validated model boundaries.")
    st.download_button("Download analysis JSON", download_report(result),
                       file_name="securelens_analysis.json", mime="application/json", icon=":material/download:")
    st.caption(DISCLAIMER)


def main():
    st.set_page_config(page_title="SecureLens | AI Image Authenticity Detection", page_icon="S", layout="wide")
    css = (Path(__file__).resolve().parent / "assets" / "streamlit.css").read_text()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    st.markdown('''<div class="brand-bar"><div class="brand-lockup"><span class="brand-mark">SL</span>
      <span>SECURELENS <span class="brand-divider">/</span> IMAGE INTELLIGENCE</span></div>
      <span class="privacy-chip"><span class="status-dot"></span>SESSION-ONLY ANALYSIS</span></div>''', unsafe_allow_html=True)
    with st.container(key="hero"):
        text, artwork = st.columns([3, 2], vertical_alignment="center")
        with text:
            st.markdown('<span class="hero-badge">AI Forensics • ELA • Frequency Analysis</span>', unsafe_allow_html=True)
            st.title("SecureLens")
            st.markdown('<h2 class="hero-heading">AI Image Authenticity<br><span>Detection</span></h2>', unsafe_allow_html=True)
            st.write("Analyze images for signs of AI generation and digital manipulation.")
            st.caption("Detect potential AI-generated and manipulated images using digital image forensics.")
        with artwork:
            st.markdown('''<div class="scanner" aria-hidden="true"><div class="scanner-grid"></div>
              <div class="scan-orbit orbit-one"></div><div class="scan-orbit orbit-two"></div>
              <div class="scan-core"><svg viewBox="0 0 80 80" fill="none"><path d="M40 8L66 19V37C66 54 54 65 40 72C26 65 14 54 14 37V19L40 8Z" stroke="currentColor" stroke-width="2"/><path d="M25 40L35 50L55 29" stroke="currentColor" stroke-width="3"/></svg></div>
              <div class="scan-beam"></div><span class="scan-label">SIGNALS INTO INSIGHT</span></div>''', unsafe_allow_html=True)
    st.markdown('''<div class="flow-strip"><span><b>01</b> Upload Image</span><i>→</i>
      <span><b>02</b> Analyze Image</span><i>→</i><span><b>03</b> Review Result</span></div>''', unsafe_allow_html=True)
    with st.container(key="upload_panel", border=True):
        st.subheader("Upload an image to analyze")
        st.caption("JPG • JPEG • PNG   |   Up to 16 MB / 40 megapixels")
        uploaded = st.file_uploader("Choose your image", type=["jpg", "jpeg", "png"],
                                   label_visibility="collapsed",
                                   help="Large images are resized for bounded processing. Uploads are not saved to disk.")
        if uploaded is None:
            st.info("Your image authenticity result will appear here after analysis.")
            st.caption(DISCLAIMER)
            return
        data = uploaded.getvalue()
        try:
            loaded = load_image(data)
            stored = st.session_state.get("analysis")
            has_result = stored is not None and stored[0] == data
            if not has_result:
                preview, action = st.columns([1, 2], vertical_alignment="center")
                with preview:
                    st.image(loaded.image, caption="Ready for analysis", width="stretch")
                with action:
                    st.markdown("**Your image is ready.**")
                    st.write("Review its authenticity indicators, then explore the supporting evidence.")
                    clicked = st.button("Analyze Image", type="primary", width="stretch")
            else:
                clicked = st.button("Analyze Image", type="primary")
            if clicked:
                with st.spinner("Analyzing authenticity indicators..."):
                    result = analyze_image(data)
                st.session_state["analysis"] = (data, result)
                st.rerun()
        except ImageAnalysisError as error:
            st.error(str(error))
            return
        except Exception:
            LOGGER.exception("Unexpected image analysis failure")
            st.error("Unable to process this image. Try a smaller JPG or PNG, or upload another file.")
            return
    stored = st.session_state.get("analysis")
    if stored is not None and stored[0] == data:
        try:
            display_results(stored[1])
        except Exception:
            LOGGER.exception("Unexpected result rendering failure")
            st.error("Unable to display these results. Please analyze the image again.")


if __name__ == "__main__":
    main()
