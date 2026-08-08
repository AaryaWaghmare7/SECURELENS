import os
import io
import json
import base64
import tempfile
from uuid import uuid4
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps
from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import LoginView, LogoutView
from django.db.models import Avg
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render

from .forms import ImageUploadForm, LoginForm, RegisterForm
from .models import ImageAnalysis

DEFAULT_MODELS = [
    ("haywoodsloan/autotrain-ai-vs-real-image-classifier", 1.2),
    ("umm-maybe/AI-image-detector", 1.0),
    ("Organika/sdxl-detector", 0.6),
]
AI_KEYWORDS = ('ai', 'fake', 'artificial', 'generated', 'synthetic', 'sdxl', 'midjourney')
REAL_KEYWORDS = ('real', 'human', 'authentic', 'natural', 'photo', 'photograph')
UNKNOWN_KEYWORDS = ('label_', 'unknown', 'other', 'fake_or_real', 'generated_or_real')


def build_model_specs():
    configured = os.getenv("SECURELENS_MODELS", "").strip()
    if not configured:
        return DEFAULT_MODELS

    specs = []
    for raw_item in configured.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if ":" in item:
            name, raw_weight = item.split(":", 1)
            try:
                specs.append((name.strip(), float(raw_weight.strip())))
                continue
            except ValueError:
                pass
        specs.append((item, 1.0))
    return specs or DEFAULT_MODELS


def summarize_model_results(results):
    if not results:
        return {
            'ai_score': 0.0,
            'real_score': 0.0,
            'unknown_score': 0.0,
            'top_score': 0.0,
            'top_label': 'unknown',
        }

    ai_score = max(
        (result['score'] for result in results if any(word in result['label'].lower() for word in AI_KEYWORDS)),
        default=0.0,
    )
    real_score = max(
        (result['score'] for result in results if any(word in result['label'].lower() for word in REAL_KEYWORDS)),
        default=0.0,
    )
    top = max(results, key=lambda item: item['score'])
    top_label = top['label'].lower()
    unknown_score = 0.0

    if ai_score == 0.0 and real_score == 0.0:
        if any(word in top_label for word in AI_KEYWORDS):
            ai_score = top['score']
        elif any(word in top_label for word in REAL_KEYWORDS):
            real_score = top['score']
        else:
            unknown_score = top['score']
    elif any(word in top_label for word in UNKNOWN_KEYWORDS):
        unknown_score = top['score']

    return {
        'ai_score': ai_score,
        'real_score': real_score,
        'unknown_score': unknown_score,
        'top_score': top['score'],
        'top_label': top['label'],
    }


def build_analysis_views(image_path):
    base = Image.open(image_path).convert("RGB")
    views = [
        ("original", base),
        ("autocontrast", ImageOps.autocontrast(base)),
        ("sharpened", base.filter(ImageFilter.SHARPEN)),
    ]
    return views


def compute_edge_density(img):
    if img is None:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    return float(np.count_nonzero(edges) / edges.size)


def compute_ela_score(image_path, quality=90):
    try:
        original = Image.open(image_path).convert("RGB")
        buffer = io.BytesIO()
        original.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer).convert("RGB")
        ela = ImageChops.difference(original, compressed)
        ela = ImageEnhance.Brightness(ela).enhance(15)
        ela_arr = np.array(ela)
        return float(np.mean(ela_arr)), float(np.std(ela_arr))
    except Exception:
        return 0.0, 0.0


def compute_entropy(img):
    if img is None:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist = hist / max(hist.sum(), 1.0)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def compute_fft_preview(img, size=64):
    if img is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fft = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.log(np.abs(fft) + 1.0)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    preview = cv2.resize(magnitude.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    return np.round(preview, 2).tolist()


def compute_frequency_features(img):
    if img is None:
        return {
            'high_frequency_ratio': 0.0,
            'spectral_centroid': 0.0,
        }

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    fft = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.abs(fft)
    height, width = magnitude.shape
    y_grid, x_grid = np.indices((height, width))
    center_y, center_x = height / 2.0, width / 2.0
    radius = np.sqrt((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2)
    max_radius = max(float(radius.max()), 1.0)
    normalized_radius = radius / max_radius
    total_energy = float(np.sum(magnitude)) or 1.0
    high_frequency_ratio = float(np.sum(magnitude[normalized_radius > 0.35]) / total_energy)
    spectral_centroid = float(np.sum(normalized_radius * magnitude) / total_energy)
    return {
        'high_frequency_ratio': high_frequency_ratio,
        'spectral_centroid': spectral_centroid,
    }


def compute_compression_profile(image_path):
    profile = {}
    try:
        original = Image.open(image_path).convert("RGB")
        original_arr = np.array(original).astype(np.float32)
        original_gray = cv2.cvtColor(original_arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        original_edges = cv2.Canny(original_gray, 80, 180)
        quality_losses = []

        for quality in (90, 70, 50, 20):
            buffer = io.BytesIO()
            original.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            compressed = Image.open(buffer).convert("RGB")
            compressed_arr = np.array(compressed).astype(np.float32)
            diff = np.abs(original_arr - compressed_arr)
            compressed_gray = cv2.cvtColor(compressed_arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            compressed_edges = cv2.Canny(compressed_gray, 80, 180)
            edge_delta = np.count_nonzero(original_edges != compressed_edges) / max(original_edges.size, 1)
            loss = {
                'mean_abs_diff': float(np.mean(diff)),
                'std_abs_diff': float(np.std(diff)),
                'max_abs_diff': float(np.max(diff)),
                'edge_delta': float(edge_delta),
            }
            profile[f'jpeg_q{quality}'] = loss
            quality_losses.append(loss)

        q90 = profile['jpeg_q90']
        q50 = profile['jpeg_q50']
        q20 = profile['jpeg_q20']
        compression_artifact_score = max(
            0.0,
            min(
                100.0,
                (q90['mean_abs_diff'] * 3.2)
                + (q50['mean_abs_diff'] * 1.15)
                + (q20['mean_abs_diff'] * 0.35)
                + (q50['edge_delta'] * 120.0),
            ),
        )
        profile['summary'] = {
            'compression_artifact_score': float(compression_artifact_score),
            'jpeg_q90_loss': q90['mean_abs_diff'],
            'jpeg_q50_loss': q50['mean_abs_diff'],
            'jpeg_q20_loss': q20['mean_abs_diff'],
            'edge_loss_q50': q50['edge_delta'],
        }
        return profile
    except Exception:
        return {
            'summary': {
                'compression_artifact_score': 0.0,
                'jpeg_q90_loss': 0.0,
                'jpeg_q50_loss': 0.0,
                'jpeg_q20_loss': 0.0,
                'edge_loss_q50': 0.0,
            }
        }


def compute_heatmap_preview(img, size=64):
    if img is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    magnitude = cv2.magnitude(grad_x, grad_y) + np.abs(laplacian)
    magnitude = cv2.GaussianBlur(magnitude, (5, 5), 0)
    normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    preview = cv2.resize(normalized.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    return np.round(preview, 2).tolist()


def compute_quality_metrics(img):
    if img is None:
        return {
            'sharpness': 0.0,
            'blur_score': 100.0,
            'blur_level': 'High',
            'brightness': 0.0,
            'contrast': 0.0,
            'quality_label': 'Unknown',
        }

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray) / 255.0 * 100.0)
    contrast = float(np.std(gray) / 255.0 * 100.0)
    blur_score = float(max(0.0, min(100.0, 100.0 - sharpness / 4.0)))

    if sharpness > 150 and contrast > 18:
        quality_label = 'Excellent'
    elif sharpness > 80 and contrast > 12:
        quality_label = 'Good'
    elif sharpness > 40:
        quality_label = 'Fair'
    else:
        quality_label = 'Soft'

    if blur_score < 35:
        blur_level = 'Low'
    elif blur_score < 70:
        blur_level = 'Medium'
    else:
        blur_level = 'High'

    return {
        'sharpness': sharpness,
        'blur_score': blur_score,
        'blur_level': blur_level,
        'brightness': brightness,
        'contrast': contrast,
        'quality_label': quality_label,
    }


def extract_image_metadata(image_path):
    metadata = {
        'resolution': 'Unknown',
        'file_size_kb': 0.0,
        'channels': 0,
        'format': 'Unknown',
        'created': 'Unavailable',
    }
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            metadata['resolution'] = f'{width} x {height}'
            metadata['channels'] = len(img.getbands())
            metadata['format'] = img.format or 'Unknown'
            exif = img.getexif()
            if exif:
                datetime_original = exif.get(36867) or exif.get(306)
                if datetime_original:
                    metadata['created'] = str(datetime_original)
    except Exception:
        pass

    try:
        metadata['file_size_kb'] = round(os.path.getsize(image_path) / 1024.0, 2)
    except Exception:
        metadata['file_size_kb'] = 0.0

    return metadata


def compare_image_files(path_a, path_b):
    with Image.open(path_a).convert("RGB") as image_a, Image.open(path_b).convert("RGB") as image_b:
        size = (256, 256)
        image_a = image_a.resize(size)
        image_b = image_b.resize(size)
        arr_a = np.array(image_a)
        arr_b = np.array(image_b)

    img_a = cv2.cvtColor(arr_a, cv2.COLOR_RGB2BGR)
    img_b = cv2.cvtColor(arr_b, cv2.COLOR_RGB2BGR)
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    hist_a = cv2.calcHist([gray_a], [0], None, [64], [0, 256])
    hist_b = cv2.calcHist([gray_b], [0], None, [64], [0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    histogram_corr = float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))
    histogram_similarity = max(0.0, min(100.0, ((histogram_corr + 1.0) / 2.0) * 100.0))

    edge_a = compute_edge_density(img_a)
    edge_b = compute_edge_density(img_b)
    edge_similarity = max(0.0, 100.0 - abs(edge_a - edge_b) * 1000.0)

    pixel_delta = float(np.mean(np.abs(arr_a.astype(np.float32) - arr_b.astype(np.float32))) / 255.0 * 100.0)
    pixel_similarity = max(0.0, 100.0 - pixel_delta)

    overall = round((histogram_similarity * 0.4) + (edge_similarity * 0.25) + (pixel_similarity * 0.35), 2)
    return {
        'similarity': overall,
        'histogram_similarity': round(histogram_similarity, 2),
        'edge_similarity': round(edge_similarity, 2),
        'pixel_similarity': round(pixel_similarity, 2),
        'edge_density_a': round(edge_a, 4),
        'edge_density_b': round(edge_b, 4),
        'hist_a': np.round(hist_a.ravel(), 4).tolist(),
        'hist_b': np.round(hist_b.ravel(), 4).tolist(),
        'image_a': path_a,
        'image_b': path_b,
    }


def apply_image_heuristics(ai_score, real_score, stats):
    edge_density = stats.get('edge_density', 0.0)
    std_pixel = stats.get('std_pixel', 0.0)
    ela_mean = stats.get('ela_mean', 0.0)
    texture_score = stats.get('texture_score', 0.0)
    entropy_score = stats.get('entropy', 0.0)
    capture_source = stats.get('capture_source', 'upload')
    quality_label = stats.get('quality_label', 'Unknown')
    blur_level = stats.get('blur_level', 'Unknown')
    high_frequency_ratio = stats.get('high_frequency_ratio', 0.0)
    spectral_centroid = stats.get('spectral_centroid', 0.0)
    compression_artifact_score = stats.get('compression_artifact_score', 0.0)

    if edge_density < 0.022 and std_pixel < 42:
        ai_score += 0.035
    elif edge_density > 0.075 and std_pixel > 55:
        real_score += 0.03

    if ela_mean < 18 and texture_score < 0.28:
        ai_score += 0.02
    elif ela_mean > 24 and texture_score > 0.34:
        real_score += 0.02

    if stats.get('contrast_score', 0.0) < 0.12 and stats.get('edge_density', 0.0) < 0.03:
        ai_score += 0.015
    elif stats.get('contrast_score', 0.0) > 0.22 and stats.get('edge_density', 0.0) > 0.06:
        real_score += 0.015

    if entropy_score < 5.4:
        ai_score += 0.012
    elif entropy_score > 6.5:
        real_score += 0.012

    if high_frequency_ratio < 0.18 and spectral_centroid < 0.20 and edge_density < 0.04:
        ai_score += 0.018
    elif high_frequency_ratio > 0.26 and edge_density > 0.045:
        real_score += 0.018

    if compression_artifact_score > 48 and std_pixel > 35:
        real_score += 0.01
    elif compression_artifact_score < 14 and entropy_score < 6.0:
        ai_score += 0.01

    if capture_source == 'webcam':
        real_score += 0.075
        if quality_label in ('Good', 'Excellent'):
            real_score += 0.015
        if blur_level in ('Low', 'Medium'):
            real_score += 0.01
        if std_pixel > 28:
            real_score += 0.012

    return ai_score, real_score


def live_capture_confidence(stats, avg_real=0.0, avg_ai=0.0):
    entropy_score = stats.get('entropy', 0.0)
    edge_density = stats.get('edge_density', 0.0)
    contrast_score = stats.get('contrast_score', 0.0)
    high_frequency_ratio = stats.get('high_frequency_ratio', 0.0)
    compression_artifact_score = stats.get('compression_artifact_score', 0.0)

    evidence = 58.0
    evidence += min(14.0, max(0.0, (entropy_score - 4.8) * 4.0))
    evidence += min(10.0, edge_density * 120.0)
    evidence += min(8.0, contrast_score * 24.0)
    evidence += min(8.0, high_frequency_ratio * 24.0)
    evidence += min(6.0, compression_artifact_score / 18.0)
    evidence += max(-10.0, min(10.0, (avg_real - avg_ai) * 20.0))
    return round(max(55.0, min(94.0, evidence)), 2)


def classify_prediction(model_runs, stats):
    if stats.get('capture_source') == 'webcam':
        if not model_runs:
            confidence = live_capture_confidence(stats)
            explanation = "This was captured live in SecureLens, so the source is trusted as real while confidence is based on image statistics."
            return 'REAL', confidence, explanation

    if not model_runs:
        return 'No Model', 0.0, "No detector could be loaded for this upload."

    weighted_ai = 0.0
    weighted_real = 0.0
    weighted_unknown = 0.0
    total_weight = 0.0

    for run in model_runs:
        weighted_ai += run.get('ai_score', 0.0) * run.get('weight', 1.0)
        weighted_real += run.get('real_score', 0.0) * run.get('weight', 1.0)
        weighted_unknown += run.get('unknown_score', 0.0) * run.get('weight', 1.0)
        total_weight += run.get('weight', 1.0)

    if total_weight == 0:
        return 'Error', 0.0, "The detector responses were empty."

    avg_ai = weighted_ai / total_weight
    avg_real = weighted_real / total_weight
    avg_unknown = weighted_unknown / total_weight
    avg_ai, avg_real = apply_image_heuristics(avg_ai, avg_real, stats)
    avg_unknown = min(1.0, avg_unknown + (0.02 if stats.get('texture_score', 0.0) > 0.38 else 0.0))

    if stats.get('capture_source') == 'webcam':
        confidence = live_capture_confidence(stats, avg_real=avg_real, avg_ai=avg_ai)
        explanation = (
            "This was captured live in SecureLens, so the source is treated as a real camera capture. "
            "The confidence is calculated from model votes plus entropy, edge density, FFT frequency energy, contrast, and compression loss."
        )
        return 'REAL', confidence, explanation

    confidence = round(max(avg_ai, avg_real, avg_unknown) * 100, 2)
    margin = abs(avg_ai - avg_real)
    strongest_signal = max(max(run.get('ai_score', 0.0), run.get('real_score', 0.0)) for run in model_runs)
    ai_votes = sum(1 for run in model_runs if run.get('ai_score', 0.0) > run.get('real_score', 0.0))
    real_votes = sum(1 for run in model_runs if run.get('real_score', 0.0) > run.get('ai_score', 0.0))
    vote_gap = abs(ai_votes - real_votes)

    if max(avg_ai, avg_real, avg_unknown) < 0.50:
        explanation = "The detector signals were too weak overall, so SecureLens marked this upload as uncertain."
        return 'UNCERTAIN', confidence, explanation

    if avg_unknown >= max(avg_ai, avg_real) and avg_unknown > 0.58:
        explanation = "The ensemble could not confidently separate the image from an unknown-looking pattern, so SecureLens stayed cautious."
        return 'UNCERTAIN', confidence, explanation

    if margin < 0.03 and strongest_signal < 0.76:
        explanation = "The detector votes were too close, so SecureLens marked this upload as uncertain instead of forcing a shaky answer."
        return 'UNCERTAIN', confidence, explanation

    if vote_gap == 0 and strongest_signal < 0.72:
        explanation = "The detector ensemble split too evenly and felt too close to call, so SecureLens left the verdict as uncertain."
        return 'UNCERTAIN', confidence, explanation

    if avg_ai > avg_real and avg_ai >= max(0.57, avg_real + 0.06):
        explanation = "Multiple detector signals leaned toward AI-generation more strongly than the real-photo signals."
        return 'AI', confidence, explanation

    explanation = "The combined model votes and image texture cues leaned toward a real photograph."
    return 'REAL', confidence, explanation


@lru_cache(maxsize=1)
def load_detectors():
    try:
        from transformers import pipeline
    except Exception as error:
        print(f"❌ Transformers import error: {error}")
        return []

    detectors = []
    for model_name, weight in build_model_specs():
        try:
            detector = pipeline("image-classification", model=model_name)
            detectors.append({
                'name': model_name,
                'weight': weight,
                'detector': detector,
            })
            print(f"✅ Model loaded: {model_name}")
        except Exception as error:
            print(f"❌ Model load error for {model_name}: {error}")
    return detectors


def collect_image_stats(img, image_path=None):
    mean_pixel = float(np.mean(img)) if img is not None else 0.0
    std_pixel = float(np.std(img)) if img is not None else 0.0
    edge_density = compute_edge_density(img) if img is not None else 0.0
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) if img is not None else None
    texture_score = float(np.std(lab[:, :, 0]) / 255.0) if lab is not None else 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img is not None else None
    contrast_score = float(np.std(gray) / 255.0) if gray is not None else 0.0
    entropy_score = compute_entropy(img) if img is not None else 0.0
    fft_preview = compute_fft_preview(img) if img is not None else []
    frequency = compute_frequency_features(img)
    ela_mean, ela_std = compute_ela_score(image_path) if image_path else (0.0, 0.0)
    compression_profile = compute_compression_profile(image_path) if image_path else {'summary': {}}
    compression_summary = compression_profile.get('summary', {})
    compression_artifact_score = float(compression_summary.get('compression_artifact_score', 0.0))
    if compression_artifact_score < 28:
        compression_label = 'Low'
    elif compression_artifact_score < 58:
        compression_label = 'Medium'
    else:
        compression_label = 'High'

    return {
        'mean_pixel': mean_pixel,
        'std_pixel': std_pixel,
        'edge_density': edge_density,
        'texture_score': texture_score,
        'contrast_score': contrast_score,
        'entropy': entropy_score,
        'fft_preview': fft_preview,
        'high_frequency_ratio': frequency.get('high_frequency_ratio', 0.0),
        'spectral_centroid': frequency.get('spectral_centroid', 0.0),
        'ela_mean': ela_mean,
        'ela_std': ela_std,
        'compression_artifact_score': compression_artifact_score,
        'compression_label': compression_label,
        'compression_profile': compression_profile,
        'jpeg_q90_loss': compression_summary.get('jpeg_q90_loss', 0.0),
        'jpeg_q50_loss': compression_summary.get('jpeg_q50_loss', 0.0),
        'jpeg_q20_loss': compression_summary.get('jpeg_q20_loss', 0.0),
        'edge_loss_q50': compression_summary.get('edge_loss_q50', 0.0),
    }


def landing_context(request):
    analyses = ImageAnalysis.objects.all().order_by('-uploaded_at')
    recent = analyses[:6]
    return {
        'analyses': recent,
        'total': analyses.count(),
        'real_count': analyses.filter(prediction='REAL').count(),
        'ai_count': analyses.filter(prediction='AI').count(),
        'uncertain_count': analyses.filter(prediction='UNCERTAIN').count(),
        'avg_confidence': analyses.aggregate(avg=Avg('confidence'))['avg'] or 0,
        'user_total': request.user.analyses.count() if request.user.is_authenticated else 0,
    }


def analyze_image_path(image_path, image_name='', capture_source=None):
    img = cv2.imread(image_path)
    stats = collect_image_stats(img, image_path)
    metadata = extract_image_metadata(image_path)
    quality = compute_quality_metrics(img)
    heatmap_preview = compute_heatmap_preview(img)
    inferred_source = 'webcam' if 'securelens-webcam' in image_name.lower() else 'upload'
    capture_source = capture_source or inferred_source
    stats['capture_source'] = capture_source
    stats['quality_label'] = quality.get('quality_label', 'Unknown')
    stats['blur_level'] = quality.get('blur_level', 'Unknown')

    model_runs = []
    views = build_analysis_views(image_path)
    for detector_info in load_detectors():
        view_summaries = []
        for view_name, view_img in views:
            results = detector_info['detector'](view_img)
            summary = summarize_model_results(results)
            summary['view_name'] = view_name
            view_summaries.append(summary)

        summary = {
            'ai_score': float(np.mean([item['ai_score'] for item in view_summaries])) if view_summaries else 0.0,
            'real_score': float(np.mean([item['real_score'] for item in view_summaries])) if view_summaries else 0.0,
            'unknown_score': float(np.mean([item['unknown_score'] for item in view_summaries])) if view_summaries else 0.0,
            'top_score': max((item['top_score'] for item in view_summaries), default=0.0),
            'top_label': max(view_summaries, key=lambda item: item['top_score'])['top_label'] if view_summaries else 'unknown',
            'views': view_summaries,
        }
        summary.update({
            'weight': detector_info['weight'],
            'model_name': detector_info['name'],
        })
        model_runs.append(summary)

    prediction, confidence, explanation = classify_prediction(model_runs, stats)
    analysis_meta = {
        'models': model_runs,
        'stats': stats,
        'metadata': metadata,
        'quality': quality,
        'heatmap_preview': heatmap_preview,
        'capture_source': capture_source,
        'explanation': explanation,
        'model_count': len(model_runs),
        'views': [name for name, _ in views],
        'fft_preview': stats.get('fft_preview', []),
    }
    return {
        'prediction': prediction,
        'confidence': confidence,
        'explanation': explanation,
        'stats': stats,
        'analysis_meta': analysis_meta,
    }


def run_analysis_pipeline(obj, capture_source=None):
    result = analyze_image_path(obj.image.path, image_name=obj.image.name, capture_source=capture_source)
    obj.mean_pixel = result['stats']['mean_pixel']
    obj.std_pixel = result['stats']['std_pixel']
    obj.prediction = result['prediction']
    obj.confidence = result['confidence']
    obj.analysis_meta = result['analysis_meta']
    explanation = result['explanation']
    return obj, explanation


def create_analysis_record(owner, uploaded_file, batch_id=None):
    obj = ImageAnalysis(owner=owner, batch_id=batch_id)
    obj.image.save(uploaded_file.name, uploaded_file, save=False)
    obj.save()
    return obj


def build_batch_result_item(obj, explanation):
    return {
        'obj': obj,
        'explanation': explanation,
        'heatmap_json': json.dumps((obj.analysis_meta or {}).get('heatmap_preview', [])),
        'fft_json': json.dumps((obj.analysis_meta or {}).get('fft_preview', [])),
    }


def compute_dataset_metrics(analyses):
    labeled = [item for item in analyses if item.ground_truth in ('AI', 'REAL') and item.prediction in ('AI', 'REAL')]
    total = len(labeled)
    if not total:
        return {
            'labeled_total': 0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'confusion': {
                'ai_ai': 0,
                'ai_real': 0,
                'real_ai': 0,
                'real_real': 0,
            },
        }

    ai_ai = sum(1 for item in labeled if item.ground_truth == 'AI' and item.prediction == 'AI')
    ai_real = sum(1 for item in labeled if item.ground_truth == 'REAL' and item.prediction == 'AI')
    real_ai = sum(1 for item in labeled if item.ground_truth == 'AI' and item.prediction == 'REAL')
    real_real = sum(1 for item in labeled if item.ground_truth == 'REAL' and item.prediction == 'REAL')

    precision_ai = ai_ai / max(ai_ai + ai_real, 1)
    recall_ai = ai_ai / max(ai_ai + real_ai, 1)
    precision_real = real_real / max(real_real + real_ai, 1)
    recall_real = real_real / max(real_real + ai_real, 1)
    macro_precision = (precision_ai + precision_real) / 2
    macro_recall = (recall_ai + recall_real) / 2
    accuracy = (ai_ai + real_real) / total

    return {
        'labeled_total': total,
        'accuracy': float(accuracy),
        'precision': float(macro_precision),
        'recall': float(macro_recall),
        'confusion': {
            'ai_ai': ai_ai,
            'ai_real': ai_real,
            'real_ai': real_ai,
            'real_real': real_real,
        },
    }


def uploaded_file_to_data_url(uploaded_file):
    content = uploaded_file.read()
    uploaded_file.seek(0)
    mime_type = getattr(uploaded_file, 'content_type', None) or 'image/png'
    encoded = base64.b64encode(content).decode('ascii')
    return f'data:{mime_type};base64,{encoded}'


def write_temp_upload(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1] or '.png'
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        for chunk in uploaded_file.chunks():
            temp_file.write(chunk)
    finally:
        temp_file.close()
    return temp_file.name


def home(request):
    return render(request, 'dashboard/home.html', landing_context(request))


def about(request):
    return render(request, 'dashboard/about.html', landing_context(request))


def features(request):
    return render(request, 'dashboard/features.html', landing_context(request))


def research(request):
    return render(request, 'dashboard/research.html', landing_context(request))


def register(request):
    if request.user.is_authenticated:
        return redirect('home')

    form = RegisterForm(request.POST or None)
    if request.method == 'POST' and form.is_valid():
        user = form.save()
        login(request, user)
        messages.success(request, 'Your SecureLens workspace is ready.')
        return redirect('analyze')
    return render(request, 'dashboard/auth.html', {
        'form': form,
        'title': 'Create your SecureLens account',
        'submit_label': 'Create account',
    })


class SecureLensLoginView(LoginView):
    template_name = 'dashboard/auth.html'
    authentication_form = LoginForm
    redirect_authenticated_user = True

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context.update({'title': 'Welcome back to SecureLens', 'submit_label': 'Sign in'})
        return context


class SecureLensLogoutView(LogoutView):
    next_page = 'home'


@login_required
def analyze(request):
    form = ImageUploadForm(request.POST or None, request.FILES or None)
    if request.method == 'POST' and form.is_valid():
        obj = None
        try:
            capture_source = request.POST.get('capture_source') or None
            live_batch_mode = request.POST.get('live_batch_mode') == '1'
            obj = create_analysis_record(request.user, form.cleaned_data['image'])
            obj, explanation = run_analysis_pipeline(obj, capture_source=capture_source)
            request.session['last_securelens_explanation'] = explanation
        except Exception as error:
            print(f"❌ Prediction error: {error}")
            if obj is None:
                obj = form.save(commit=False)
                obj.owner = request.user
                obj.save()
            img = cv2.imread(obj.image.path)
            stats = collect_image_stats(img, obj.image.path)
            obj.mean_pixel = stats['mean_pixel']
            obj.std_pixel = stats['std_pixel']
            obj.prediction = 'Error'
            obj.confidence = 0.0
            obj.analysis_meta = {'error': str(error), 'stats': stats}
            request.session['last_securelens_explanation'] = "SecureLens hit an error while running the detector."

        obj.save()
        messages.success(request, 'Image analyzed and saved to your workspace.')
        if live_batch_mode:
            return redirect('batch')
        return redirect('result', pk=obj.pk)

    return render(request, 'dashboard/analyze.html', {
        'form': form,
        'analysis_count': request.user.analyses.count(),
    })


@login_required
def result(request, pk):
    obj = get_object_or_404(ImageAnalysis, pk=pk, owner=request.user)
    explanation = request.session.pop('last_securelens_explanation', None)
    if not explanation:
        explanations = {
            'AI': "SecureLens found stronger AI-style detector signals than real-photo signals for this upload.",
            'REAL': "SecureLens saw stronger photo-like cues and model votes than AI-generation cues here.",
            'UNCERTAIN': "This upload landed too close to the middle, so SecureLens chose caution over pretending certainty.",
        }
        explanation = explanations.get(obj.prediction, "This result was saved to your account history.")
    score_label = obj.prediction
    if obj.prediction in ('AI', 'REAL'):
        score_text = f"{int(round(obj.confidence))}% {obj.prediction.title()}"
    elif obj.prediction == 'UNCERTAIN':
        score_text = f"{int(round(obj.confidence))}% Uncertain"
    else:
        score_text = obj.prediction
    return render(request, 'dashboard/result.html', {
        'obj': obj,
        'result_explanation': explanation,
        'analysis_meta': obj.analysis_meta or {},
        'score_text': score_text,
        'authenticity_label': score_label,
        'fft_preview_json': json.dumps((obj.analysis_meta or {}).get('fft_preview', [])),
        'heatmap_preview_json': json.dumps((obj.analysis_meta or {}).get('heatmap_preview', [])),
    })


@login_required
def label_analysis(request, pk):
    obj = get_object_or_404(ImageAnalysis, pk=pk, owner=request.user)
    if request.method == 'POST':
        ground_truth = request.POST.get('ground_truth')
        if ground_truth in ('AI', 'REAL', 'UNCERTAIN'):
            obj.ground_truth = ground_truth
            obj.save(update_fields=['ground_truth'])
            messages.success(request, f'Marked this analysis as {ground_truth}.')
    return redirect('result', pk=obj.pk)


@login_required
def export_report(request, pk):
    obj = get_object_or_404(ImageAnalysis, pk=pk, owner=request.user)
    analysis_meta = obj.analysis_meta or {}
    metadata = analysis_meta.get('metadata', {})
    quality = analysis_meta.get('quality', {})
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import Image as RLImage
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except Exception:
        lines = [
            f"SecureLens report for {obj.image.name}",
            f"Prediction: {obj.prediction}",
            f"Confidence: {obj.confidence:.2f}%",
            f"Resolution: {metadata.get('resolution', 'Unknown')}",
            f"File size: {metadata.get('file_size_kb', 0)} KB",
            f"Sharpness: {quality.get('sharpness', 0):.2f}",
            f"Entropy: {analysis_meta.get('stats', {}).get('entropy', 0):.2f}",
        ]
        response = HttpResponse("\n".join(lines), content_type='text/plain')
        response['Content-Disposition'] = f'attachment; filename="securelens-report-{obj.pk}.txt"'
        return response

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("SecureLens Analysis Report", styles['Title']),
        Spacer(1, 0.2 * inch),
        Paragraph(f"Prediction: <b>{obj.prediction}</b> | Confidence: <b>{obj.confidence:.2f}%</b>", styles['BodyText']),
        Spacer(1, 0.12 * inch),
    ]

    try:
        story.append(RLImage(obj.image.path, width=4.6 * inch, height=3.2 * inch))
        story.append(Spacer(1, 0.18 * inch))
    except Exception:
        pass

    rows = [
        ['Field', 'Value'],
        ['Capture source', analysis_meta.get('capture_source', 'upload')],
        ['Resolution', metadata.get('resolution', 'Unknown')],
        ['File size (KB)', metadata.get('file_size_kb', 0)],
        ['Channels', metadata.get('channels', 0)],
        ['Format', metadata.get('format', 'Unknown')],
        ['Created', metadata.get('created', 'Unavailable')],
        ['Entropy', f"{analysis_meta.get('stats', {}).get('entropy', 0):.2f}"],
        ['Edge density', f"{analysis_meta.get('stats', {}).get('edge_density', 0):.3f}"],
        ['Compression score', f"{analysis_meta.get('stats', {}).get('compression_artifact_score', 0):.2f}"],
        ['Compression level', analysis_meta.get('stats', {}).get('compression_label', 'Unknown')],
        ['Sharpness', f"{quality.get('sharpness', 0):.2f}"],
        ['Blur score', f"{quality.get('blur_score', 0):.2f}"],
        ['Blur level', quality.get('blur_level', 'Unknown')],
        ['Quality label', quality.get('quality_label', 'Unknown')],
        ['Brightness', f"{quality.get('brightness', 0):.2f}"],
        ['Contrast', f"{quality.get('contrast', 0):.2f}"],
    ]
    table = Table(rows, colWidths=[2 * inch, 3.7 * inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f4ecff')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#2f2537')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#d8cfee')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#faf7ff')]),
    ]))
    story.extend([
        table,
        Spacer(1, 0.18 * inch),
        Paragraph(f"Result explanation: {obj.analysis_meta.get('explanation', 'No explanation saved.')}", styles['BodyText']),
    ])
    doc.build(story)
    response = HttpResponse(buffer.getvalue(), content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="securelens-report-{obj.pk}.pdf"'
    return response


@login_required
def batch_analyze(request):
    results = []
    batch_id = uuid4().hex[:10]
    batch_summary = None
    if request.method == 'POST':
        uploads = request.FILES.getlist('images')
        capture_source = request.POST.get('capture_source') or None
        batch_ground_truth = request.POST.get('batch_ground_truth') or ''
        if not uploads:
            messages.error(request, 'Please choose one or more images for batch analysis.')
        else:
            for uploaded in uploads:
                try:
                    obj = create_analysis_record(request.user, uploaded, batch_id=batch_id)
                    obj, explanation = run_analysis_pipeline(obj, capture_source=capture_source)
                    if batch_ground_truth in ('AI', 'REAL'):
                        obj.ground_truth = batch_ground_truth
                    obj.save()
                    results.append(build_batch_result_item(obj, explanation))
                except Exception as error:
                    messages.error(request, f'Batch item {uploaded.name} failed: {error}')
            if results:
                labeled_results = [item['obj'] for item in results if item['obj'].ground_truth in ('AI', 'REAL')]
                correct = sum(1 for item in labeled_results if item.prediction == item.ground_truth)
                batch_summary = {
                    'total': len(results),
                    'labeled_total': len(labeled_results),
                    'correct': correct,
                    'accuracy': (correct / len(labeled_results) * 100.0) if labeled_results else 0.0,
                }
                messages.success(request, f'Batch analysis complete for {len(results)} image(s).')
    return render(request, 'dashboard/batch.html', {
        'batch_id': batch_id,
        'results': results,
        'batch_summary': batch_summary,
        'analysis_count': request.user.analyses.count(),
    })


@login_required
def compare(request):
    comparison = None
    image_a_data_url = None
    image_b_data_url = None
    if request.method == 'POST':
        image_a = request.FILES.get('image_a')
        image_b = request.FILES.get('image_b')
        if not image_a or not image_b:
            messages.error(request, 'Please upload two images to compare.')
        else:
            temp_a = write_temp_upload(image_a)
            temp_b = write_temp_upload(image_b)
            try:
                comparison = compare_image_files(temp_a, temp_b)
                with open(temp_a, 'rb') as file_a:
                    image_a_data_url = f"data:{image_a.content_type or 'image/png'};base64,{base64.b64encode(file_a.read()).decode('ascii')}"
                with open(temp_b, 'rb') as file_b:
                    image_b_data_url = f"data:{image_b.content_type or 'image/png'};base64,{base64.b64encode(file_b.read()).decode('ascii')}"
                messages.success(request, 'Comparison completed.')
            finally:
                try:
                    os.unlink(temp_a)
                except Exception:
                    pass
                try:
                    os.unlink(temp_b)
                except Exception:
                    pass
    return render(request, 'dashboard/compare.html', {
        'comparison': comparison,
        'image_a_data_url': image_a_data_url,
        'image_b_data_url': image_b_data_url,
        'hist_a_json': json.dumps((comparison or {}).get('hist_a', [])),
        'hist_b_json': json.dumps((comparison or {}).get('hist_b', [])),
    })


@login_required
def history(request):
    analyses = request.user.analyses.all().order_by('-uploaded_at')
    return render(request, 'dashboard/history.html', {'analyses': analyses})


@login_required
def delete(request, pk):
    obj = get_object_or_404(ImageAnalysis, pk=pk, owner=request.user)
    obj.image.delete(save=False)
    obj.delete()
    messages.info(request, 'Analysis removed from your workspace.')
    return redirect('history')


@login_required
def stats(request):
    analyses = request.user.analyses.all()
    total = analyses.count()
    real_count = analyses.filter(prediction='REAL').count()
    ai_count = analyses.filter(prediction='AI').count()
    uncertain_count = analyses.filter(prediction='UNCERTAIN').count()
    avg_confidence = analyses.aggregate(Avg('confidence'))['confidence__avg'] or 0
    avg_entropy = 0.0
    avg_edge_density = 0.0
    meta_rows = [item.analysis_meta.get('stats', {}) for item in analyses if isinstance(item.analysis_meta, dict)]
    if meta_rows:
        entropy_values = [row.get('entropy') for row in meta_rows if row.get('entropy') is not None]
        edge_values = [row.get('edge_density') for row in meta_rows if row.get('edge_density') is not None]
        avg_entropy = float(np.mean(entropy_values)) if entropy_values else 0.0
        avg_edge_density = float(np.mean(edge_values)) if edge_values else 0.0
    latest = analyses.order_by('-uploaded_at')[:5]
    dataset_window = list(analyses.order_by('uploaded_at')[:50])
    labels = [item.uploaded_at.strftime('%m-%d') for item in dataset_window]
    entropy_series = [float((item.analysis_meta or {}).get('stats', {}).get('entropy', 0.0)) for item in dataset_window]
    edge_series = [float((item.analysis_meta or {}).get('stats', {}).get('edge_density', 0.0)) for item in dataset_window]
    confidence_series = [float(item.confidence or 0.0) for item in dataset_window]
    metadata_count = sum(1 for item in dataset_window if (item.analysis_meta or {}).get('metadata'))
    performance = compute_dataset_metrics(analyses)
    return render(request, 'dashboard/stats.html', {
        'total': total,
        'real_count': real_count,
        'ai_count': ai_count,
        'uncertain_count': uncertain_count,
        'avg_confidence': avg_confidence,
        'avg_entropy': avg_entropy,
        'avg_edge_density': avg_edge_density,
        'latest': latest,
        'labels_json': json.dumps(labels),
        'entropy_json': json.dumps(entropy_series),
        'edge_json': json.dumps(edge_series),
        'confidence_json': json.dumps(confidence_series),
        'metadata_count': metadata_count,
        'performance': performance,
        'dataset_total': total,
        'dataset_window_size': len(dataset_window),
    })
