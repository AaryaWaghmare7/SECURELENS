import os
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image
from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import LoginView, LogoutView
from django.db.models import Avg
from django.shortcuts import get_object_or_404, redirect, render

from .forms import ImageUploadForm, LoginForm, RegisterForm
from .models import ImageAnalysis

DEFAULT_MODELS = [
    ("umm-maybe/AI-image-detector", 1.0),
    ("Organika/sdxl-detector", 0.9),
    ("haywoodsloan/autotrain-ai-vs-real-image-classifier", 0.8),
]
AI_KEYWORDS = ('ai', 'fake', 'artificial', 'generated', 'synthetic', 'sdxl', 'midjourney')
REAL_KEYWORDS = ('real', 'human', 'authentic', 'natural', 'photo', 'photograph')


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

    if ai_score == 0.0 and real_score == 0.0:
        label = top['label'].lower()
        if any(word in label for word in AI_KEYWORDS):
            ai_score = top['score']
        else:
            real_score = top['score']

    return {
        'ai_score': ai_score,
        'real_score': real_score,
        'top_score': top['score'],
        'top_label': top['label'],
    }


def compute_edge_density(img):
    if img is None:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    return float(np.count_nonzero(edges) / edges.size)


def apply_image_heuristics(ai_score, real_score, stats):
    edge_density = stats['edge_density']
    std_pixel = stats['std_pixel']

    if edge_density < 0.022 and std_pixel < 42:
        ai_score += 0.035
    elif edge_density > 0.075 and std_pixel > 55:
        real_score += 0.03

    return ai_score, real_score


def classify_prediction(model_runs, stats):
    if not model_runs:
        return 'No Model', 0.0, "No detector could be loaded for this upload."

    weighted_ai = 0.0
    weighted_real = 0.0
    total_weight = 0.0

    for run in model_runs:
        weighted_ai += run['ai_score'] * run['weight']
        weighted_real += run['real_score'] * run['weight']
        total_weight += run['weight']

    if total_weight == 0:
        return 'Error', 0.0, "The detector responses were empty."

    avg_ai = weighted_ai / total_weight
    avg_real = weighted_real / total_weight
    avg_ai, avg_real = apply_image_heuristics(avg_ai, avg_real, stats)

    confidence = round(max(avg_ai, avg_real) * 100, 2)
    margin = abs(avg_ai - avg_real)
    strongest_signal = max(max(run['ai_score'], run['real_score']) for run in model_runs)
    ai_votes = sum(1 for run in model_runs if run['ai_score'] > run['real_score'])
    real_votes = sum(1 for run in model_runs if run['real_score'] > run['ai_score'])
    vote_gap = abs(ai_votes - real_votes)

    if max(avg_ai, avg_real) < 0.52:
        explanation = "The detector signals were too weak overall, so SecureLens marked this upload as uncertain."
        return 'UNCERTAIN', confidence, explanation

    if margin < 0.035 and strongest_signal < 0.78:
        explanation = "The detector votes were too close, so SecureLens marked this upload as uncertain instead of forcing a shaky answer."
        return 'UNCERTAIN', confidence, explanation

    if vote_gap == 0 and strongest_signal < 0.74:
        explanation = "The detector ensemble split too evenly on this image, so SecureLens left the verdict as uncertain."
        return 'UNCERTAIN', confidence, explanation

    if avg_ai > avg_real:
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


def collect_image_stats(img):
    return {
        'mean_pixel': float(np.mean(img)) if img is not None else 0.0,
        'std_pixel': float(np.std(img)) if img is not None else 0.0,
        'edge_density': compute_edge_density(img) if img is not None else 0.0,
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


def home(request):
    return render(request, 'dashboard/home.html', landing_context(request))


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
        obj = form.save(commit=False)
        obj.owner = request.user
        obj.save()

        img = cv2.imread(obj.image.path)
        stats = collect_image_stats(img)
        obj.mean_pixel = stats['mean_pixel']
        obj.std_pixel = stats['std_pixel']

        try:
            pil_img = Image.open(obj.image.path).convert('RGB')
            model_runs = []
            for detector_info in load_detectors():
                results = detector_info['detector'](pil_img)
                summary = summarize_model_results(results)
                summary.update({
                    'weight': detector_info['weight'],
                    'model_name': detector_info['name'],
                })
                model_runs.append(summary)

            obj.prediction, obj.confidence, explanation = classify_prediction(model_runs, stats)
            request.session['last_securelens_explanation'] = explanation
        except Exception as error:
            print(f"❌ Prediction error: {error}")
            obj.prediction = 'Error'
            obj.confidence = 0.0
            request.session['last_securelens_explanation'] = "SecureLens hit an error while running the detector."

        obj.save()
        messages.success(request, 'Image analyzed and saved to your workspace.')
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
    return render(request, 'dashboard/result.html', {
        'obj': obj,
        'result_explanation': explanation,
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
    latest = analyses.order_by('-uploaded_at')[:5]
    return render(request, 'dashboard/stats.html', {
        'total': total,
        'real_count': real_count,
        'ai_count': ai_count,
        'uncertain_count': uncertain_count,
        'avg_confidence': avg_confidence,
        'latest': latest,
    })
