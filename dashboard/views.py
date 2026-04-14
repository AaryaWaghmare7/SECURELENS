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

DEFAULT_MODEL = os.getenv("SECURELENS_MODEL", "umm-maybe/AI-image-detector")


def classify_prediction(results):
    if not results:
        return 'Error', 0.0

    ai_keywords = ('ai', 'fake', 'artificial', 'generated', 'synthetic')
    real_keywords = ('real', 'human', 'authentic', 'natural')

    ai_score = max(
        (result['score'] for result in results if any(word in result['label'].lower() for word in ai_keywords)),
        default=None,
    )
    real_score = max(
        (result['score'] for result in results if any(word in result['label'].lower() for word in real_keywords)),
        default=None,
    )

    if ai_score is not None and real_score is not None:
        if ai_score > real_score:
            return 'AI', round(ai_score * 100, 2)
        return 'REAL', round(real_score * 100, 2)

    top = max(results, key=lambda item: item['score'])
    label = top['label'].lower()
    prediction = 'AI' if any(word in label for word in ai_keywords) else 'REAL'
    return prediction, round(top['score'] * 100, 2)


@lru_cache(maxsize=1)
def load_model():
    try:
        from transformers import pipeline

        detector = pipeline("image-classification", model=DEFAULT_MODEL)
        print(f"✅ Model loaded: {DEFAULT_MODEL}")
        return detector
    except Exception as error:
        print(f"❌ Model load error: {error}")
        return None


def landing_context(request):
    analyses = ImageAnalysis.objects.all().order_by('-uploaded_at')
    recent = analyses[:6]
    return {
        'analyses': recent,
        'total': analyses.count(),
        'real_count': analyses.filter(prediction='REAL').count(),
        'ai_count': analyses.filter(prediction='AI').count(),
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
    return render(request, 'dashboard/auth.html', {'form': form, 'title': 'Create your SecureLens account', 'submit_label': 'Create account'})


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
        if img is not None:
            obj.mean_pixel = float(np.mean(img))
            obj.std_pixel = float(np.std(img))

        try:
            detector = load_model()
            if detector:
                pil_img = Image.open(obj.image.path).convert('RGB')
                results = detector(pil_img)
                obj.prediction, obj.confidence = classify_prediction(results)
            else:
                obj.prediction = 'No Model'
                obj.confidence = 0.0
        except Exception as error:
            print(f"❌ Prediction error: {error}")
            obj.prediction = 'Error'
            obj.confidence = 0.0

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
    return render(request, 'dashboard/result.html', {'obj': obj})


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
    avg_confidence = analyses.aggregate(Avg('confidence'))['confidence__avg'] or 0
    latest = analyses.order_by('-uploaded_at')[:5]
    return render(request, 'dashboard/stats.html', {
        'total': total,
        'real_count': real_count,
        'ai_count': ai_count,
        'avg_confidence': avg_confidence,
        'latest': latest,
    })
