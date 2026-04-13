import cv2
import numpy as np
import os
from django.shortcuts import render, redirect, get_object_or_404
from django.db.models import Avg
from .models import ImageAnalysis
from .forms import ImageUploadForm

DEFAULT_MODEL = os.getenv("SECURELENS_MODEL", "umm-maybe/AI-image-detector")


def classify_prediction(results):
    if not results:
        return 'Error', 0.0

    ai_keywords = ('ai', 'fake', 'artificial', 'generated', 'synthetic')
    real_keywords = ('real', 'human', 'authentic', 'natural')

    ai_score = max(
        (r['score'] for r in results if any(word in r['label'].lower() for word in ai_keywords)),
        default=None,
    )
    real_score = max(
        (r['score'] for r in results if any(word in r['label'].lower() for word in real_keywords)),
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

def load_model():
    try:
        from transformers import pipeline
        detector = pipeline(
            "image-classification",
            model=DEFAULT_MODEL
        )
        print(f"✅ Model loaded: {DEFAULT_MODEL}")
        return detector
    except Exception as e:
        print(f"❌ Model load error: {e}")
        return None

def home(request):
    analyses   = ImageAnalysis.objects.all().order_by('-uploaded_at')[:10]
    total      = ImageAnalysis.objects.count()
    real_count = ImageAnalysis.objects.filter(prediction='REAL').count()
    ai_count   = ImageAnalysis.objects.filter(prediction='AI').count()
    no_model   = ImageAnalysis.objects.filter(prediction='No Model').count()
    return render(request, 'dashboard/home.html', {
        'analyses':   analyses,
        'total':      total,
        'real_count': real_count,
        'ai_count':   ai_count,
        'no_model':   no_model,
    })

def analyze(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            obj = form.save()
            img = cv2.imread(obj.image.path)
            obj.mean_pixel = float(np.mean(img))
            obj.std_pixel  = float(np.std(img))

            try:
                from PIL import Image
                detector = load_model()
                if detector:
                    pil_img = Image.open(obj.image.path).convert('RGB')
                    results = detector(pil_img)
                    print(f"Raw results: {results}")
                    obj.prediction, obj.confidence = classify_prediction(results)

                    print(f"✅ Final: {obj.prediction} ({obj.confidence}%)")
                else:
                    obj.prediction = 'No Model'
                    obj.confidence = 0.0
            except Exception as e:
                print(f"❌ Prediction error: {e}")
                obj.prediction = 'Error'
                obj.confidence = 0.0

            obj.save()
            return redirect('result', pk=obj.pk)
    else:
        form = ImageUploadForm()
    return render(request, 'dashboard/analyze.html', {'form': form})

def result(request, pk):
    obj = get_object_or_404(ImageAnalysis, pk=pk)
    return render(request, 'dashboard/result.html', {'obj': obj})

def history(request):
    analyses = ImageAnalysis.objects.all().order_by('-uploaded_at')
    return render(request, 'dashboard/history.html', {'analyses': analyses})

def delete(request, pk):
    obj = get_object_or_404(ImageAnalysis, pk=pk)
    obj.image.delete()
    obj.delete()
    return redirect('history')

def stats(request):
    total          = ImageAnalysis.objects.count()
    real_count     = ImageAnalysis.objects.filter(prediction='REAL').count()
    ai_count       = ImageAnalysis.objects.filter(prediction='AI').count()
    avg_confidence = ImageAnalysis.objects.aggregate(Avg('confidence'))['confidence__avg'] or 0
    return render(request, 'dashboard/stats.html', {
        'total':          total,
        'real_count':     real_count,
        'ai_count':       ai_count,
        'avg_confidence': avg_confidence,
    })
