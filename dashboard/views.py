import cv2
import numpy as np
import os
from django.shortcuts import render, redirect, get_object_or_404
from django.db.models import Avg
from .models import ImageAnalysis
from .forms import ImageUploadForm

def load_model():
    try:
        from transformers import pipeline
        detector = pipeline(
            "image-classification",
            model="Organika/sdxl-detector"
        )
        print("✅ Model loaded!")
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

                    # Find AI score specifically
                    ai_score   = next((r['score'] for r in results if 'artificial' in r['label'].lower() or 'fake' in r['label'].lower() or 'ai' in r['label'].lower()), None)
                    real_score = next((r['score'] for r in results if 'real' in r['label'].lower() or 'human' in r['label'].lower()), None)

                    print(f"AI score: {ai_score}, Real score: {real_score}")

                    if ai_score and real_score:
                        if ai_score > real_score:
                            obj.prediction = 'AI'
                            obj.confidence = round(ai_score * 100, 2)
                        else:
                            obj.prediction = 'REAL'
                            obj.confidence = round(real_score * 100, 2)
                    else:
                        # fallback to top result
                        top   = results[0]
                        label = top['label'].lower()
                        obj.prediction = 'AI' if any(w in label for w in ['artificial', 'fake', 'ai', 'generated', 'synthetic']) else 'REAL'
                        obj.confidence = round(top['score'] * 100, 2)

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
