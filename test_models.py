from transformers import pipeline
from PIL import Image
import sys
ssl_fix = __import__('ssl')
ssl_fix._create_default_https_context = ssl_fix._create_unverified_context

# Test image path — pass any image as argument
img_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample.jpg"
img = Image.open(img_path).convert('RGB')

models_to_test = [
    "umm-maybe/AI-image-detector",
    "Organika/sdxl-detector",
    "haywoodsloan/autotrain-ai-vs-real-image-classifier",
]

for model_name in models_to_test:
    try:
        print(f"\nTesting: {model_name}")
        detector = pipeline("image-classification", model=model_name)
        result   = detector(img)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Failed: {e}")
