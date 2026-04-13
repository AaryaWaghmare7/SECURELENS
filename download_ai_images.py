import os
import urllib.request
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

os.makedirs("data/dataset/ai", exist_ok=True)

# Use Picsum with different seeds for AI-style images
# We'll use grayscale + different filters to simulate AI images
print("Downloading AI images...")
for i in range(200, 400):
    url = f"https://picsum.photos/seed/{i}/224/224?grayscale&blur=1"
    urllib.request.urlretrieve(url, f"data/dataset/ai/{i-200}.jpg")
    print(f"AI: {i-199}/200", end="\r")

print("\n✅ AI images done!")
print(f"Real images: {len(os.listdir('data/dataset/real'))}")
print(f"AI images  : {len(os.listdir('data/dataset/ai'))}")
