import os
import urllib.request
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

os.makedirs("data/dataset/real", exist_ok=True)
os.makedirs("data/dataset/ai", exist_ok=True)

# Download real photos from Lorem Picsum
print("Downloading real images...")
for i in range(200):
    url = f"https://picsum.photos/seed/{i}/224/224"
    urllib.request.urlretrieve(url, f"data/dataset/real/{i}.jpg")
    print(f"Real: {i+1}/200", end="\r")

print("\n✅ Real images done! Now downloading AI images...")

# Download AI images from This Person Does Not Exist
for i in range(200):
    url = "https://thispersondoesnotexist.com"
    urllib.request.urlretrieve(url, f"data/dataset/ai/{i}.jpg")
    print(f"AI: {i+1}/200", end="\r")

print("\n✅ All done!")
print(f"Real images: {len(os.listdir('data/dataset/real'))}")
print(f"AI images  : {len(os.listdir('data/dataset/ai'))}")
