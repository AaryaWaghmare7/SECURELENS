import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = "data"

# Supported image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

data = []

for img_name in os.listdir(DATA_DIR):
    # Fix 2: skip non-image files early
    if os.path.splitext(img_name)[1].lower() not in IMAGE_EXTS:
        continue

    img_path = os.path.join(DATA_DIR, img_name)
    if not os.path.isfile(img_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w, c = img.shape
    mean_pixel = np.mean(img)
    std_pixel = np.std(img)
    data.append([img_name, "unknown", w, h, c, mean_pixel, std_pixel])

df = pd.DataFrame(
    data, columns=["Image", "Label", "Width", "Height", "Channels", "Mean", "Std"]
)
print(df)

# --- Scatter plot ---
plt.figure(figsize=(6, 6))
plt.scatter(df["Mean"], df["Std"], c="green")

for i, txt in enumerate(df["Image"]):
    # Fix 1: use .iloc[i] for safe positional indexing
    plt.annotate(txt, (df["Mean"].iloc[i], df["Std"].iloc[i]))

plt.title("Mean vs Std Dev of Images")
plt.xlabel("Mean Pixel Value")
plt.ylabel("Std Deviation")
plt.tight_layout()
plt.show()
print("Saved scatter_plot.png")

# --- Image grid ---
plt.figure(figsize=(12, 6))

# Fix 3: index=False avoids the implicit 'Index' attribute on each tuple
for i, row in enumerate(df.itertuples(index=False)):
    if i >= 6:
        break
    img_path = os.path.join(DATA_DIR, row.Image)
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(2, 3, i + 1)
    plt.imshow(img_rgb)
    plt.title(f"{row.Image}\nMean:{row.Mean:.1f}  Std:{row.Std:.1f}")
    plt.axis("off")

plt.tight_layout()
plt.savefig("image_grid.png", dpi=150)
plt.close()
print("Saved image_grid.png")
