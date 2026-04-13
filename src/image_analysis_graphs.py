import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

IMAGE_FOLDER = "data"

# Collect image stats
data = []

for img_name in os.listdir(IMAGE_FOLDER):
    img_path = os.path.join(IMAGE_FOLDER, img_name)
    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w, c = img.shape
    mean_pixel = np.mean(img)
    std_pixel = np.std(img)
    data.append([img_name, w, h, c, mean_pixel, std_pixel])

# Create DataFrame
df = pd.DataFrame(data, columns=["Image", "Width", "Height", "Channels", "Mean", "Std"])
print(df)

# --- Plot 1: Histogram of mean pixel values ---
plt.figure(figsize=(8, 4))
plt.hist(df["Mean"], bins=10, color="skyblue", edgecolor="black")
plt.title("Histogram of Mean Pixel Values")
plt.xlabel("Mean Pixel Value")
plt.ylabel("Number of Images")
plt.show()

# --- Plot 2: Histogram of standard deviation ---
plt.figure(figsize=(8, 4))
plt.hist(df["Std"], bins=10, color="salmon", edgecolor="black")
plt.title("Histogram of Pixel Standard Deviation")
plt.xlabel("Std Deviation")
plt.ylabel("Number of Images")
plt.show()

# --- Plot 3: Scatter plot (Mean vs Std) ---
plt.figure(figsize=(6, 6))
plt.scatter(df["Mean"], df["Std"], c="green")
for i, txt in enumerate(df["Image"]):
    plt.annotate(txt, (df["Mean"][i], df["Std"][i]))
plt.title("Scatter Plot: Mean vs Std Dev of Images")
plt.xlabel("Mean Pixel Value")
plt.ylabel("Std Deviation")
plt.show()

# --- Plot 4: Display sample images in a grid ---
plt.figure(figsize=(10, 5))
for i, img_name in enumerate(os.listdir(IMAGE_FOLDER)[:6]):  # first 6 images
    img_path = os.path.join(IMAGE_FOLDER, img_name)
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(2, 3, i + 1)
    plt.imshow(img_rgb)
    plt.title(img_name)
    plt.axis("off")
plt.tight_layout()
plt.show()
