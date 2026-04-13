import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

DATA_DIR = "data"

means, noises, names = [], [], []

for file in os.listdir(DATA_DIR):
    if file.endswith(".jpg"):
        img = cv2.imread(os.path.join(DATA_DIR, file), 0)
        if img is None:
            continue
        means.append(np.mean(img))
        noises.append(cv2.Laplacian(img, cv2.CV_64F).var())
        names.append(file)

plt.figure()
plt.scatter(means, noises)
plt.xlabel("Mean Pixel Value")
plt.ylabel("Noise Variance")
plt.title("SecureLens: Image Feature Distribution")

for i, name in enumerate(names):
    plt.annotate(name, (means[i], noises[i]), fontsize=8)

plt.show()
