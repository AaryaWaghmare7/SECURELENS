"""Create disposable, procedurally generated fixtures; not labeled ML data."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parents[1] / "backend/.local/demo"
root.mkdir(parents=True, exist_ok=True)
x, y = np.meshgrid(np.linspace(0, 1, 900), np.linspace(0, 1, 600))
pixels = np.stack([235 - y * 45 + x * 10, 214 - y * 55, 211 - y * 38], axis=-1)
image = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
draw = ImageDraw.Draw(image)
draw.ellipse((625, 75, 760, 210), fill="#fff0ce")
draw.polygon([(0, 385), (230, 195), (510, 405), (740, 245), (900, 365), (900, 600), (0, 600)], fill="#a98796")
draw.polygon([(0, 480), (370, 320), (660, 480), (900, 395), (900, 600), (0, 600)], fill="#775b69")
draw.text((28, 560), "SecureLens procedural test fixture / not a real camera image", fill="white")
image.save(root / "sample-a.png")
image.crop((15, 0, 900, 600)).resize(image.size).save(root / "sample-b.jpg", quality=75)
print(f"Disposable fixtures: {root}")
