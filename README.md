# 🔐 SecureLens

> AI-powered image authenticity detection — distinguishing real images from AI-generated ones using deep learning.

---

## 📌 Overview

SecureLens is a computer vision project that uses transfer learning (MobileNetV2) to classify images as real or AI-generated. It includes a full training pipeline with two-phase fine-tuning, class imbalance handling, and image analysis utilities.

---

## 🚀 Features

- ✅ Binary image classification** — Real vs. AI-generated
- ✅ Transfer learning** with MobileNetV2 (pretrained on ImageNet)
- ✅ Two-phase fine-tuning** — frozen base → selective unfreeze
- ✅ Class imbalance handling** via dynamic class weights
- ✅ Data augmentation** — rotation, flip, zoom, brightness, shear
- ✅ Training callbacks** — EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
- ✅ Image analysis dashboard** — mean/std pixel stats, scatter plots, image grids
- ✅ Full training history plots** — accuracy, loss, AUC, precision, recall

---

## 🗂️ Project Structure

```
SecureLens/
├── src/
│   ├── dashboard_analysis.py   # Image stats & visualization
│   └── train.py                # CNN training pipeline
├── data/
│   └── dataset/
│       ├── real/               # Real image samples
│       └── ai/                 # AI-generated image samples
├── models/
│   ├── securelens_best.keras   # Best checkpoint (Phase 1)
│   ├── securelens_cnn.keras    # Final trained model
│   └── training_history.png   # Training plots
├── logs/
│   └── securelens/             # TensorBoard logs
├── .venv/                      # Virtual environment
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/SecureLens.git
cd SecureLens
```

### 2. Create and activate virtual environment
```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install tensorflow opencv-python numpy pandas matplotlib
```

---

## 🧠 Model Architecture

```
MobileNetV2 (pretrained, ImageNet)
        ↓
GlobalAveragePooling2D
        ↓
Dense(128, ReLU) → Dropout(0.4)
        ↓
Dense(64, ReLU)  → Dropout(0.3)
        ↓
Dense(1, Sigmoid)   ← Binary output
```

**Training config:**
| Parameter | Value |
|-----------|-------|
| Image size | 224 × 224 |
| Batch size | 16 |
| Phase 1 epochs | 30 (+ EarlyStopping) |
| Phase 2 epochs | 15 (+ EarlyStopping) |
| Phase 1 LR | 1e-4 |
| Phase 2 LR | 1e-5 |
| Optimizer | Adam |
| Loss | Binary Crossentropy |

---

## 🏋️ Training

### Prepare your dataset
Organize images into two subfolders:
```
data/dataset/
├── real/        ← real photographs
└── ai/          ← AI-generated images
```

### Run training
```bash
python src/train.py
```

Training runs in two phases:
1. Phase 1 — Base MobileNetV2 frozen, only top layers trained
2. Phase 2 — Top 30 layers of MobileNetV2 unfrozen for fine-tuning

Models are saved to `models/` and training history is plotted automatically.

---

## 📊 Image Analysis Dashboard

Analyze raw image statistics (mean pixel value, standard deviation) and visualize your dataset:

```bash
python src/dashboard_analysis.py
```

Outputs:
- Printed DataFrame of image stats
- Scatter plot: Mean vs Std Dev per image
- Grid view of up to 6 sample images

---

## 📈 TensorBoard

Monitor training live:
```bash
tensorboard --logdir logs/securelens
```
Then open `http://127.0.0.1:8000/` in your browser.

---

## 📋 Metrics Tracked

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct predictions |
| Loss | Binary crossentropy |
| Precision | How many predicted AI images were actually AI |
| Recall | How many actual AI images were caught |
| AUC | Area under ROC curve |

---

## 🛠️ Requirements

- Python 3.8+
- TensorFlow 2.x
- OpenCV (`opencv-python`)
- NumPy
- Pandas
- Matplotlib

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 👤 Author

**Aarya Waghmare**  
[GitHub](https://github.com/AaryaWaghmare7) · [LinkedIn](https://www.linkedin.com/in/aarya-waghmare-90389031b/)
---

> *SecureLens — See through the artificial.*
