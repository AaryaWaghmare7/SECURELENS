import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────
DATA_DIR   = "data/dataset"
MODEL_DIR  = "models"
IMG_SIZE   = 224
BATCH_SIZE = 8
EPOCHS     = 5
DEVICE     = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(f"🖥️  Using device: {DEVICE}")
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────────
# 1. Data
# ─────────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

full_dataset = datasets.ImageFolder(DATA_DIR)
print(f"📁 Classes: {full_dataset.class_to_idx}")
print(f"🖼️  Total images: {len(full_dataset)}")

val_size   = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

train_ds.dataset.transform = train_transforms
val_ds.dataset.transform   = val_transforms

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

# ─────────────────────────────────────────────────
# 2. Model — MobileNetV2
# ─────────────────────────────────────────────────
model = models.mobilenet_v2(weights='IMAGENET1K_V1')

# Freeze base layers
for param in model.parameters():
    param.requires_grad = False

# Replace classifier
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.last_channel, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 1),
    nn.Sigmoid()
)
model = model.to(DEVICE)

# ─────────────────────────────────────────────────
# 3. Training
# ─────────────────────────────────────────────────
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-4)

train_accs, val_accs, train_losses, val_losses = [], [], [], []
best_val_acc = 0.0

print("\n🚀 Phase 1 — Training top layers...\n")

for epoch in range(EPOCHS):
    # Train
    model.train()
    running_loss, correct, total = 0, 0, 0
    for imgs, labels in train_loader:
        imgs   = imgs.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        preds    = (outputs > 0.5).float()
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    train_acc  = correct / total
    train_loss = running_loss / len(train_loader)

    # Validate
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs   = imgs.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            val_loss    += loss.item()
            preds        = (outputs > 0.5).float()
            val_correct += (preds == labels).sum().item()
            val_total   += labels.size(0)

    val_acc  = val_correct / val_total
    val_loss = val_loss / len(val_loader)

    train_accs.append(train_acc)
    val_accs.append(val_acc)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "securelens_best.pth"))
        print(f"   ✅ Best model saved (val_acc: {val_acc:.4f})")

# ─────────────────────────────────────────────────
# 4. Fine Tuning — Unfreeze last layers
# ─────────────────────────────────────────────────
print("\n🔓 Phase 2 — Fine-tuning...\n")
for param in model.features[-5:].parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-5
)

for epoch in range(15):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for imgs, labels in train_loader:
        imgs   = imgs.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        preds    = (outputs > 0.5).float()
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    print(f"FT Epoch {epoch+1}/15 | Loss: {running_loss/len(train_loader):.4f} | Acc: {correct/total:.4f}")

# ─────────────────────────────────────────────────
# 5. Save Final Model
# ─────────────────────────────────────────────────
torch.save(model.state_dict(), os.path.join(MODEL_DIR, "securelens_cnn.pth"))
print("\n✅ Final model saved to models/securelens_cnn.pth")

# ─────────────────────────────────────────────────
# 6. Plot
# ─────────────────────────────────────────────────
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs,   label='Val Acc')
plt.title('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses,   label='Val Loss')
plt.title('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("models/training_history.png")
plt.show()
print("📊 Training history saved.")
