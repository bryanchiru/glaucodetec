"""
GlaucoDetec - Entrenamiento con EfficientNetB0
Dataset: EyePACS AIROGS (NRG / RG)
Ejecutar: python ml/train.py
"""
import os
import json
import time
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import numpy as np

# ─── Configuración ────────────────────────────────────────────────────────────
DATA_DIR   = Path(r"c:\Users\bryan\OneDrive\Escritorio\GlaucoDetec\ml\data\eyepacs\EyePACS AIROGS - Luz\release-crop\release-crop")
MODEL_DIR  = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "glaucodetec_best.pth"

BATCH_SIZE   = 32
NUM_EPOCHS   = 30
LR           = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE     = 5
IMG_SIZE     = 224
NUM_CLASSES  = 2
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
print(f"Device: {DEVICE}")

# ─── Transforms ───────────────────────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ─── Dataset ──────────────────────────────────────────────────────────────────
train_ds = datasets.ImageFolder(DATA_DIR / "train",      transform=train_transform)
val_ds   = datasets.ImageFolder(DATA_DIR / "validation", transform=val_transform)
test_ds  = datasets.ImageFolder(DATA_DIR / "test",       transform=val_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
print(f"Clases: {train_ds.classes}")

# ─── Modelo ───────────────────────────────────────────────────────────────────
def build_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Linear(256, NUM_CLASSES),
    )
    return model

model     = build_model().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# ─── Loop de entrenamiento ────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * imgs.size(0)
        correct  += out.argmax(1).eq(labels).sum().item()
        total    += labels.size(0)
    return loss_sum / total, correct / total

def eval_epoch(model, loader, criterion):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    probs_all, labels_all = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out   = model(imgs)
            loss  = criterion(out, labels)
            probs = torch.softmax(out, 1)[:, 1]
            loss_sum += loss.item() * imgs.size(0)
            correct  += out.argmax(1).eq(labels).sum().item()
            total    += labels.size(0)
            probs_all.extend(probs.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
    auc = roc_auc_score(labels_all, probs_all)
    return loss_sum / total, correct / total, auc

history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_auc": []}
best_val_acc   = 0.0
patience_count = 0

print(f"\nEntrenando {NUM_EPOCHS} épocas en {DEVICE}...")
print("-" * 70)

for epoch in range(1, NUM_EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_acc           = train_epoch(model, train_loader, optimizer, criterion)
    vl_loss, vl_acc, vl_auc  = eval_epoch(model, val_loader, criterion)
    scheduler.step()

    history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
    history["val_loss"].append(vl_loss);   history["val_acc"].append(vl_acc)
    history["val_auc"].append(vl_auc)

    flag = " << BEST" if vl_acc > best_val_acc else ""
    print(f"Ep {epoch:02d}/{NUM_EPOCHS} | "
          f"tr_loss:{tr_loss:.4f} tr_acc:{tr_acc:.4f} | "
          f"val_loss:{vl_loss:.4f} val_acc:{vl_acc:.4f} AUC:{vl_auc:.4f} | "
          f"{time.time()-t0:.1f}s{flag}")

    if vl_acc > best_val_acc:
        best_val_acc   = vl_acc
        best_weights   = copy.deepcopy(model.state_dict())
        torch.save(best_weights, MODEL_PATH)
        patience_count = 0
    else:
        patience_count += 1
        if patience_count >= PATIENCE:
            print(f"\nEarly stopping en época {epoch}")
            break

# ─── Evaluación final ─────────────────────────────────────────────────────────
print("\nCargando mejor modelo...")
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
_, test_acc, test_auc = eval_epoch(model, test_loader, criterion)

model.eval()
preds_all, labels_all = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        out = model(imgs.to(DEVICE))
        preds_all.extend(out.argmax(1).cpu().numpy())
        labels_all.extend(labels.numpy())

print("\n" + "=" * 55)
print("RESULTADOS FINALES — TEST SET")
print("=" * 55)
print(f"Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"AUC-ROC:  {test_auc:.4f}")
print("\nReporte de clasificación:")
print(classification_report(labels_all, preds_all, target_names=train_ds.classes))
print("Matriz de confusión:")
print(confusion_matrix(labels_all, preds_all))

(MODEL_DIR / "training_history.json").write_text(json.dumps(history, indent=2))
print(f"\nModelo guardado: {MODEL_PATH}")
