"""
train_all.py – Единый скрипт для подготовки данных и обучения классификаторов.
Перед переобучением запрашивает подтверждение, если модели уже существуют.
"""
import os, sys, shutil, json
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

# ---------- НАСТРОЙКИ ----------
EPOCHS_SPECIES = 10
EPOCHS_DISEASE = 10
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Устройство: {DEVICE}")

BASE_DIR = Path(__file__).resolve().parent.parent   # из to_train/ в корень проекта
RAW_DIR = BASE_DIR / "dataset" / "raw_data"
CLASSIFICATION_DIR = BASE_DIR / "classification_data"
MODELS_DIR = BASE_DIR / "ml_models"
MODELS_DIR.mkdir(exist_ok=True)

CLASS_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def check_existing_models():
    """Проверяет, есть ли уже обученные модели, и запрашивает подтверждение на перезапись."""
    species_exists = (MODELS_DIR / 'species_model.pth').exists()
    disease_exists = any(MODELS_DIR.glob('disease_*_resnet50.pth'))
    if species_exists or disease_exists:
        print("⚠️  Обнаружены ранее обученные модели в ml_models/")
        answer = input("Перезаписать их? [y/N]: ").strip().lower()
        if answer != 'y':
            print("❌ Обучение отменено пользователем.")
            sys.exit(0)
        else:
            print("🔄 Перезапись моделей...")

def prepare_data():
    print("\n" + "="*50)
    print("📦 ШАГ 1: Извлечение кропов из аннотаций")
    print("="*50)

    if CLASSIFICATION_DIR.exists():
        shutil.rmtree(CLASSIFICATION_DIR)

    for split in ['train', 'val']:
        (CLASSIFICATION_DIR / 'species' / split).mkdir(parents=True, exist_ok=True)
        (CLASSIFICATION_DIR / 'disease' / split).mkdir(parents=True, exist_ok=True)

    def process_csv(csv_name, image_subdir, out_split):
        csv_path = RAW_DIR / csv_name
        if not csv_path.exists():
            print(f"❌ Файл не найден: {csv_path}")
            sys.exit(1)

        df = pd.read_csv(csv_path)

        col_file = [c for c in df.columns if 'file' in c.lower() or 'image' in c.lower()][0]
        col_xmin = [c for c in df.columns if 'xmin' in c.lower()][0]
        col_ymin = [c for c in df.columns if 'ymin' in c.lower()][0]
        col_xmax = [c for c in df.columns if 'xmax' in c.lower()][0]
        col_ymax = [c for c in df.columns if 'ymax' in c.lower()][0]
        col_class = [c for c in df.columns if 'class' in c.lower()][0]

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Обработка {csv_name}"):
            filename = row[col_file]
            img_path = RAW_DIR / image_subdir / filename
            if not img_path.exists():
                stem = Path(filename).stem
                for ext in ['.jpg', '.jpeg', '.png', '.JPG']:
                    candidate = RAW_DIR / image_subdir / (stem + ext)
                    if candidate.exists():
                        img_path = candidate
                        break
            if not img_path.exists():
                continue

            try:
                img = Image.open(img_path).convert('RGB')
                w, h = img.size
            except:
                continue

            xmin = max(0, int(row[col_xmin]))
            ymin = max(0, int(row[col_ymin]))
            xmax = min(w, int(row[col_xmax]))
            ymax = min(h, int(row[col_ymax]))
            if xmax <= xmin or ymax <= ymin:
                continue

            crop = img.crop((xmin, ymin, xmax, ymax))
            if crop.width < 10 or crop.height < 10:
                continue

            disease_full = str(row[col_class]).strip()
            parts = disease_full.split()
            species = parts[0] if len(parts) >= 2 else disease_full
            disease = disease_full

            species_dir = CLASSIFICATION_DIR / 'species' / out_split / species
            species_dir.mkdir(parents=True, exist_ok=True)
            crop.save(species_dir / f"{Path(filename).stem}_{xmin}_{ymin}.jpg")

            disease_dir = CLASSIFICATION_DIR / 'disease' / out_split / species / disease
            disease_dir.mkdir(parents=True, exist_ok=True)
            crop.save(disease_dir / f"{Path(filename).stem}_{xmin}_{ymin}.jpg")

    process_csv('train_labels.csv', 'TRAIN', 'train')
    process_csv('test_labels.csv', 'TEST', 'val')
    print("✅ Кропы сохранены в", CLASSIFICATION_DIR)

def train_species():
    print("\n" + "="*50)
    print("🧬 ШАГ 2: Обучение классификатора вида")
    print("="*50)

    data_dir = CLASSIFICATION_DIR / 'species'
    train_ds = datasets.ImageFolder(data_dir / 'train', transform=CLASS_TRANSFORM)
    val_ds   = datasets.ImageFolder(data_dir / 'val',   transform=CLASS_TRANSFORM)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    with open(MODELS_DIR / 'species_classes.json', 'w') as f:
        json.dump(train_ds.class_to_idx, f)
    num_classes = len(train_ds.classes)
    print(f"Количество видов: {num_classes}")

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_acc = 0
    for epoch in range(EPOCHS_SPECIES):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / total
        print(f"Epoch {epoch+1}/{EPOCHS_SPECIES} Val Acc = {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODELS_DIR / 'species_model.pth')
    print(f"✅ Классификатор вида сохранён (лучшая точность {best_acc:.4f})")

def train_diseases():
    print("\n" + "="*50)
    print("🦠 ШАГ 3: Обучение классификаторов болезней")
    print("="*50)

    base_dir = CLASSIFICATION_DIR / 'disease'
    for species_dir in base_dir.glob('train/*'):
        if not species_dir.is_dir():
            continue
        species = species_dir.name
        val_path = base_dir / 'val' / species
        if not val_path.exists():
            print(f"⚠️  Пропущен {species}: нет валидационных данных")
            continue

        print(f"\n--- {species} ---")
        train_ds = datasets.ImageFolder(species_dir, transform=CLASS_TRANSFORM)
        val_ds   = datasets.ImageFolder(val_path,   transform=CLASS_TRANSFORM)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

        with open(MODELS_DIR / f'disease_{species}_classes.json', 'w') as f:
            json.dump(train_ds.class_to_idx, f)

        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))
        model.to(DEVICE)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        best_acc = 0
        for epoch in range(EPOCHS_DISEASE):
            model.train()
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    pred = model(x).argmax(1)
                    correct += (pred == y).sum().item()
                    total += y.size(0)
            acc = correct / total
            print(f"  Epoch {epoch+1}/{EPOCHS_DISEASE} Acc = {acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), MODELS_DIR / f'disease_{species}_resnet50.pth')
        print(f"  ✅ {species} готов (лучшая точность {best_acc:.4f})")

if __name__ == "__main__":
    check_existing_models()
    prepare_data()
    train_species()
    train_diseases()
    print("\n" + "="*50)
    print("🏁 Все модели обучены! Можно запускать pipeline.py")
    print("="*50)