"""
pipeline.py – Финальный пайплайн детекции и классификации болезней растений.
Модели загружаются из ml_models/, тестовое изображение из dataset/raw_data/TRAIN/
"""
import cv2, torch, json
from pathlib import Path
from ultralytics import YOLO
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

# ---------- НАСТРОЙКИ ----------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Устройство: {DEVICE}")

BASE_DIR = Path(__file__).resolve().parent          # корень проекта
MODELS_DIR = BASE_DIR / 'ml_models'
DATASET_DIR = BASE_DIR / 'dataset' / 'raw_data'

# Детектор
DETECTOR_PATH = MODELS_DIR / 'leaf_detector.pt'
detector = YOLO(str(DETECTOR_PATH))

CLASS_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---------- Загрузка классификатора вида ----------
with open(MODELS_DIR / 'species_classes.json', 'r') as f:
    species_idx = json.load(f)
species_names = {v: k for k, v in species_idx.items()}
num_species = len(species_names)

species_model = models.efficientnet_b0(weights=None)
species_model.classifier[1] = torch.nn.Linear(1280, num_species)
species_model.load_state_dict(torch.load(MODELS_DIR / 'species_model.pth', map_location=DEVICE))
species_model.to(DEVICE)
species_model.eval()

# ---------- Загрузка классификаторов болезней ----------
disease_models = {}
disease_classes = {}
for pth_path in MODELS_DIR.glob('disease_*_resnet50.pth'):
    sp = pth_path.stem.replace('disease_', '').replace('_resnet50', '')
    with open(MODELS_DIR / f'disease_{sp}_classes.json', 'r') as f:
        d_idx = json.load(f)
    disease_classes[sp] = {v: k for k, v in d_idx.items()}
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(d_idx))
    model.load_state_dict(torch.load(pth_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    disease_models[sp] = model

# ---------- Функции ----------
def classify_species(tensor):
    with torch.no_grad():
        out = species_model(tensor.to(DEVICE))
        prob = torch.softmax(out, dim=1)
        conf, idx = torch.max(prob, dim=1)
    return species_names[idx.item()], conf.item()

def classify_disease(species, tensor):
    if species not in disease_models:
        return "Unknown", 0.0
    model = disease_models[species]
    with torch.no_grad():
        out = model(tensor.to(DEVICE))
        prob = torch.softmax(out, dim=1)
        conf, idx = torch.max(prob, dim=1)
    return disease_classes[species][idx.item()], conf.item()

def detect_leaves(img_path, conf=0.5):
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector(img_rgb, conf=conf, verbose=False)
    leaves = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img_rgb[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            leaves.append({
                'image': Image.fromarray(crop),
                'bbox': (x1, y1, x2, y2),
                'confidence': float(box.conf[0])
            })
    return leaves

def analyze_image(image_path):
    print(f"🔍 Анализ: {image_path}")
    leaves = detect_leaves(image_path)
    print(f"✅ Найдено листьев: {len(leaves)}")
    results = []
    for i, leaf in enumerate(leaves, 1):
        tensor = CLASS_TRANSFORM(leaf['image']).unsqueeze(0)
        species, sp_conf = classify_species(tensor)
        disease, dis_conf = classify_disease(species, tensor)
        print(f"  🍃 Лист #{i}: {species} ({sp_conf:.2f}) → {disease} ({dis_conf:.2f})")
        results.append({
            'species': species,
            'disease': disease,
            'species_conf': sp_conf,
            'disease_conf': dis_conf,
            'bbox': leaf['bbox']
        })
    return results

def visualize(image_path, results, out_path='result_preview.png'):
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for r in results:
        x1, y1, x2, y2 = r['bbox']
        color = (0, 255, 0) if r['disease_conf'] > 0.7 else (0, 165, 255)
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 2)
        label = f"{r['species']}: {r['disease']}"
        cv2.putText(img_rgb, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"💾 Результат сохранён: {out_path}")

# ---------- Запуск ----------
if __name__ == "__main__":
    # Ищем тестовое изображение в dataset/raw_data/TRAIN
    test_img = DATASET_DIR / "TRAIN" / "0.jpg"
    if not test_img.exists():
        # Берём первое попавшееся jpg-изображение из TRAIN
        train_dir = DATASET_DIR / "TRAIN"
        if train_dir.exists():
            test_img = next(train_dir.glob("*.jpg"), None)
    if test_img and test_img.exists():
        results = analyze_image(test_img)
        if results:
            visualize(test_img, results)
    else:
        print("❌ Нет тестового изображения. Положите фото в dataset/raw_data/TRAIN/ или укажите путь вручную.")