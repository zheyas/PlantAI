
import os
from pathlib import Path
from flask import Flask, render_template, request
from PIL import Image, ImageDraw, ImageFont
import cv2
import torch
import numpy as np
import json
from ultralytics import YOLO
from torchvision import transforms, models
import logging

from model_adapter import ModelAdapter

from dotenv import load_dotenv
load_dotenv()

from download_models import main as download_models
download_models()   # выполнится только если моделей ещё нет локально


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------- ИНИЦИАЛИЗАЦИЯ ----------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / 'ml_models'
STATIC_DIR = BASE_DIR / 'static'
UPLOAD_FOLDER = STATIC_DIR / 'uploads'
RESULT_FOLDER = STATIC_DIR / 'results'
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

CLASS_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

logger.info("Загрузка моделей...")
detector = YOLO(str(MODELS_DIR / 'leaf_detector.pt'))

with open(MODELS_DIR / 'species_classes.json', 'r') as f:
    species_idx = json.load(f)
species_names = {v: k for k, v in species_idx.items()}
num_species = len(species_names)
species_model = models.efficientnet_b0(weights=None)
species_model.classifier[1] = torch.nn.Linear(1280, num_species)
species_model.load_state_dict(torch.load(MODELS_DIR / 'species_model.pth', map_location=DEVICE))
species_model.to(DEVICE)
species_model.eval()
logger.info("Классификатор вида загружен")

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
logger.info(f"Загружены модели болезней для {len(disease_models)} видов")

# ---------- ФУНКЦИИ ПАЙПЛАЙНА ----------
def detect_leaves(img_path, conf=0.5):
    logger.info(f"Детекция на {img_path}")
    img = cv2.imread(str(img_path))
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
    logger.info(f"Найдено {len(leaves)} листьев")
    return leaves, img_rgb

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

def draw_results(img_rgb, results):
    img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    for r in results:
        x1, y1, x2, y2 = r['bbox']
        color = (0, 255, 0) if r['disease_conf'] > 0.7 else (255, 165, 0)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{r['species']}: {r['disease']}"
        draw.text((x1, y1-10), label, fill=color, font=font)
    return img

def process_image_and_get_results(image_path):
    logger.info(f"Начало обработки: {image_path}")
    leaves, img_rgb = detect_leaves(image_path, conf=0.5)
    if not leaves:
        logger.warning("Листья не обнаружены")
        return None, None, img_rgb

    h, w, _ = img_rgb.shape
    results = []
    for leaf in leaves:
        tensor = CLASS_TRANSFORM(leaf['image']).unsqueeze(0)
        species, sp_conf = classify_species(tensor)
        disease, dis_conf = classify_disease(species, tensor)
        x1, y1, x2, y2 = leaf['bbox']
        # Добавляем оба представления координат
        results.append({
            'species': species,
            'disease': disease,
            'species_conf': sp_conf,
            'disease_conf': dis_conf,
            'bbox_norm': [x1/w, y1/h, x2/w, y2/h],   # для фронтенда
            'bbox_abs': [x1, y1, x2, y2],            # абсолютные
            'bbox': [x1, y1, x2, y2],                # для draw_results (чтобы не сломать старый код)
            'det_conf': leaf['confidence']
        })
    logger.info(f"Успешно обработано {len(results)} листьев")
    result_img = draw_results(img_rgb, results)   # здесь используется 'bbox'
    return result_img, results, img_rgb

# ---------- FLASK ----------
app = Flask(__name__)

# Где-то после загрузки всех моделей:
adapter = ModelAdapter(
    detector=detector,
    species_model=species_model,
    species_names=species_names,
    disease_models=disease_models,
    disease_classes=disease_classes,
    transform=CLASS_TRANSFORM
)

@app.route('/', methods=['GET', 'POST'])
def index():
    uploaded_image = None
    result_image = None
    results = None
    error = None

    if request.method == 'POST':
        if 'file' not in request.files:
            error = "Файл не выбран"
        else:
            file = request.files['file']
            if file.filename == '':
                error = "Файл не выбран"
            else:
                filename = file.filename
                upload_path = UPLOAD_FOLDER / filename
                file.save(str(upload_path))

                # Используем адаптер
                response = adapter.predict(upload_path)
                error = response.get('error')
                if not error:
                    # Сохраняем размеченное изображение
                    result_filename = f"result_{upload_path.stem}.png"
                    result_path = RESULT_FOLDER / result_filename
                    response['annotated_image'].save(str(result_path))
                    uploaded_image = f'static/uploads/{filename}'
                    result_image = f'static/results/{result_filename}'
                    results = response['results']

    summary = None
    if results is not None:
        diseased = sum(1 for r in results if r['disease_conf'] < 0.7)
        healthy = len(results) - diseased
        summary = {
            'total': len(results),
            'diseased': diseased,
            'healthy': healthy
        }

    return render_template('index.html',
                           uploaded_image=uploaded_image,
                           result_image=result_image,
                           results=results,
                           summary=summary,
                           error=error)

if __name__ == '__main__':
    app.run(debug=True)