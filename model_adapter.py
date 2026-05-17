from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import cv2
import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------- Словарь перевода болезней ----------
DISEASE_TRANSLATIONS = {
    "Apple Scab Leaf": "Парша яблони",
    "Apple leaf": "Здоровый лист яблони",
    "Apple rust leaf": "Ржавчина яблони",
    "Bell_pepper leaf spot": "Пятнистость листьев болгарского перца",
    "Bell_pepper leaf": "Здоровый лист болгарского перца",
    "Blueberry leaf": "Здоровый лист черники",
    "Cherry leaf": "Здоровый лист вишни",
    "Corn Gray leaf spot": "Серая пятнистость кукурузы",
    "Corn leaf blight": "Фитофтороз кукурузы",
    "Corn rust leaf": "Ржавчина кукурузы",
    "Peach leaf": "Здоровый лист персика",
    "Potato leaf early blight": "Ранняя гниль картофеля",
    "Potato leaf late blight": "Фитофтороз картофеля",
    "Potato leaf": "Здоровый лист картофеля",
    "Raspberry leaf": "Здоровый лист малины",
    "Soyabean leaf": "Здоровый лист сои",
    "Squash Powdery mildew leaf": "Мучнистая роса тыквы",
    "Strawberry leaf": "Здоровый лист клубники",
    "Tomato Early blight leaf": "Ранняя гниль томата",
    "Tomato Septoria leaf spot": "Септориоз томата",
    "Tomato leaf bacterial spot": "Бактериальная пятнистость томата",
    "Tomato leaf late blight": "Фитофтороз томата",
    "Tomato leaf mosaic virus": "Мозаика томата",
    "Tomato leaf yellow virus": "Жёлтая мозаика томата",
    "Tomato leaf": "Здоровый лист томата",
    "Tomato mold leaf": "Плесень томата",
    "Tomato two spotted spider mites leaf": "Паутинный клещ томата",
    "grape leaf black rot": "Чёрная гниль винограда",
    "grape leaf": "Здоровый лист винограда",
    "Bell_pepper leaf spot": "Пятнистость листьев болгарского перца",
}
# ----------------------------------------------


class ModelAdapter:
    """
    Адаптер, преобразующий сырые результаты моделей в унифицированный формат.
    Использует паттерн Adapter для разделения ответственности:
        - загрузка/хранение моделей (модели),
        - логика пред- и постобработки (адаптер),
        - HTTP-представление (маршруты Flask).
    """

    def __init__(self, detector, species_model, species_names,
                 disease_models, disease_classes, transform):
        self.detector = detector
        self.species_model = species_model
        self.species_names = species_names
        self.disease_models = disease_models
        self.disease_classes = disease_classes
        self.transform = transform
        self.device = next(species_model.parameters()).device

    def _is_healthy(self, species: str, original_disease: str) -> bool:
        """
        Определяет, является ли предсказанный класс болезни «здоровым» для данного вида.
        Работает с оригинальным (английским) названием болезни.
        """
        healthy_pattern = f"{species} leaf"
        if original_disease == healthy_pattern:
            return True
        if 'healthy' in original_disease.lower():
            return True
        return False

    def predict(self, image_path: Path, conf=0.5) -> dict:
        """
        Основной метод: принимает путь к изображению, возвращает адаптированный ответ.
        """
        # --- Шаг 1: Детекция листьев ---
        leaves, img_rgb = self._detect_leaves(image_path, conf)

        if not leaves:
            return {
                'error': 'Листья не обнаружены',
                'results': [],
                'summary': {'total': 0, 'diseased': 0, 'healthy': 0}
            }

        # --- Шаг 2: Классификация каждого листа ---
        h, w, _ = img_rgb.shape
        results = []
        for leaf in leaves:
            tensor = self.transform(leaf['image']).unsqueeze(0).to(self.device)
            species, species_conf = self._classify_species(tensor)
            original_disease, disease_conf = self._classify_disease(species, tensor)

            # Определяем здоров/болен по оригинальному названию
            is_healthy = self._is_healthy(species, original_disease)

            # Переводим болезнь на русский
            disease_ru = DISEASE_TRANSLATIONS.get(original_disease, original_disease)

            x1, y1, x2, y2 = leaf['bbox']
            results.append({
                'species': species,
                'disease': disease_ru,          # Теперь на русском
                'species_conf': float(species_conf),
                'disease_conf': float(disease_conf),
                'bbox_norm': [x1/w, y1/h, x2/w, y2/h],
                'bbox_abs': [x1, y1, x2, y2],
                'det_conf': leaf['confidence'],
                'is_healthy': is_healthy
            })

        # --- Шаг 3: Отрисовка bounding boxes ---
        result_img = self._draw_boxes(img_rgb, results)

        # --- Шаг 4: Составление статистики ---
        diseased = sum(1 for r in results if not r['is_healthy'])
        healthy = len(results) - diseased

        return {
            'error': None,
            'results': results,
            'annotated_image': result_img,
            'summary': {
                'total': len(results),
                'diseased': diseased,
                'healthy': healthy
            }
        }

    # ---------- приватные методы ----------
    def _detect_leaves(self, image_path, conf):
        img = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        det_results = self.detector(img_rgb, conf=conf, verbose=False)
        leaves = []
        for r in det_results:
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
        logger.info(f"Детекция: найдено {len(leaves)} листьев")
        return leaves, img_rgb

    def _classify_species(self, tensor):
        with torch.no_grad():
            out = self.species_model(tensor)
            prob = torch.softmax(out, dim=1)
            conf, idx = torch.max(prob, dim=1)
        return self.species_names[idx.item()], conf.item()

    def _classify_disease(self, species, tensor):
        if species not in self.disease_models:
            return "Unknown", 0.0
        model = self.disease_models[species]
        with torch.no_grad():
            out = model(tensor)
            prob = torch.softmax(out, dim=1)
            conf, idx = torch.max(prob, dim=1)
        return self.disease_classes[species][idx.item()], conf.item()

    def _draw_boxes(self, img_rgb, results):
        img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        for r in results:
            x1, y1, x2, y2 = r['bbox_abs']
            color = (0, 255, 0) if r['is_healthy'] else (255, 165, 0)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            # Надпись на русском
            label = f"{r['species']}: {r['disease']}"
            draw.text((x1, y1-10), label, fill=color, font=font)
        return img