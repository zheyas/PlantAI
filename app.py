import gradio as gr
from pathlib import Path
import torch
from torchvision import transforms
from ultralytics import YOLO
import logging
from huggingface_hub import snapshot_download
from model_adapter import ModelAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = Path("ml_models")

# Если папка с моделями отсутствует, скачиваем её из Hugging Face Hub
if not MODELS_DIR.exists():
    logger.info("Скачивание моделей из zheyas/plantai-models...")
    snapshot_download(repo_id="zheyas/plantai-models", local_dir=str(MODELS_DIR), repo_type="model")
    logger.info("Модели загружены.")

device = torch.device("cpu")
detector = YOLO(str(MODELS_DIR / "leaf_detector.pt"))

# Загружаем классификатор вида
species_model = torch.load(MODELS_DIR / "species_model.pth", map_location=device)
species_model.eval()

# Загружаем классификаторы болезней (только .pth файлы)
disease_models = {}
for pth_path in MODELS_DIR.glob("disease_*_resnet50.pth"):
    sp = pth_path.stem.replace("disease_", "").replace("_resnet50", "")
    disease_models[sp] = torch.load(pth_path, map_location=device)
    disease_models[sp].eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

adapter = ModelAdapter(
    detector=detector,
    species_model=species_model,
    disease_models=disease_models,
    transform=transform
)

def predict(image):
    tmp_path = Path("temp_upload.jpg")
    image.save(tmp_path)
    result = adapter.predict(tmp_path)
    if result["error"]:
        return None, result["error"]
    annotated = result["annotated_image"]
    lines = []
    for r in result["results"]:
        status = "ЗДОРОВ" if r["is_healthy"] else "БОЛЕН"
        lines.append(f"{r['species']}: {r['disease']} ({status}, уверенность {r['disease_conf']:.0%})")
    return annotated, "\n".join(lines)

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Загрузите фото листа"),
    outputs=[
        gr.Image(label="Результат"),
        gr.Textbox(label="Диагноз")
    ],
    title="PlantAI — Диагностика растений",
    description="Загрузите фото листа, и нейросеть определит вид растения и болезнь."
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0")