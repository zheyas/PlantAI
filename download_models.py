"""
download_models.py – загружает папку ml_models с Яндекс.Диска.
Токен читается из переменной окружения YANDEX_DISK_TOKEN.
"""
import os
import yadisk
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

YANDEX_TOKEN = os.getenv("YANDEX_DISK_TOKEN")
REMOTE_DIR = "/PlantAI_models"
LOCAL_DIR = Path(__file__).parent / "ml_models"

def download_folder(client, remote_path: str, local_path: Path):
    local_path.mkdir(parents=True, exist_ok=True)

    for item in client.listdir(remote_path):
        remote_item = f"{remote_path}/{item.name}"
        local_item = local_path / item.name

        if item.type == 'dir':
            logger.info(f"📁 Вход в папку: {remote_item}")
            download_folder(client, remote_item, local_item)
        else:
            if local_item.exists():
                logger.info(f"⏭️  Файл уже существует: {item.name}")
                continue
            size_info = getattr(item, 'size', '?')
            logger.info(f"⬇️  Скачивание: {item.name} ({size_info} байт)")
            client.download(remote_item, str(local_item))

def main():
    if not YANDEX_TOKEN:
        logger.error("❌ Не задан YANDEX_DISK_TOKEN в .env или переменных окружения")
        return

    if LOCAL_DIR.exists() and any(LOCAL_DIR.iterdir()):
        logger.info("✅ Модели уже загружены локально. Пропускаем.")
        return

    logger.info("🌐 Подключение к Яндекс.Диску...")
    client = yadisk.Client(token=YANDEX_TOKEN)

    if not client.check_token():
        logger.error("❌ Токен недействителен. Проверьте его.")
        return

    logger.info(f"📦 Скачивание папки '{REMOTE_DIR}' в '{LOCAL_DIR}'...")
    download_folder(client, REMOTE_DIR, LOCAL_DIR)
    logger.info("🏁 Загрузка завершена!")

if __name__ == "__main__":
    main()