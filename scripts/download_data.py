import os
import zipfile
import json
from pathlib import Path
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi

load_dotenv()

def download_disaster_dataset():
    # --- 1. СОЗДАЕМ ФАЙЛ ТОКЕНА ВРУЧНУЮ ---
    token = os.getenv('KAGGLE_API_TOKEN')
    
    if not token:
        print("❌ ОШИБКА: KAGGLE_API_TOKEN не найден в .env")
        print("Получи токен: https://www.kaggle.com/settings/api")
        return
    
    # Создаем папку .kaggle, если нет
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    
    # Сохраняем токен в access_token
    token_path = kaggle_dir / "access_token"
    with open(token_path, "w") as f:
        f.write(token)
    
    # Права доступа (для Windows не критично)
    try:
        os.chmod(token_path, 0o600)
    except:
        pass
    
    # --- 2. АВТОРИЗАЦИЯ ---
    api = KaggleApi()
    api.authenticate()  # Теперь найдет access_token
    
    # --- 3. СКАЧИВАНИЕ ---
    Path("data").mkdir(exist_ok=True)
    print("📥 Скачивание датасета...")
    
    api.competition_download_files(
        competition="nlp-getting-started",
        path="data/"
    )
    
    # Распаковка
    zip_path = "data/nlp-getting-started.zip"
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall("data/")
    os.remove(zip_path)
    
    print("✅ Датасет скачан в data/train.csv")

if __name__ == "__main__":
    download_disaster_dataset()