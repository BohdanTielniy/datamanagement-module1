import json
import joblib
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Union
from matplotlib.figure import Figure

class IOHandler:
    """Клас для роботи з файловою системою"""
    
    @staticmethod
    def ensure_dirs(*paths: Union[str, Path]) -> None:
        """Створює всі передані директорії, якщо їх не існує"""
        for p in paths:
            Path(p).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def load_csv(path: Union[str, Path]) -> pd.DataFrame:
        """Завантажує дані з CSV файлу у DataFrame"""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Файл не знайдено: {file_path}")
        return pd.read_csv(file_path)

    @staticmethod
    def save_csv(df: pd.DataFrame, path: Union[str, Path]) -> None:
        """Зберігає DataFrame у CSV файл"""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path, index=False)

    @staticmethod
    def save_json(data: Dict[str, Any], path: Union[str, Path]) -> None:
        """Зберігає словник у форматі JSON"""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    @staticmethod
    def save_text(text: str, path: Union[str, Path]) -> None:
        """Зберігає текст у файл"""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(text, encoding="utf-8")

    @staticmethod
    def save_joblib(obj: Any, path: Union[str, Path]) -> None:
        """Зберігає Python-об'єкт за допомогою joblib"""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, file_path)

    @staticmethod
    def save_figure(figure: Figure, path: Union[str, Path]) -> None:
        """Зберігає фігуру matplotlib.figure, як картинку png"""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path)