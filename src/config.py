import yaml
from pathlib import Path
from typing import Any, Dict

class ConfigManager:
    """Клас для завантаження та управління налаштуваннями проєкту."""
    
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self.config_data: Dict[str, Any] = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Файл конфігурації не знайдено: {self.config_path}")
        
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Отримує значення з вкладеного словника конфігурації.\n
        Приклад: config.get("model", "params", "max_depth")
        """
        current_level = self.config_data
        for key in keys:
            if isinstance(current_level, dict) and key in current_level:
                current_level = current_level[key]
            else:
                return default
        return current_level