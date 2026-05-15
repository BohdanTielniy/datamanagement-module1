from typing import Any, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class ModelFactory:
    """Клас для створення моделей машинного навчання на основі конфігурації"""
    @staticmethod
    def build_model(task: str, name: str, params: Dict[str, Any]):
        """Ініціалізує модель та передає їй параметри"""
        if task != "classification":
            raise ValueError(f"Підтримується лише classification, отримано: {task}")

        seed = params.pop("seed", 42)

        if name == "rf":
            return RandomForestClassifier(
                random_state=seed, 
                n_jobs=-1,
                **params
            )
        elif name == "logreg":
            return LogisticRegression(
                random_state=seed,
                **params
            )
        else:
            raise ValueError(f"Невідома модель: {name}")