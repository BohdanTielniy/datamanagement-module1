import numpy as np
import pandas as pd
from typing import Dict, Optional, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

ArrayLike = Union[np.ndarray, pd.Series, list]

class Evaluator:
    """Клас для розрахунку метрик"""
    
    @staticmethod
    def evaluate_classification(
        y_true: ArrayLike,
        y_pred: ArrayLike,
        y_proba: Optional[ArrayLike] = None
    ) -> Dict[str, float]:
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }

        if y_proba is not None:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            except Exception:
                metrics["roc_auc"] = None

        return metrics