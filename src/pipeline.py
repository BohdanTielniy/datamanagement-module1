import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import set_config
from sklearn.utils import estimator_html_repr

from .config import ConfigManager
from .logger import ProjectLogger
from .io import IOHandler
from .preprocess import DataPreprocessor
from .train import ModelFactory
from .evaluate import Evaluator
from .visualize import Visualizer

class PipelineRunner:
    """Клас, який керує процесом машинного навчання"""
    
    def __init__(self, config_path: str | Path):
        self.config_path = config_path
        self.cfg = ConfigManager(config_path)
        
        # Налаштовуємо ID запуску
        run_id_mode = self.cfg.get("project", "run_id_mode", default="timestamp")
        self.run_id = time.strftime("%Y%m%d_%H%M%S") if run_id_mode == "timestamp" else self.cfg.get("project", "run_id", default="run_local")
        
        # Ініціалізуємо логер
        logs_dir = self.cfg.get("output", "logs_dir", default="logs")
        self.logger = ProjectLogger(logs_dir, self.run_id).get_logger()
        
        # Шлях до папки з результатами цього запуску
        self.artifacts_dir = Path(self.cfg.get("output", "artifacts_dir", default="artifacts")) / f"run_{self.run_id}"

    def run(self):
        self.logger.info(f"Run ID: {self.run_id}")
        
        # Завантаження даних
        input_csv = self.cfg.get("data", "input_csv")
        target_col = self.cfg.get("data", "target")
        drop_cols = self.cfg.get("data", "drop_cols", default=[])
        
        self.logger.info(f"Завантаження даних з {input_csv}")
        df = IOHandler.load_csv(input_csv)
        
        if drop_cols:
            df = df.drop(columns=drop_cols, errors="ignore")
            
        # Очищення та препроцесинг
        self.logger.info("Попередня обробка даних...")
        preprocessor = DataPreprocessor(
            num_cols=self.cfg.get("data", "num_cols", default=[]),
            cat_cols=self.cfg.get("data", "cat_cols", default=[])
        )
        df_cleaned = preprocessor.clean_dataframe(df)
        
        X = df_cleaned.drop(columns=[target_col])
        y = df_cleaned[target_col]
        
        if y.isin(['Yes', 'No']).any():
            y = y.replace({'Yes': 1, 'No': 0}).astype(int)
            
        # Розбиття на train/test
        test_size = self.cfg.get("data", "test_size", default=0.2)
        seed = self.cfg.get("project", "seed", default=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        
        # Створення моделі
        task = self.cfg.get("model", "task")
        model_name = self.cfg.get("model", "name")
        params = self.cfg.get("model", "params", default={})
        params["seed"] = seed
        
        classifier = ModelFactory.build_model(task, model_name, params)
        
        # Об'єднуємо обробку ознак та класифікатор у єдиний Pipeline від scikit-learn
        full_pipeline = Pipeline([
            ("preprocessor", preprocessor.build_transformer()),
            ("classifier", classifier)
        ])
        
        # Навчання
        self.logger.info(f"Навчання моделі {model_name}")
        full_pipeline.fit(X_train, y_train)
        
        # Оцінка
        self.logger.info("Оцінка моделі на тестовій вибірці")
        y_pred = full_pipeline.predict(X_test)
        y_proba = full_pipeline.predict_proba(X_test)[:, 1] if hasattr(full_pipeline, "predict_proba") else None
        
        metrics = Evaluator.evaluate_classification(y_test, y_pred, y_proba)
        self.logger.info(f"Отримані метрики: {metrics}")
        
        # Збереження артефактів
        self.logger.info(f"Збереження результатів у {self.artifacts_dir}")
        
        if self.cfg.get("output", "save_model"):
            IOHandler.save_joblib(full_pipeline, self.artifacts_dir / "models" / "model.joblib")
            
        IOHandler.save_json(metrics, self.artifacts_dir / "metrics" / "metrics.json")
        
        if self.cfg.get("output", "save_predictions"):
            pred_df = X_test.copy()
            pred_df["y_true"] = y_test.values
            pred_df["y_pred"] = y_pred
            if y_proba is not None:
                pred_df["y_proba"] = y_proba
            IOHandler.save_csv(pred_df, self.artifacts_dir / "predictions" / "predictions.csv")
            
        with open(self.config_path, "r", encoding="utf-8") as f:
            IOHandler.save_text(f.read(), self.artifacts_dir / "reports" / "config_snapshot.yaml")
        
        # Візуалізація результатів
        visualization = Visualizer.confusion_matrix_roc_curve(pred_df)
        IOHandler.save_figure(visualization, self.artifacts_dir / "plots" / "confusion_matrix_roc_curve.png")

        self.logger.info("Пайплайн завершено!")