# Telco Customer Churn Pipeline

Мета - побудова відтворюваного пайплайну для класифікації відтоку клієнтів.

## Структура проєкту

```
project
├─ README.md
├─ artifacts
├─ configs
│  └─ config.yaml
├─ data
│  ├─ processed
│  └─ raw
│     └─ WA_Fn-UseC_-Telco-Customer-Churn.csv
├─ logs
├─ requirements.txt
├─ run.py
└─ src
   ├─ __init__.py
   ├─ config.py
   ├─ evaluate.py
   ├─ io.py
   ├─ logger.py
   ├─ pipeline.py
   ├─ preprocess.py
   ├─ train.py
   └─ visualize.py
```

- `configs/` - YAML файли з налаштуваннями (шляхи, гіперпараметри).
- `data/raw/` — вихідні дані.
- `src/` — вихідний код.
- `run.py` — точка входу.

## Як запустити

1. **Створіть та активуйте віртуальне середовище:**
```bash
python -m venv .venv

# Для Windows:
.venv\Scripts\activate

# Для Linux/Mac:
source .venv/bin/activate
```
2. **Встановіть залежності**
```bash
pip install -r requirements.txt
```
3. **Запустіть пайплайн**
```bash
python run.py --config configs/config.yaml
```

## Результати

Після запуску всі артефакти будуть збережені у папці `artifacts/run_<timestamp>/`:

- `models/model.joblib` - навчена модель з препроцесингом.

- `metrics/metrics.json` - результати оцінки (Accuracy, F1, ROC-AUC тощо).

- `predictions/predictions.csv` - прогнози моделі.

- `reports/config_snapshot.yaml` - копія конфігу для відтворюваності.

- `plots/confusion_matrix_roc_curve.png` - графіки з візуалізацією матриці помилок та ROC-крива