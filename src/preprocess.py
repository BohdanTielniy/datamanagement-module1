import pandas as pd
import numpy as np
from typing import List
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class DataPreprocessor:
    """Клас для очищення даних та побудови sklearn-пайплайну для препроцесингу"""
    
    def __init__(self, num_cols: List[str], cat_cols: List[str]):
        self.num_cols = num_cols
        self.cat_cols = cat_cols

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(" ", np.nan))
            
        return df

    def build_transformer(self) -> ColumnTransformer:
        # Пайплайн для числових даних: 
        # 1. Заповнюємо пропуски (NaN) медіаною.
        # 2. Масштабуємо (робимо середнє 0, а стандартне відхилення 1).
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        # Пайплайн для категоріальних даних:
        # 1. Заповнюємо пропуски найчастішим значенням.
        # 2. Перетворюємо категорії у бінарні колонки (One-Hot). 
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        # Об'єднуємо обидва пайплайни
        preprocessor = ColumnTransformer([
            ("num", num_pipe, self.num_cols),
            ("cat", cat_pipe, self.cat_cols)
        ])

        return preprocessor