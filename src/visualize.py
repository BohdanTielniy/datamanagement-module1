import pandas as pd
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

class Visualizer():
    @staticmethod
    def confusion_matrix_roc_curve(predictions: pd.DataFrame) -> Figure:
        """Створює графіки на основі збережених прогнозів."""
        
        y_true = predictions['y_true']
        y_pred = predictions['y_pred']
        y_proba = predictions['y_proba']

        # Створюємо полотно для двох графіків
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
        # Матриця помилок
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, 
            ax=ax[0], 
            cmap='Blues', 
            display_labels=['No Churn (0)', 'Churn (1)']
        )
        ax[0].set_title('Матриця помилок')

        # ROC-крива
        RocCurveDisplay.from_predictions(
            y_true, y_proba, 
            ax=ax[1], 
            curve_kwargs={'color': 'darkorange'}
        )
        ax[1].set_title('ROC-крива')
        ax[1].plot([0, 1], [0, 1], linestyle='--', color='gray')

        plt.tight_layout()
        
        return fig