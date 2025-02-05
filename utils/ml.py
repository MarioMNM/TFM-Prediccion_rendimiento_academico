import pickle
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
)
from enum import Enum
import os

import warnings

warnings.filterwarnings("ignore")


# Enum para los nombres de los archivos de los modelos
class ModelFiles(Enum):
    DT = "decision_tree.pkl"
    RF = "random_forest.pkl"
    XGB = "xgboost.pkl"
    LGBM = "lightgbm.pkl"


def load_ml_model(model_name, ml_dir):
    """
    Carga un modelo preentrenado desde un archivo pickle.

    Args:
        model_name (str): Nombre del modelo (DT, RF, XGB, LGBM).
        ml_dir (str): Directorio donde se encuentra el modelo.

    Returns:
        model: El modelo cargado.

    Raises:
        ValueError: Si el nombre del modelo no es válido.
        FileNotFoundError: Si el archivo del modelo no se encuentra en la ruta especificada.
        RuntimeError: Si ocurre algún otro error al cargar el modelo.
    """
    # Verificar que el nombre del modelo sea un miembro del Enum
    try:
        model_file = ModelFiles[model_name.upper()].value
    except KeyError:
        valid_models = [e.name.lower() for e in ModelFiles]
        raise ValueError(
            f"El modelo '{model_name}' no es válido. Debe ser uno de: {valid_models}"
        )

    # Construir la ruta completa del archivo
    model_path = os.path.join(ml_dir, model_file)

    # Intentar cargar el modelo
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"No se encontró el archivo del modelo: {model_path}")
    except Exception as e:
        raise RuntimeError(f"Error al cargar el modelo desde {model_path}: {e}")


def model_results(grid_search, plot: bool = True):
    """
    Muestra los resultados del GridSearchCV, mostrando los mejores resultados y generando un gráfico de boxplots.

    Args:
        grid_search (GridSearchCV): Objeto GridSearchCV que contiene los resultados de la búsqueda.
        plot (bool): Si es True, genera un gráfico de boxplot con los resultados. Default es True.

    Returns:
        pd.DataFrame: Los resultados ordenados por el ranking de test score.
    """
    results = pd.DataFrame(grid_search.cv_results_)

    # Ordenar por el ranking de los mejores resultados
    sorted_results = results.sort_values(by="rank_test_score", ascending=True).head(5)

    # Determinar el número de splits dinámicamente
    num_splits = len(
        [
            col
            for col in results.columns
            if col.startswith("split") and "test_score" in col
        ]
    )

    # Extraer las columnas correspondientes a los splits
    split_columns = [f"split{i}_test_score" for i in range(num_splits)]

    # Preparar los resultados para los boxplots
    res_list = [
        sorted_results[split_columns].iloc[i].values for i in range(len(sorted_results))
    ]

    # Graficar si está habilitado
    if plot:
        plt.boxplot(
            res_list,
            labels=[f"res_{i+1}" for i in range(len(res_list))],
        )
        plt.title("Boxplots de AUC para los CV Splits")
        plt.xlabel("Resultados ordenados por ranking")
        plt.ylabel("Métrica de evaluación")
        plt.show()

    return sorted_results


def goodness_of_fit(model, X, y):
    """
    Muestra la matriz de confusión y el reporte de clasificación para evaluar el desempeño del modelo.

    Args:
        model: El modelo entrenado que se evaluará.
        X (pd.DataFrame or np.ndarray): Datos de entrada para hacer predicciones.
        y (pd.DataFrame or np.ndarray): Etiquetas verdaderas.

    Prints:
        Matriz de confusión y medidas de desempeño (precisión, recall, f1-score, etc.).
    """
    y_test_pred = model.predict(X)

    conf_matrix = confusion_matrix(y, y_test_pred)
    print("Matriz de Confusión:")
    print(conf_matrix)
    print("\nMedidas de Desempeño:")
    print(classification_report(y, y_test_pred))


def plot_roc_curve(model, X, y):
    """
    Dibuja la curva ROC y calcula el AUC para evaluar el desempeño del modelo.

    Args:
        model: El modelo entrenado que se evaluará.
        X (pd.DataFrame or np.ndarray): Datos de entrada para hacer predicciones.
        y (pd.DataFrame or np.ndarray): Etiquetas verdaderas.
    """
    y_prob_test = model.predict_proba(X)[:, 1]

    fpr, tpr, thresholds = roc_curve(y, y_prob_test)
    roc_auc = auc(fpr, tpr)
    print(f"\nÁrea bajo la curva ROC (AUC): {roc_auc:.5f}")

    # Graficar la curva ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.5f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("Tasa de Falsos Positivos (FPR)")
    plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
    plt.title("Curva ROC")
    plt.legend(loc="lower right")
    plt.show()


def plot_features_importance(model):
    """
    Muestra un gráfico de barras con la importancia de las características del modelo.

    Args:
        model: El modelo entrenado que tiene las características y su importancia.
    """
    df_importancia_c = pd.DataFrame(
        {
            "Variable": model.feature_names_in_,
            "Importancia": model.feature_importances_,
        }
    ).sort_values(by="Importancia", ascending=False)

    plt.bar(
        df_importancia_c["Variable"], df_importancia_c["Importancia"], color="skyblue"
    )
    plt.xlabel("Variable")
    plt.ylabel("Importancia")
    plt.title("Importancia de las características")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.show()
