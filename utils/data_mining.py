import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

import warnings

warnings.filterwarnings("ignore")


def create_feature_df(df):
    """
    Crea un DataFrame con características estadísticas agregadas a nivel de usuario, contenido y conjunto de preguntas.

    Este proceso calcula estadísticas como la media, el conteo, la desviación estándar, la mediana y el sesgo
    de las respuestas correctas para cada usuario, contenido y conjunto de preguntas. Luego, estos valores se
    agregan al DataFrame original.

    Args:
        df (pd.DataFrame): El DataFrame original con las respuestas de los estudiantes y las preguntas.

    Returns:
        pd.DataFrame: El DataFrame con las nuevas características agregadas.
    """
    questions_df = df[df["content_type"]=="q"]

    features = [
        "prior_question_elapsed_time",
        "bundle_had_explanation",
        "cumulative_correct_answers",
        "recent_accuracy",
        "cumulative_responses_by_bundle",
        "cumulative_explanations_seen",
        "cumulative_lectures_seen",
    ]
    questions_df= fill_missing_values(questions_df, features)

    grouped_user_df = questions_df.groupby("user_id")

    user_answers_df = grouped_user_df.agg(
        {"answered_correctly": ["mean", "count", "std", "median", "skew"]}
    ).copy()

    user_answers_df.columns = [
        "user_mean",
        "user_count",
        "user_std",
        "user_median",
        "user_skew",
    ]

    grouped_content_df = questions_df.groupby("content_id")

    content_answers_df = grouped_content_df.agg(
        {"answered_correctly": ["mean", "count", "std", "median", "skew"]}
    ).copy()

    content_answers_df.columns = [
        "content_mean",
        "content_count",
        "content_std",
        "content_median",
        "content_skew",
    ]

    grouped_bundle_df = questions_df.groupby("bundle_id")

    bundle_answers_df = grouped_bundle_df.agg(
        {"answered_correctly": ["mean", "count", "std", "median", "skew"]}
    ).copy()

    bundle_answers_df.columns = [
        "bundle_mean",
        "bundle_count",
        "bundle_std",
        "bundle_median",
        "bundle_skew",
    ]

    del grouped_user_df
    del grouped_content_df
    del grouped_bundle_df

    questions_df = questions_df.merge(user_answers_df, how="left", on="user_id")
    questions_df = questions_df.merge(content_answers_df, how="left", on="content_id")
    questions_df = questions_df.merge(bundle_answers_df, how="left", on="bundle_id")

    return questions_df


def to_bool(x):
    """
    Convierte un valor booleano a 0 o 1.

    Args:
        x (bool): El valor booleano a convertir.

    Returns:
        int: 0 si el valor es False, 1 si es True.
    """
    if x is False:
        return 0
    else:
        return 1


def fill_missing_values(df, features, filling_type="mean"):
    """
    Rellena los valores faltantes en el DataFrame con el tipo de imputación especificado.

    Este método gestiona los valores faltantes de las variables numéricas y cualitativas del DataFrame,
    usando técnicas como la imputación por media, mediana o valores aleatorios.

    Args:
        df (pd.DataFrame): El DataFrame con los datos a procesar.
        features (list): Lista de las características a procesar.
        filling_type (str, opcional): El tipo de imputación a usar ("mean", "median", "random"). El valor por defecto es "mean".

    Returns:
        pd.DataFrame: El DataFrame con los valores faltantes imputados.
    """
    if "bundle_had_explanation" in features:
        df["bundle_had_explanation"] = df["bundle_had_explanation"].apply(to_bool)

    df = df.replace([np.inf, -np.inf], np.nan)

    if "prior_question_elapsed_time" in features:
        df["prior_question_elapsed_time"] = df["prior_question_elapsed_time"].fillna(0)

    # Imputaciones
    numerical_vars = df.select_dtypes(
        include=["int", "int32", "int64", "float", "float32", "float64"]
    ).columns
    categorical_vars = [
        variable for variable in features if variable not in numerical_vars
    ]

    for x in numerical_vars:
        df[x] = quantitative_imputation(df[x], filling_type)

    for x in categorical_vars:
        df[x] = qualitative_imputation(df[x], filling_type)

    return df


def scale_data(features_data=None, train=True, features_to_keep=None, target=None):
    """
    Escala los datos proporcionados utilizando StandardScaler.

    Este método estandariza las columnas seleccionadas de los datos, transformando los valores para que tengan media 0
    y desviación estándar 1. En el caso de entrenamiento, mantiene la columna objetivo.

    Args:
        features_data (pd.DataFrame): El DataFrame con los datos a escalar.
        train (bool, opcional): Si es True, se mantiene la columna objetivo en el resultado. El valor por defecto es True.
        features_to_keep (list): Lista de las características a mantener para la escala.
        target (str, opcional): El nombre de la columna objetivo a mantener en los datos escalados.

    Returns:
        pd.DataFrame: El DataFrame con los datos escalados.
    """
    # Seleccionar solo las columnas que queremos escalar
    data_for_standardization = features_data[features_to_keep]

    # Convertir a matriz NumPy
    matrix = data_for_standardization.values
    scaled_matrix = StandardScaler().fit_transform(matrix)

    # Crear un DataFrame escalado
    scaled_data = pd.DataFrame(scaled_matrix, columns=data_for_standardization.columns)

    # Mantener la columna de target si estamos en modo entrenamiento
    if train and target in features_data.columns:
        scaled_data[target] = features_data[target].values

    return scaled_data


def analyze_categorical_variables(data):
    """
    Analiza las variables categóricas de un DataFrame y devuelve un resumen de frecuencia.

    Este método calcula el conteo y el porcentaje de frecuencia de cada valor en las variables categóricas del DataFrame.

    Args:
        data (pd.DataFrame): El DataFrame que contiene las variables a analizar.

    Returns:
        dict: Un diccionario con las variables categóricas como claves y un DataFrame con las frecuencias y porcentajes
              de cada valor en esas variables como valores.
    """
    results = {}

    variables = list(data.columns)

    # Seleccionar las columnas numéricas
    numerical = data.select_dtypes(
        include=["int", "int32", "int64", "float", "float32", "float64"]
    ).columns

    # Seleccionar las columnas categóricas
    categorical = [variable for variable in variables if variable not in numerical]

    # Iterar sobre las variables categóricas
    for category in categorical:
        if category in data.columns:
            summary = pd.DataFrame(
                {
                    "n": data[category].value_counts(),  # Conteo de frecuencias
                    "%": data[category].value_counts(
                        normalize=True
                    ),  # Porcentaje de frecuencias
                }
            )
            results[category] = summary
        else:
            results[category] = None

    return results


def count_distinct(data):
    """
    Cuenta los valores distintos en cada variable numérica de un DataFrame.

    Args:
        data (DataFrame): El DataFrame que contiene los datos.

    Returns:
        DataFrame: Un DataFrame con las variables y valores distintos en cada una de ellas
    """
    numerical = data.select_dtypes(
        include=["int", "int32", "int64", "float", "float32", "float64"]
    )

    distinct_count = numerical.apply(lambda x: len(x.unique()))

    result = pd.DataFrame(
        {"Column": distinct_count.index, "Distinct": distinct_count.values}
    )

    return result


def handle_outliers(varaux):
    """
    Identifica valores atípicos en una serie de datos y los reemplaza por NaN.

    Args:
        varaux (Series): Serie de datos en la que se buscarán valores atípicos.

    Returns:
        Tuple: Una nueva serie con valores atípicos reemplazados por NaN y el número de valores atípicos encontrados.
    """
    if abs(varaux.skew()) < 1:
        # Si es simétrica, calcular valores atípicos basados en desviación estándar
        criterion1 = abs((varaux - varaux.mean()) / varaux.std()) > 3
    else:
        # Si es asimétrica, usar Desviación Absoluta de la Mediana (MAD)
        mad = sm.robust.mad(varaux, axis=0)
        criterion1 = abs((varaux - varaux.median()) / mad) > 8

    qnt = varaux.quantile([0.25, 0.75]).dropna()
    Q1 = qnt.iloc[0]
    Q3 = qnt.iloc[1]
    H = 3 * (Q3 - Q1)

    criterion2 = (varaux < (Q1 - H)) | (varaux > (Q3 + H))

    var = varaux.copy()
    var[criterion1 & criterion2] = np.nan

    return [var, sum(criterion1 & criterion2)]


def quantitative_imputation(var, filling_type):
    """
    Imputa los valores faltantes en una variable cuantitativa.

    Args:
        var (Series): Serie de datos cuantitativos con valores faltantes.
        filling_type (str): Tipo de imputación ('mean', 'median', 'random').

    Returns:
        Series: Serie con valores imputados.
    """
    vv = var.copy()

    if filling_type == "mean":
        vv[np.isnan(vv)] = round(np.nanmean(vv), 4)
    elif filling_type == "median":
        vv[np.isnan(vv)] = round(np.nanmedian(vv), 4)
    elif filling_type == "random":
        x = vv[~np.isnan(vv)]
        frec = x.value_counts(normalize=True).reset_index()
        frec.columns = ["Value", "Freq"]
        frec = frec.sort_values(by="Value")
        frec["FreqAcum"] = frec["Freq"].cumsum()
        random_values = np.random.uniform(
            min(frec["FreqAcum"]), 1, np.sum(np.isnan(vv))
        )
        imputed_values = list(
            map(lambda x: list(frec["Value"][frec["FreqAcum"] <= x])[-1], random_values)
        )
        vv[np.isnan(vv)] = [round(x, 4) for x in imputed_values]

    return vv


def qualitative_imputation(var, filling_type):
    """
    Imputa los valores faltantes en una variable cualitativa.

    Args:
        var (Series): Serie de datos cualitativos con valores faltantes.
        filling_type (str): Tipo de imputación ('mode' o 'random').

    Returns:
        Series: Serie con valores imputados.
    """
    vv = var.copy()

    if filling_type == "mode":
        frecuencies = vv[~vv.isna()].value_counts()
        mode = frecuencies.index[np.argmax(frecuencies)]
        vv[vv.isna()] = mode
    elif filling_type == "random":
        non_missing_values = vv.dropna().sample(n=vv.isna().sum(), replace=True)
        vv[vv.isna()] = non_missing_values.values

    return vv
