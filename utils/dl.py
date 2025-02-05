import os
import pickle

import pandas as pd
import matplotlib.pyplot as plt

# Redes neuronales
from keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Dropout,
    BatchNormalization,
    Flatten,
    LSTM,
    Input,
    Dense,
    GaussianNoise,
    Conv1D,
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
)
from tensorflow.keras import models
import keras_tuner as kt
import tensorflow as tf


import warnings

warnings.filterwarnings("ignore")

# Configura el nivel de logging para que solo se muestren los errores
tf.get_logger().setLevel("ERROR")


def reshape_lstm_data(df):
    """
    Redimensiona los datos para usarlos con una red neuronal LSTM.

    Args:
    df (pandas.DataFrame): Datos a redimensionar.

    Returns:
    numpy.ndarray: Datos redimensionados para LSTM.
    """
    data = df.copy()
    data = data.values if isinstance(data, pd.DataFrame) else data
    data = data.reshape(data.shape[0], 1, data.shape[1])
    return data


def reshape_cnn_data(df):
    """
    Redimensiona los datos para usarlos con una red neuronal CNN.

    Args:
    df (pandas.DataFrame): Datos a redimensionar.

    Returns:
    numpy.ndarray: Datos redimensionados para CNN.
    """
    data = df.copy()
    data = data.values if isinstance(data, pd.DataFrame) else data
    data = data.reshape(data.shape[0], data.shape[1], 1)
    return data


def build_keras_lr_model(X_train):
    """
    Construye un modelo de red neuronal densa para clasificación binaria.

    Args:
    X_train (numpy.ndarray): Datos de entrenamiento para determinar la forma de entrada.

    Returns:
    keras.Model: Modelo de red neuronal construido.
    """
    inp = Input(shape=(X_train.shape[1],), name="inp")
    x = Dense(512, activation="relu")(inp)
    x = Dropout(0.08)(x)
    x = GaussianNoise(0.01)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.08)(x)
    x = GaussianNoise(0.01)(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.08)(x)
    x = GaussianNoise(0.01)(x)
    x = BatchNormalization()(x)
    preds = Dense(1, activation="sigmoid", name="out")(x)
    model = models.Model(inp, preds)

    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05)
    model.compile(loss=loss, optimizer="adam", metrics=[tf.keras.metrics.AUC()])
    print(model.summary())

    return model


def get_keras_lr_model(
    X_train,
    y_train,
    X_test,
    y_test,
    dl_dir,
    model_name="logistic_regresion",
    load_trained_model=True,
):
    """
    Obtiene un modelo de regresión logística entrenado, cargándolo desde disco si ya está guardado,
    o entrenándolo desde cero.

    Args:
    X_train (numpy.ndarray): Datos de entrenamiento.
    y_train (numpy.ndarray): Etiquetas de entrenamiento.
    X_test (numpy.ndarray): Datos de prueba.
    y_test (numpy.ndarray): Etiquetas de prueba.
    dl_dir (str): Directorio donde guardar los modelos y resultados.
    model_name (str): Nombre del modelo.
    load_trained_model (bool): Si es True, carga el modelo entrenado desde disco.

    Returns:
    model: El modelo entrenado.
    history_json (dict): Historial del entrenamiento.
    """
    models_dir = dl_dir + "model"
    history_dir = dl_dir + "history"

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(history_dir, exist_ok=True)

    model_path = os.path.join(models_dir, model_name + ".h5")
    history_path = os.path.join(history_dir, model_name + ".pkl")

    if (
        os.path.exists(model_path)
        and os.path.exists(history_path)
        and load_trained_model
    ):
        print(f"Cargando modelo guardado en {model_path}")
        model = load_model(model_path)
        print(f"Cargando historial del modelo guardado en {history_path}")
        with open(history_path, "rb") as f:
            history_json = pickle.load(f)
    else:
        print(
            f"No se encontró modelo en {model_path} o historial en {history_path}, construyendo uno nuevo."
        )
        model = build_keras_lr_model(X_train)

        history = model.fit(
            X_train, y_train, validation_data=(X_test, y_test), batch_size=64, epochs=25
        )
        history_json = history.history
        model.save(model_path)
        with open(history_path, "wb") as f:
            pickle.dump(history_json, f)

    return model, history_json


def build_tuner_lr_model(hp, input_dim):
    """
    Construye un modelo de regresión logística con hiperparámetros optimizados utilizando Keras Tuner.

    Args:
    hp (kt.HyperParameters): Objeto de Keras Tuner para la optimización de hiperparámetros.
    input_dim (int): Dimensión de la entrada.

    Returns:
    keras.Model: El modelo de regresión logística construido.
    """
    model = models.Sequential()
    model.add(Input(shape=(input_dim,)))
    for i in range(hp.Int("num_layers", 1, 3)):
        model.add(Dense(hp.Int(f"units_{i}", 32, 256, step=32), activation="relu"))
        model.add(Dropout(hp.Float(f"dropout_{i}", 0.0, 0.5, step=0.1)))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
        ),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC()],
    )
    return model


def get_tuner_lr_model(X_train, y_train, X_test, y_test, dl_dir, model_name="tuner_lr"):
    """
    Obtiene el mejor modelo de regresión logística utilizando Keras Tuner.

    Args:
    X_train (numpy.ndarray): Datos de entrenamiento.
    y_train (numpy.ndarray): Etiquetas de entrenamiento.
    X_test (numpy.ndarray): Datos de prueba.
    y_test (numpy.ndarray): Etiquetas de prueba.
    dl_dir (str): Directorio donde guardar los resultados de la optimización.
    model_name (str): Nombre del modelo.

    Returns:
    model: El modelo optimizado.
    tuner: El objeto de Keras Tuner utilizado para la optimización.
    """
    input_dim = X_train.shape[1]
    tuner = kt.Hyperband(
        lambda hp: build_tuner_lr_model(hp, input_dim),
        objective=kt.Objective("val_auc", direction="max"),
        max_epochs=20,
        factor=3,
        directory=dl_dir,
        project_name=model_name,
    )

    tuner.search(
        X_train,
        y_train,
        epochs=20,
        validation_data=(X_test, y_test),
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)],
    )

    model = tuner.get_best_models(num_models=1)[0]
    model.summary()
    return model, tuner


def build_lstm_model(X_train):
    """
    Construye y compila un modelo LSTM para clasificación binaria.

    Parámetros:
    X_train (array-like): Datos de entrada para determinar la forma de las secuencias.

    Retorna:
    model (tf.keras.models.Sequential): El modelo LSTM compilado.
    """
    model = Sequential()
    model.add(LSTM(50, input_shape=(1, X_train.shape[2]), return_sequences=True))
    model.add(LSTM(150))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC()],
    )
    print(model.summary())
    return model


def get_lstm_model(
    X_train, y_train, X_test, y_test, dl_dir, model_name="lstm", load_trained_model=True
):
    """
    Carga un modelo LSTM entrenado o entrena uno nuevo si no existe. Guarda el modelo y su historial.

    Parámetros:
    X_train (array-like): Datos de entrenamiento.
    y_train (array-like): Etiquetas de entrenamiento.
    X_test (array-like): Datos de prueba.
    y_test (array-like): Etiquetas de prueba.
    dl_dir (str): Directorio donde se almacenarán los modelos entrenados y su historial.
    model_name (str): Nombre del modelo (por defecto "lstm").
    load_trained_model (bool): Si es True, intenta cargar un modelo previamente entrenado.

    Retorna:
    model (tf.keras.models.Sequential): El modelo entrenado o cargado.
    history_json (dict): Historial del entrenamiento (precisión, pérdida, etc.).
    """
    dl_dir = "./trained_models/deep_learning/"
    models_dir = dl_dir + "model"
    history_dir = dl_dir + "history"

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(history_dir, exist_ok=True)

    model_path = os.path.join(models_dir, model_name + ".h5")
    history_path = os.path.join(history_dir, model_name + ".pkl")

    # Intentar cargar el modelo guardado
    if (
        os.path.exists(model_path)
        and os.path.exists(history_path)
        and load_trained_model
    ):
        print(f"Cargando modelo guardado en {model_path}")
        model = load_model(model_path)
        print(f"Cargando historial del modelo guardado en {history_path}")
        with open(history_path, "rb") as f:
            history_json = pickle.load(f)

    else:
        print(
            f"No se encontró modelo en {model_path} o historial en {history_path}, construyendo uno nuevo."
        )

        model = build_lstm_model(X_train)

        # Entrenar el modelo con EarlyStopping para evitar sobreajuste
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            batch_size=64,
            epochs=25,
            callbacks=[early_stopping],
        )

        history_json = history.history
        # Guardar el modelo en formato HDF5
        model.save(model_path)
        # Guardar el history como Pickle
        with open(history_path, "wb") as f:
            pickle.dump(history_json, f)

    return model, history_json


def build_cnn_model(X_train):
    """
    Construye y compila un modelo CNN para clasificación binaria.

    Parámetros:
    X_train (array-like): Datos de entrada para determinar la forma de las secuencias.

    Retorna:
    model (tf.keras.models.Sequential): El modelo CNN compilado.
    """
    model = Sequential()
    model.add(
        Conv1D(64, 2, activation="relu", input_shape=(X_train.shape[1], 1))
    )
    model.add(Conv1D(128, 2, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC()]
    )
    return model


def get_cnn_model(
    X_train, y_train, X_test, y_test, dl_dir, model_name="cnn", load_trained_model=True
):
    """
    Carga un modelo CNN entrenado o entrena uno nuevo si no existe. Guarda el modelo y su historial.

    Parámetros:
    X_train (array-like): Datos de entrenamiento.
    y_train (array-like): Etiquetas de entrenamiento.
    X_test (array-like): Datos de prueba.
    y_test (array-like): Etiquetas de prueba.
    dl_dir (str): Directorio donde se almacenarán los modelos entrenados y su historial.
    model_name (str): Nombre del modelo (por defecto "cnn").
    load_trained_model (bool): Si es True, intenta cargar un modelo previamente entrenado.

    Retorna:
    model (tf.keras.models.Sequential): El modelo entrenado o cargado.
    history_json (dict): Historial del entrenamiento (precisión, pérdida, etc.).
    """
    dl_dir = "./trained_models/deep_learning/"
    models_dir = dl_dir + "model"
    history_dir = dl_dir + "history"

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(history_dir, exist_ok=True)

    model_path = os.path.join(models_dir, model_name + ".h5")
    history_path = os.path.join(history_dir, model_name + ".pkl")

    # Intentar cargar el modelo guardado
    if (
        os.path.exists(model_path)
        and os.path.exists(history_path)
        and load_trained_model
    ):
        print(f"Cargando modelo guardado en {model_path}")
        model = load_model(model_path)
        print(f"Cargando historial del modelo guardado en {history_path}")
        with open(history_path, "rb") as f:
            history_json = pickle.load(f)

    else:
        print(
            f"No se encontró modelo en {model_path} o historial en {history_path}, construyendo uno nuevo."
        )

        model = build_cnn_model(X_train)
        # Entrenar el modelo con EarlyStopping para evitar sobreajuste
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            batch_size=64,
            epochs=25,
            callbacks=[early_stopping],
        )

        history_json = history.history
        # Guardar el modelo en formato HDF5
        model.save(model_path)
        # Guardar el history como Pickle
        with open(history_path, "wb") as f:
            pickle.dump(history_json, f)

    return model, history_json


def plot_history(history):
    """
    Dibuja un gráfico con la pérdida del modelo a lo largo de las épocas de entrenamiento.

    Parámetros:
    history (dict): Historial de entrenamiento, típicamente del objeto `history` retornado por `model.fit`.

    Retorna:
    None
    """
    plt.plot(history["loss"])
    plt.plot(history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Val"], loc="upper right")
    plt.show()
