# TFM - Predicción del Rendimiento Académico

## Abstract
Este proyecto tiene como objetivo predecir el rendimiento académico de los estudiantes utilizando el dataset **EdNet**. Se han aplicado modelos de **Machine Learning** y **Deep Learning** para evaluar la precisión y la capacidad explicativa de cada uno. Finalmente, se ha seleccionado **LightGBM** como el modelo más adecuado por su equilibrio entre rendimiento, interpretabilidad y escalabilidad. Además, se destaca la importancia de explorar modelos avanzados de **Knowledge Tracing**, que han demostrado en la literatura un mejor rendimiento en datasets similares.

## Estructura del Proyecto
El proyecto está organizado en las siguientes carpetas y archivos:

```
├── data
│   ├── read_data.ipynb       # Notebook para lectura y exploración del dataset original
│   ├── create_dataset.ipynb  # Notebook para la creación del dataset procesado
│   ├── ednet_dataset.csv     # Conjunto de datos con las transformaciones necesarias para el trabajo
│
├── trained_models
│   ├── deep_learning
│   │   ├── history          # Historial de entrenamiento de modelos de Deep Learning
│   │   ├── model            # Pesos y arquitectura de los modelos entrenados
│   ├── machine_learning
│   │   ├── model            # Modelos de Machine Learning entrenados
│
├── utils
│   ├── __init__.py          # Inicialización del módulo
│   ├── data_mining.py       # Funciones de preprocesamiento y análisis de datos
│   ├── dl.py                # Funciones relacionadas con Deep Learning
│   ├── ml.py                # Funciones relacionadas con Machine Learning
│
├── requirements.txt         # Dependencias del proyecto
├── TFM.ipynb                # Notebook principal del TFM
```

## Sobre el Dataset
El conjunto de datos utilizado en este proyecto proviene de **EdNet**, un dataset público diseñado para la investigación en educación y aprendizaje automático.

- **Fuente**: EdNet: A Large-Scale Hierarchical Dataset in Education
- **Licencia**: EdNet está bajo la licencia **CC BY-NC 4.0**, lo que permite su uso para investigación y proyectos no comerciales.

Dado el alto costo computacional, en este trabajo se ha utilizado una **muestra reducida** del dataset original. Esto puede haber afectado el rendimiento de los modelos de Deep Learning, ya que estos suelen beneficiarse de grandes volúmenes de datos. Si se utilizara el dataset completo, los modelos de redes neuronales podrían mejorar su desempeño y competir más estrechamente con los modelos de Machine Learning.
