# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#


from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score  # Importación de cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    precision_score,
    balanced_accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import gzip
from joblib import dump
import json
import pandas as pd
import numpy as np
from tqdm import tqdm  # Barra de progreso
import os

# Paso 1: Cargar y limpiar los datasets
def load_and_clean_data(file_path):
    # Leer el archivo comprimido como CSV
    df = pd.read_csv(file_path, compression='zip')

    # Renombrar columna objetivo
    df.rename(columns={"default payment next month": "default"}, inplace=True)

    # Eliminar columna ID
    if "ID" in df.columns:
        df.drop(columns=["ID"], inplace=True)

    # Eliminar registros con información no disponible
    df.replace({"EDUCATION": {0: np.nan}, "MARRIAGE": {0: np.nan}}, inplace=True)
    df.dropna(inplace=True)

    # Agrupar niveles superiores de educación
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: x if x <= 4 else 4)

    return df

# Actualiza las rutas a los archivos
train_path = "files/input/train_data.csv.zip"
test_path = "files/input/test_data.csv.zip"

# Cargar y limpiar los datos
train_df = load_and_clean_data(train_path)
test_df = load_and_clean_data(test_path)

# Paso 2: Dividir datasets en características y variable objetivo
x_train, y_train = train_df.drop(columns=["default"]), train_df["default"]
x_test, y_test = test_df.drop(columns=["default"]), test_df["default"]

# Paso 3: Crear un pipeline para el modelo de clasificación
categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
preprocessor = ColumnTransformer([
    ("onehot", OneHotEncoder(), categorical_features)
], remainder="passthrough")

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# Paso 4: Optimizar hiperparámetros usando validación cruzada con barra de progreso
param_grid = {
    "classifier__n_estimators": [50, 70, 100],
    "classifier__max_depth": [None, 10, 20, 40],
    "classifier__min_samples_split": [2, 5]
}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Crear una lista de combinaciones de hiperparámetros para la barra de progreso
param_combinations = [
    {"classifier__n_estimators": n, "classifier__max_depth": d, "classifier__min_samples_split": s}
    for n in param_grid["classifier__n_estimators"]
    for d in param_grid["classifier__max_depth"]
    for s in param_grid["classifier__min_samples_split"]
]

best_score = 0
best_params = None

# Barra de progreso para las combinaciones
with tqdm(total=len(param_combinations), desc="Optimización de hiperparámetros") as pbar:
    for params in param_combinations:
        pipeline.set_params(**params)
        scores = cross_val_score(pipeline, x_train, y_train, cv=cv, scoring="balanced_accuracy")
        mean_score = scores.mean()

        if mean_score > best_score:
            best_score = mean_score
            best_params = params

        pbar.update(1)  # Actualizar la barra de progreso

# Configurar el pipeline con los mejores parámetros
pipeline.set_params(**best_params)
pipeline.fit(x_train, y_train)

# Mejor modelo
best_model = pipeline

# Paso 5: Guardar el modelo
output_model_path = "files/models/model.pkl.gz"

# Crear el directorio si no existe
os.makedirs(os.path.dirname(output_model_path), exist_ok=True)

# Guardar el modelo
with gzip.open(output_model_path, "wb") as f:
    dump(best_model, f)

# Paso 6: Calcular métricas y guardarlas con barra de progreso
metrics = []

def calculate_metrics(x, y, dataset_name):
    y_pred = best_model.predict(x)
    metrics.append({
        "type": "metrics",
        "dataset": dataset_name,
        "precision": precision_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1_score": f1_score(y, y_pred)
    })
    cm = confusion_matrix(y, y_pred)
    metrics.append({
        "type": "cm_matrix",
        "dataset": dataset_name,
        "true_0": {"predicted_0": cm[0, 0], "predicted_1": cm[0, 1]},
        "true_1": {"predicted_0": cm[1, 0], "predicted_1": cm[1, 1]}
    })

# Barra de progreso para calcular las métricas
datasets = [("train", x_train, y_train), ("test", x_test, y_test)]
with tqdm(total=len(datasets), desc="Cálculo de métricas") as pbar:
    for dataset_name, x, y in datasets:
        calculate_metrics(x, y, dataset_name)
        pbar.update(1)  # Actualizar la barra de progreso

# Guardar métricas en un archivo JSON, cada métrica en una línea separada
output_metrics_path = "files/output/metrics.json"

# Crear el directorio si no existe
os.makedirs(os.path.dirname(output_metrics_path), exist_ok=True)

# Convertir objetos NumPy a tipos compatibles con JSON
def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# Guardar las métricas, una por línea en el archivo JSON
with open(output_metrics_path, "w") as f:
    for metric in metrics:
        json.dump(metric, f, indent=4, default=convert_to_serializable)
        f.write("\n")  # Escribir cada métrica en una línea separada
