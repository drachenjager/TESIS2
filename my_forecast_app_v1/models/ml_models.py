"""Modelos clásicos de Machine Learning para series temporales."""

import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def create_supervised_data(series, lag=1):
    """Convierte una serie en un conjunto supervisado X,y."""
    df = pd.DataFrame(series, columns=["y"])
    df["x_lag"] = df["y"].shift(lag)
    df.dropna(inplace=True)
    X = df[["x_lag"]].values  # <-- 2D array
    y = df["y"].values       # <-- 1D array
    return X, y


def train_linear_regression(train_data, test_data, forecast_steps=1):
    """Entrena una regresión lineal y pronostica."""
    X, y = create_supervised_data(train_data, lag=1)
    model = LinearRegression()
    model.fit(X, y)

    test_predictions = []
    current_input = train_data[-1]

    # Predicción recursiva: usamos el último valor conocido como entrada
    for actual_value in test_data:
        X_test = np.array([current_input]).reshape(1, -1)
        y_pred = model.predict(X_test)[0]
        test_predictions.append(y_pred)
        # Para la siguiente iteración usamos el valor real observado
        current_input = actual_value

    test_predictions = np.array(test_predictions)

    # Cálculo de métricas
    mae = np.mean(np.abs(test_predictions - test_data))
    rmse = math.sqrt(np.mean((test_predictions - test_data) ** 2))
    mape = np.mean(
        np.abs((test_predictions - test_data)
               / np.where(test_data == 0, np.finfo(float).eps, test_data))
    ) * 100
    r2 = (
        1
        - (np.sum((test_data - test_predictions) ** 2)
           / np.sum((test_data - np.mean(test_data)) ** 2))
        if np.sum((test_data - np.mean(test_data)) ** 2) != 0
        else float("nan")
    )

    # Pronóstico hacia adelante
    forecast_next = []
    current_input = test_data[-1]
    for _ in range(forecast_steps):
        X_next = np.array([current_input]).reshape(1, -1)
        next_pred = model.predict(X_next)[0]
        forecast_next.append(next_pred)
        current_input = next_pred

    metrics = {
        "Modelo": "Regresión Lineal",
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4),
        "R^2": round(r2, 4),
    }

    return metrics, test_predictions, [float(x) for x in forecast_next]


def train_random_forest(train_data, test_data, forecast_steps=1):
    """Entrena un bosque aleatorio para pronosticar."""
    X, y = create_supervised_data(train_data, lag=1)
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    test_predictions = []
    current_input = train_data[-1]

    for actual_value in test_data:
        X_test = np.array([current_input]).reshape(1, -1)
        y_pred = model.predict(X_test)[0]
        test_predictions.append(y_pred)
        current_input = actual_value

    test_predictions = np.array(test_predictions)

    mae = np.mean(np.abs(test_predictions - test_data))
    rmse = math.sqrt(np.mean((test_predictions - test_data) ** 2))
    mape = np.mean(
        np.abs((test_predictions - test_data)
               / np.where(test_data == 0, np.finfo(float).eps, test_data))
    ) * 100
    r2 = (
        1
        - (np.sum((test_data - test_predictions) ** 2)
           / np.sum((test_data - np.mean(test_data)) ** 2))
        if np.sum((test_data - np.mean(test_data)) ** 2) != 0
        else float("nan")
    )

    forecast_next = []
    current_input = test_data[-1]
    for _ in range(forecast_steps):
        X_next = np.array([current_input]).reshape(1, -1)
        next_pred = model.predict(X_next)[0]
        forecast_next.append(next_pred)
        current_input = next_pred

    metrics = {
        "Modelo": "Random Forest",
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4),
        "R^2": round(r2, 4),
    }

    return metrics, test_predictions, [float(x) for x in forecast_next]
