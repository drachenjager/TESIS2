"""Modelos de Deep Learning para la serie USD/MXN."""

import numpy as np
import pandas as pd
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM


def prepare_data_dl(series, lag=1):
    """Crea matrices supervisadas para RNN/LSTM.

    Parameters
    ----------
    series : array-like
        Serie de tiempo original.
    lag : int, opcional
        Número de rezagos a utilizar como entrada del modelo.

    Returns
    -------
    tuple
        ``(X, y)`` arreglos listos para alimentar redes recurrentes.
    """
    df = pd.DataFrame(series, columns=["y"])
    df["x_lag"] = df["y"].shift(lag)
    df.dropna(inplace=True)
    X = df[["x_lag"]].values
    y = df["y"].values

    # Para RNN/LSTM: reshape a (muestras, timesteps, features)
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    return X, y


def train_rnn(train_data, test_data, forecast_steps=1):
    """Entrena una red neuronal recurrente simple y genera pronósticos."""
    X_train, y_train = prepare_data_dl(train_data)

    # Definimos la arquitectura de la RNN
    model = Sequential()
    model.add(SimpleRNN(50, activation="relu", input_shape=(1, 1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    # Entrenamiento de la red
    model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)

    # Predicción en el conjunto de prueba, de forma recursiva
    predictions = []
    current_input = np.array([[train_data[-1]]])  # shape=(1,1)
    current_input = current_input.reshape((1, 1, 1))

    for actual_value in test_data:
        y_pred = model.predict(current_input, verbose=0)
        predictions.append(y_pred[0][0])
        # Actualizamos input con el valor real observado
        current_input = np.array([[actual_value]])
        current_input = current_input.reshape((1, 1, 1))

    predictions = np.array(predictions)

    # Cálculo de métricas de error y de ajuste
    mae = np.mean(np.abs(predictions - test_data))
    rmse = math.sqrt(np.mean((predictions - test_data) ** 2))
    mape = np.mean(
        np.abs((predictions - test_data)
               / np.where(test_data == 0, np.finfo(float).eps, test_data))
    ) * 100
    r2 = (
        1
        - (np.sum((test_data - predictions) ** 2)
           / np.sum((test_data - np.mean(test_data)) ** 2))
        if np.sum((test_data - np.mean(test_data)) ** 2) != 0
        else float("nan")
    )

    # Forecast del siguiente punto(s)
    forecast_next = []
    next_input = np.array([[test_data[-1]]]).reshape((1, 1, 1))
    for _ in range(forecast_steps):
        next_pred = model.predict(next_input, verbose=0)[0][0]
        forecast_next.append(next_pred)
        next_input = np.array([[next_pred]]).reshape((1, 1, 1))

    metrics = {
        "Modelo": "RNN",
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4),
        "R^2": round(r2, 4),
    }
    return metrics, predictions, [float(x) for x in forecast_next]


def train_lstm(train_data, test_data, forecast_steps=1):
    """Entrena una red LSTM y devuelve pronósticos."""
    X_train, y_train = prepare_data_dl(train_data)

    # Definimos la arquitectura LSTM
    model = Sequential()
    model.add(LSTM(50, activation="relu", input_shape=(1, 1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)

    predictions = []
    current_input = np.array([[train_data[-1]]])  # shape=(1,1)
    current_input = current_input.reshape((1, 1, 1))

    for actual_value in test_data:
        y_pred = model.predict(current_input, verbose=0)
        predictions.append(y_pred[0][0])
        current_input = np.array([[actual_value]])
        current_input = current_input.reshape((1, 1, 1))

    predictions = np.array(predictions)

    mae = np.mean(np.abs(predictions - test_data))
    rmse = math.sqrt(np.mean((predictions - test_data) ** 2))
    mape = np.mean(
        np.abs((predictions - test_data)
               / np.where(test_data == 0, np.finfo(float).eps, test_data))
    ) * 100
    r2 = (
        1
        - (np.sum((test_data - predictions) ** 2)
           / np.sum((test_data - np.mean(test_data)) ** 2))
        if np.sum((test_data - np.mean(test_data)) ** 2) != 0
        else float("nan")
    )

    forecast_next = []
    next_input = np.array([[test_data[-1]]]).reshape((1, 1, 1))
    for _ in range(forecast_steps):
        next_pred = model.predict(next_input, verbose=0)[0][0]
        forecast_next.append(next_pred)
        next_input = np.array([[next_pred]]).reshape((1, 1, 1))

    metrics = {
        "Modelo": "LSTM",
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4),
        "R^2": round(r2, 4),
    }
    return metrics, predictions, [float(x) for x in forecast_next]
