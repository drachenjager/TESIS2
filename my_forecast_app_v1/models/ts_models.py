# models/ts_models.py
import numpy as np
import pandas as pd
import math
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def train_sarima(train_data, test_data):
    # Por simplicidad, usaremos un (1,1,1) y estacionalidad = 12
    # En la práctica, se deben seleccionar p,d,q y parámetros estacionales mediante un proceso de búsqueda.
    model = SARIMAX(train_data, order=(1,1,1), seasonal_order=(1,1,1,12), enforce_stationarity=False, enforce_invertibility=False)
    sarima_fit = model.fit(disp=False)
    
    # Predicción en el set de prueba
    predictions = sarima_fit.predict(start=len(train_data), end=len(train_data)+len(test_data)-1, dynamic=False)
    
    # Métricas (ejemplo: MAE, RMSE)
    mae = np.mean(np.abs(predictions - test_data))
    rmse = math.sqrt(np.mean((predictions - test_data)**2))
    
    # Pronóstico del siguiente punto
    forecast_next = sarima_fit.predict(start=len(train_data)+len(test_data), end=len(train_data)+len(test_data))
    
    metrics = {
        "Modelo": "SARIMA",
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4)
    }
    
    return metrics, predictions, float(forecast_next)

def train_holtwinters(train_data, test_data):
    # Ver cuántos datos hay en train
    n_train = len(train_data)
    # Solo usar estacionalidad si hay >= 2 ciclos de 12
    if n_train < 24:
        # O bien no usamos componente estacional
        model = ExponentialSmoothing(train_data, trend='add', seasonal=None)
    else:
        # O el caso "original"
        model = ExponentialSmoothing(train_data, seasonal_periods=12, trend='add', seasonal='mul')
    
    hw_fit = model.fit()
    
    # Predicción
    predictions = hw_fit.predict(start=len(train_data), end=len(train_data)+len(test_data)-1)
    
    mae = np.mean(np.abs(predictions - test_data))
    rmse = math.sqrt(np.mean((predictions - test_data)**2))
    
    forecast_next = hw_fit.predict(start=len(train_data)+len(test_data), end=len(train_data)+len(test_data))
    
    metrics = {
        "Modelo": "Holt-Winters",
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4)
    }
    
    return metrics, predictions, float(forecast_next)
