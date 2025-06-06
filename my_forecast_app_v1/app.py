from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Importamos funciones de modelado (veremos detalles luego)
from models.ts_models import train_sarima, train_holtwinters
from models.ml_models import train_linear_regression, train_random_forest
from models.dl_models import train_rnn, train_lstm, prepare_data_dl

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    En esta vista cargamos la página principal con el combo box para el período.
    Al hacer POST, obtenemos los datos, entrenamos y mostramos métricas.
    """
    if request.method == 'POST':
        # 1. Leemos el período seleccionado en el formulario
        selected_period = request.form.get('period_select')
        
        # Porcentaje para testing
        test_percent = float(request.form.get('test_percent', 20))

        # 3. Obtenemos los datos de Yahoo Finance para ese período
        df = get_data_from_yahoo(period=selected_period)
        # 4. Entrenamos los modelos y obtenemos predicciones + métricas
        test_size = max(1, int(len(df) * test_percent / 100))
        metrics_df, forecast_values, pred_series, train_series, test_series = train_and_evaluate_all_models(
            df,
            forecast_steps=test_size,
            test_size=test_size
        )
        # 5. Renderizamos la plantilla con los resultados
        return render_template('index.html',
                               period=selected_period,
                               tables=[metrics_df.to_html(classes='table table-striped', index=False)],
                               forecast_values=forecast_values,
                               train_series=train_series,
                               test_series=test_series,
                               pred_series=pred_series
                               )
    else:
        # Método GET: solo mostramos el formulario vacío
        return render_template('index.html')

def get_data_from_yahoo(period='1y'):
    """
    Descarga datos de USD/MXN de Yahoo Finance en base a un string de período:
    '1d','5d','1mo','3mo','6mo','1y','2y','5y','10y','ytd','max'
    """
    ticker = "MXN=X"  # El par USD/MXN en Yahoo Finance se identifica como "MXN=X"
    data = yf.download(ticker, period=period, interval="1d")

    # yfinance>=0.2 puede devolver columnas MultiIndex incluso para un solo ticker
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].iloc[:, 0]
    else:
        close = data["Close"]

    close = close.dropna().reset_index()
    close.columns = ["Date", "Close"]
    return close

def train_and_evaluate_all_models(df, forecast_steps=1, test_size=5):
    ts = df['Close'].values
    if test_size >= len(ts):
        test_size = max(1, len(ts) // 2)
    train_data = ts[:-test_size]
    test_data = ts[-test_size:]
    
    # Series de tiempo
    sarima_metrics, sarima_pred, sarima_forecast = train_sarima(train_data, test_data, forecast_steps)
    hw_metrics, _, hw_forecast = train_holtwinters(train_data, test_data, forecast_steps)

    # ML
    linreg_metrics, _, linreg_forecast = train_linear_regression(train_data, test_data, forecast_steps)
    rf_metrics, _, rf_forecast = train_random_forest(train_data, test_data, forecast_steps)

    # Deep Learning
    rnn_metrics, _, rnn_forecast = train_rnn(train_data, test_data, forecast_steps)
    lstm_metrics, _, lstm_forecast = train_lstm(train_data, test_data, forecast_steps)
    
    # Construimos DataFrame de métricas
    metrics_df = pd.DataFrame([
        sarima_metrics,
        hw_metrics,
        linreg_metrics,
        rf_metrics,
        rnn_metrics,
        lstm_metrics
    ])
    
    # Forecast "del siguiente punto" (por simplicidad tomamos el forecast devuelto)
    forecast_values = {
        "SARIMA": sarima_forecast,
        "Holt-Winters": hw_forecast,
        "Regresión Lineal": linreg_forecast,
        "Random Forest": rf_forecast,
        "RNN": rnn_forecast,
        "LSTM": lstm_forecast
    }
    
    train_series = train_data.tolist() + [None] * len(test_data)
    test_series = [None] * len(train_data) + test_data.tolist()
    pred_series = [None] * len(train_data) + sarima_pred.tolist()

    return metrics_df, forecast_values, pred_series, train_series, test_series

if __name__ == '__main__':
    # app.run(debug=True)  # Para desarrollo local
    app.run(host='0.0.0.0', port=8080)  # Ajusta el puerto si es necesario
