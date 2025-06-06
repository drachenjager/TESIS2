from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Importamos funciones de modelado (veremos detalles luego)
from models.ts_models import train_sarima, train_holtwinters
from models.ml_models import train_linear_regression, train_random_forest
from models.dl_models import train_rnn, train_lstm

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
        
        # 2. Leemos cuántos puntos quiere pronosticar el usuario
        horizon = int(request.form.get('forecast_horizon', 1))
        # Porcentaje para testing
        test_percent = float(request.form.get('test_percent', 20))

        # 3. Obtenemos los datos de Yahoo Finance para ese período
        df = get_data_from_yahoo(period=selected_period)
        metrics_df, _, _, _, _ = train_and_evaluate_all_models(
            df,
            forecast_steps=horizon,
            test_size=max(1, int(len(df) * test_percent / 100))
        )
        return render_template(
            'index.html',
            period=selected_period,
            horizon=horizon,
            test_percent=test_percent,
            tables=[metrics_df.to_html(classes='table table-striped', index=False)]
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


@app.route('/results', methods=['POST'])
def results_view():
    selected_model = request.form.get('model_choice')
    selected_period = request.form.get('period_select')
    horizon = int(request.form.get('forecast_horizon', 1))
    test_percent = float(request.form.get('test_percent', 20))

    df = get_data_from_yahoo(period=selected_period)
    ts = df['Close'].values
    test_size = max(1, int(len(df) * test_percent / 100))
    train_data = ts[:-test_size]
    test_data = ts[-test_size:]

    if selected_model == 'SARIMA':
        _, preds, _ = train_sarima(train_data, test_data, horizon)
    elif selected_model == 'Holt-Winters':
        _, preds, _ = train_holtwinters(train_data, test_data, horizon)
    elif selected_model == 'Regresión Lineal':
        _, preds, _ = train_linear_regression(train_data, test_data, horizon)
    elif selected_model == 'Random Forest':
        _, preds, _ = train_random_forest(train_data, test_data, horizon)
    elif selected_model == 'RNN':
        _, preds, _ = train_rnn(train_data, test_data, horizon)
    else:  # LSTM
        _, preds, _ = train_lstm(train_data, test_data, horizon)

    train_series = train_data.tolist() + [None] * len(test_data)
    test_series = [None] * len(train_data) + test_data.tolist()
    pred_series = [None] * len(train_data) + preds.tolist()
    labels = df['Date'].dt.strftime('%Y-%m-%d').tolist()
    rows = list(zip(labels, train_series, test_series, pred_series))

    return render_template(
        'results.html',
        model=selected_model,
        rows=rows,
        labels=labels,
        train_series=train_series,
        test_series=test_series,
        pred_series=pred_series
    )

if __name__ == '__main__':
    # app.run(debug=True)  # Para desarrollo local
    app.run(host='0.0.0.0', port=8080)  # Ajusta el puerto si es necesario
