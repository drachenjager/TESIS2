from flask import Flask, render_template, request
import json
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Importamos funciones de modelado (veremos detalles luego)
from models.ts_models import train_sarima, train_holtwinters
from models.ml_models import train_linear_regression, train_random_forest
from models.dl_models import train_rnn, train_lstm, prepare_data_dl

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    """
    En esta vista cargamos la página principal con el combo box para el período.
    Al hacer POST, obtenemos los datos, entrenamos y mostramos métricas.
    """
    if request.method == "POST":
        # 1. Leemos el período seleccionado en el formulario
        selected_period = request.form.get("period_select")

        # Porcentaje para testing
        test_percent = float(request.form.get("test_percent", 20))

        # 3. Obtenemos los datos de Yahoo Finance para ese período
        df = get_data_from_yahoo(period=selected_period)
        # 4. Entrenamos los modelos y obtenemos predicciones + métricas
        test_size = max(1, int(len(df) * test_percent / 100))
        metrics_df, forecast_values, train_series, test_series, predictions_dict = (
            train_and_evaluate_all_models(
                df, forecast_steps=test_size, test_size=test_size
            )
        )

        dates = df["Date"].dt.strftime("%Y-%m-%d").tolist()
        # 5. Renderizamos la plantilla con los resultados
        metrics_table = metrics_df.to_html(classes="table table-striped", index=False)
        return render_template(
            "index.html",
            period=selected_period,
            metrics_table=metrics_table,
            forecast_values=forecast_values,
            train_series=train_series,
            test_series=test_series,
            predictions_dict=predictions_dict,
            dates=dates,
        )
    else:
        # Método GET: solo mostramos el formulario vacío
        return render_template("index.html")


def get_data_from_yahoo(period="1y"):
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
    ts = df["Close"].values
    if test_size >= len(ts):
        test_size = max(1, len(ts) // 2)
    train_data = ts[:-test_size]
    test_data = ts[-test_size:]

    # Series de tiempo
    sarima_metrics, sarima_pred, sarima_forecast = train_sarima(
        train_data, test_data, forecast_steps
    )
    hw_metrics, hw_pred, hw_forecast = train_holtwinters(
        train_data, test_data, forecast_steps
    )

    # ML
    linreg_metrics, linreg_pred, linreg_forecast = train_linear_regression(
        train_data, test_data, forecast_steps
    )
    rf_metrics, rf_pred, rf_forecast = train_random_forest(
        train_data, test_data, forecast_steps
    )

    # Deep Learning
    rnn_metrics, rnn_pred, rnn_forecast = train_rnn(
        train_data, test_data, forecast_steps
    )
    lstm_metrics, lstm_pred, lstm_forecast = train_lstm(
        train_data, test_data, forecast_steps
    )

    # Construimos DataFrame de métricas
    metrics_df = pd.DataFrame(
        [
            sarima_metrics,
            hw_metrics,
            linreg_metrics,
            rf_metrics,
            rnn_metrics,
            lstm_metrics,
        ]
    )

    # Forecast "del siguiente punto" (por simplicidad tomamos el forecast devuelto)
    forecast_values = {
        "SARIMA": sarima_forecast,
        "Holt-Winters": hw_forecast,
        "Regresión Lineal": linreg_forecast,
        "Random Forest": rf_forecast,
        "RNN": rnn_forecast,
        "LSTM": lstm_forecast,
    }

    train_series = train_data.tolist() + [None] * len(test_data)
    test_series = [None] * len(train_data) + test_data.tolist()

    predictions_dict = {
        "SARIMA": [None] * len(train_data) + sarima_pred.tolist(),
        "Holt-Winters": [None] * len(train_data) + hw_pred.tolist(),
        "Regresión Lineal": [None] * len(train_data) + linreg_pred.tolist(),
        "Random Forest": [None] * len(train_data) + rf_pred.tolist(),
        "RNN": [None] * len(train_data) + rnn_pred.tolist(),
        "LSTM": [None] * len(train_data) + lstm_pred.tolist(),
    }

    return metrics_df, forecast_values, train_series, test_series, predictions_dict


@app.route("/plot", methods=["POST"])
def plot():
    model_name = request.form.get("model_choice")
    train_series = json.loads(request.form.get("train_series"))
    test_series = json.loads(request.form.get("test_series"))
    dates = json.loads(request.form.get("dates"))
    pred_series = json.loads(request.form.get(f"pred_{model_name}"))

    return render_template(
        "plot.html",
        model_name=model_name,
        train_series=train_series,
        test_series=test_series,
        pred_series=pred_series,
        dates=dates,
    )


if __name__ == "__main__":
    # app.run(debug=True)  # Para desarrollo local
    app.run(host="0.0.0.0", port=8080)  # Ajusta el puerto si es necesario
