from flask import Flask, render_template, request, jsonify, redirect, url_for
import json
from datetime import datetime, timedelta
from ml.train import train_model
from ml.predict import predict_signal
from aws.s3_utils import list_saved_models

app = Flask(__name__)

POPULAR_STOCKS = [
    {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology"},
    {"symbol": "GOOGL", "name": "Alphabet Inc.", "sector": "Technology"},
    {"symbol": "MSFT", "name": "Microsoft Corp.", "sector": "Technology"},
    {"symbol": "AMZN", "name": "Amazon.com Inc.", "sector": "Consumer"},
    {"symbol": "TSLA", "name": "Tesla Inc.", "sector": "Automotive"},
    {"symbol": "META", "name": "Meta Platforms", "sector": "Technology"},
    {"symbol": "NVDA", "name": "NVIDIA Corp.", "sector": "Semiconductors"},
    {"symbol": "JPM", "name": "JPMorgan Chase", "sector": "Finance"},
    {"symbol": "NFLX", "name": "Netflix Inc.", "sector": "Entertainment"},
    {"symbol": "AMD", "name": "Advanced Micro Devices", "sector": "Semiconductors"},
    {"symbol": "RELIANCE.NS", "name": "Reliance Industries", "sector": "Conglomerate"},
    {"symbol": "TCS.NS", "name": "Tata Consultancy Services", "sector": "Technology"},
    {"symbol": "INFY.NS", "name": "Infosys Ltd.", "sector": "Technology"},
    {"symbol": "HDFCBANK.NS", "name": "HDFC Bank", "sector": "Finance"},
]

@app.route("/")
def index():
    return render_template("index.html", stocks=POPULAR_STOCKS)

@app.route("/analyze", methods=["POST"])
def analyze():
    symbol = request.form.get("symbol", "").upper().strip()
    if not symbol:
        return redirect(url_for("index"))

    try:
        # Train model on 10 months of historical data
        model, scaler, metrics = train_model(symbol, months=10)

        # Get prediction signal
        signal, confidence, chart_data = predict_signal(symbol, model, scaler)

        stock_info = next((s for s in POPULAR_STOCKS if s["symbol"] == symbol), {
            "symbol": symbol, "name": symbol, "sector": "Unknown"
        })

        return render_template(
            "result.html",
            symbol=symbol,
            stock=stock_info,
            signal=signal,
            confidence=round(confidence * 100, 2),
            metrics=metrics,
            chart_data=json.dumps(chart_data),
            timestamp=(datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%B %d, %Y at %I:%M %p")
        )
    except ValueError as e:
        error_msg = str(e)
        if "infinity" in error_msg.lower() or "nan" in error_msg.lower():
            error_msg = f"Could not analyze {symbol}: The data quality is poor or the stock has insufficient trading history. Try another ticker."
        elif "insufficient" in error_msg.lower():
            error_msg = f"Insufficient data for {symbol}. Please try a stock with more trading history."
        return render_template("error.html", error=error_msg, symbol=symbol)
    except Exception as e:
        error_msg = f"Could not analyze {symbol}: {str(e)}"
        return render_template("error.html", error=error_msg, symbol=symbol)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API endpoint for AJAX calls."""
    data = request.get_json()
    symbol = data.get("symbol", "").upper()
    try:
        model, scaler, metrics = train_model(symbol, months=10)
        signal, confidence, chart_data = predict_signal(symbol, model, scaler)
        return jsonify({
            "symbol": symbol,
            "signal": signal,
            "confidence": round(confidence * 100, 2),
            "metrics": metrics,
            "chart_data": chart_data,
            "status": "success"
        })
    except ValueError as e:
        error_msg = str(e)
        if "infinity" in error_msg.lower() or "nan" in error_msg.lower():
            error_msg = f"Data quality issue for {symbol}. The stock may have insufficient trading history."
        elif "insufficient" in error_msg.lower():
            error_msg = f"Insufficient data for {symbol}. Try a different stock."
        return jsonify({"status": "error", "message": error_msg}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": f"Analysis failed for {symbol}: {str(e)}"}), 500

@app.route("/history")
def history():
    models = list_saved_models()
    return render_template("history.html", models=models)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
