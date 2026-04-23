import numpy as np
from ml.train import fetch_historical_data, engineer_features, LOOKBACK

try:
    from tensorflow.keras.models import Sequential
    USE_LSTM = True
except ImportError:
    USE_LSTM = False


def predict_signal(symbol: str, model, scaler):
    """
    Given a trained model + scaler, predict Buy/Sell/Hold for the stock.
    Returns: (signal: str, confidence: float, chart_data: dict)
    """
    # engineer_features drops ~50 rows for NaN warm-up
    # We need at least LOOKBACK (60) clean rows after that drop.
    # Fetching 6 months (~180 trading days) gives a safe margin on any ticker.
    df = fetch_historical_data(symbol, months=6)
    df = engineer_features(df)

    feature_cols = [
        "Close", "MA7", "MA21", "MA50", "ROC", "RSI", "MACD", "MACD_Signal",
        "BB_Width", "Stoch_K", "Stoch_D", "ATR", "Vol_Change", "Vol_Ratio", "Close_Norm"
    ]
    X_raw = df[feature_cols].values

    # Validate no infinity or NaN in data
    if np.isinf(X_raw).any() or np.isnan(X_raw).any():
        raise ValueError(f"Data contains NaN or infinity values. Data shape: {X_raw.shape}")

    if USE_LSTM and hasattr(model, 'predict') and hasattr(model, 'layers'):
        X_scaled = scaler.transform(X_raw)
        
        # Final validation after scaling
        if np.isinf(X_scaled).any() or np.isnan(X_scaled).any():
            raise ValueError("Scaled data contains NaN or infinity values")
        
        if len(X_scaled) < LOOKBACK:
            raise ValueError(
                f"Only {len(X_scaled)} clean rows available after feature engineering; "
                f"need at least {LOOKBACK}. Try a stock with more trading history."
            )
        sequence = X_scaled[-LOOKBACK:].reshape(1, LOOKBACK, len(feature_cols))
        prob = float(model.predict(sequence, verbose=0)[0][0])
        
        # For LSTM, treat 0.5 as neutral threshold
        pred_class = 1 if prob >= 0.5 else 0
        confidence_margin = abs(prob - 0.5) * 2
        
        print(f"[predict-LSTM] {symbol}: prob={prob:.4f}, pred_class={pred_class}, margin={confidence_margin:.4f}")
    else:
        X_scaled = scaler.transform(X_raw)
        
        # Final validation after scaling
        if np.isinf(X_scaled).any() or np.isnan(X_scaled).any():
            raise ValueError("Scaled data contains NaN or infinity values")
        
        latest = X_scaled[-1].reshape(1, -1)
        proba = model.predict_proba(latest)[0]  # [prob_class_0, prob_class_1]
        
        # Use the max probability as confidence, but shift threshold based on prediction
        pred_class = model.predict(latest)[0]  # Get the predicted class
        prob = proba[1]  # Probability of class 1 (price up)
        
        # Use feature importance-weighted confidence
        # If the model is confident in its prediction, show BUY/SELL, otherwise HOLD
        confidence_margin = abs(prob - 0.5) * 2  # Ranges from 0 to 1
        
        print(f"[predict] {symbol}: prob={prob:.4f}, pred_class={pred_class}, margin={confidence_margin:.4f}")

    # Signal thresholds - based on predicted class + confidence
    if pred_class == 1 and confidence_margin >= 0.30:
        signal = "BUY"
        confidence = max(prob, 1 - prob)  # Use higher of the two probabilities
    elif pred_class == 0 and confidence_margin >= 0.30:
        signal = "SELL"
        confidence = max(prob, 1 - prob)  # Use higher of the two probabilities
    else:
        signal = "HOLD"
        confidence = confidence_margin  # Use the margin itself for HOLD

    # Chart data: last 3 months of raw closes
    chart_df  = fetch_historical_data(symbol, months=3)
    closes    = chart_df["Close"].values.flatten().tolist()
    dates     = [str(d.date()) for d in chart_df.index]
    close_arr = __import__('numpy').array(closes)
    ma7  = __import__('numpy').convolve(close_arr, __import__('numpy').ones(7)  / 7,  mode='valid').tolist()
    ma21 = __import__('numpy').convolve(close_arr, __import__('numpy').ones(21) / 21, mode='valid').tolist()

    chart_data = {
        "dates":       dates,
        "closes":      [round(float(c), 2) for c in closes],
        "ma7":         [round(float(v), 2) for v in ma7],
        "ma21":        [round(float(v), 2) for v in ma21],
        "ma7_offset":  6,
        "ma21_offset": 20,
    }

    return signal, confidence, chart_data
