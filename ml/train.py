import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import joblib
import os

# Try importing Keras/TensorFlow; fallback to sklearn RandomForest
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    USE_LSTM = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    USE_LSTM = False

from aws.s3_utils import upload_model, download_model

LOOKBACK = 60  # Days used as input sequence


def fetch_historical_data(symbol: str, months: int = 10):
    """Fetch stock data for the past N months using yfinance."""
    end = datetime.today()
    start = end - timedelta(days=months * 31)
    df = yf.download(symbol, start=start.strftime("%Y-%m-%d"),
                     end=end.strftime("%Y-%m-%d"), progress=False)
    if df.empty:
        raise ValueError(f"No data found for symbol: {symbol}")
    return df


def engineer_features(df):
    """Add technical indicators as features."""
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # Price momentum indicators
    df["MA7"] = close.rolling(7).mean()
    df["MA21"] = close.rolling(21).mean()
    df["MA50"] = close.rolling(50).mean()
    
    # Price rate of change
    df["ROC"] = close.pct_change(periods=10)
    df["ROC"] = df["ROC"].replace([np.inf, -np.inf], np.nan)

    # RSI - with safeguards against infinity
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    rs = rs.replace([np.inf, -np.inf], np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].replace([np.inf, -np.inf], np.nan).fillna(50)

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()
    df["MACD"] = df["MACD"].replace([np.inf, -np.inf], np.nan)
    df["MACD_Signal"] = df["MACD_Signal"].replace([np.inf, -np.inf], np.nan)

    # Bollinger Band Width - with safeguards
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["BB_Width"] = (bb_std * 2) / (bb_mid + 1e-9)
    df["BB_Upper"] = bb_mid + (2 * bb_std)
    df["BB_Lower"] = bb_mid - (2 * bb_std)
    df["BB_Width"] = df["BB_Width"].replace([np.inf, -np.inf], np.nan)
    
    # Stochastic Oscillator
    lowest_low = low.rolling(14).min()
    highest_high = high.rolling(14).max()
    df["Stoch_K"] = (close - lowest_low) / (highest_high - lowest_low + 1e-9) * 100
    df["Stoch_K"] = df["Stoch_K"].replace([np.inf, -np.inf], np.nan)
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()
    
    # ATR (Average True Range) for volatility
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    # Volume indicators
    vol_change = volume.pct_change()
    vol_ma = volume.rolling(20).mean()
    vol_ratio = volume / (vol_ma + 1e-9)
    
    df["Vol_Change"] = vol_change
    df["Vol_Ratio"] = vol_ratio
    
    df["Vol_Change"] = df["Vol_Change"].replace([np.inf, -np.inf], np.nan)
    df["Vol_Ratio"] = df["Vol_Ratio"].replace([np.inf, -np.inf], np.nan)
    df["Vol_Change"] = df["Vol_Change"].fillna(0)
    df["Vol_Ratio"] = df["Vol_Ratio"].fillna(1)

    # Close price normalized to rolling window
    df["Close_Norm"] = (close - close.rolling(20).mean()) / (close.rolling(20).std() + 1e-9)
    df["Close_Norm"] = df["Close_Norm"].replace([np.inf, -np.inf], np.nan)

    # Target: 1 if price goes up next day, 0 if down
    df["Target"] = (close.shift(-1) > close).astype(int)

    # Forward fill and then backward fill for any remaining NaNs
    df = df.ffill().bfill()
    
    # Remove any rows still with NaN or infinity
    df.dropna(inplace=True)
    df = df[~df.isin([np.inf, -np.inf]).any(axis=1)]

    if len(df) < LOOKBACK + 20:
        raise ValueError(f"Insufficient clean data after feature engineering: {len(df)} rows < {LOOKBACK + 20} required")

    return df


def prepare_sequences(df, scaler=None):
    """Prepare LSTM sequences from feature DataFrame."""
    feature_cols = [
        "Close", "MA7", "MA21", "MA50", "ROC", "RSI", "MACD", "MACD_Signal", 
        "BB_Width", "Stoch_K", "Stoch_D", "ATR", "Vol_Change", "Vol_Ratio", "Close_Norm"
    ]
    X_raw = df[feature_cols].values

    # Remove any rows with infinity or NaN in raw data
    valid_rows = ~np.isinf(X_raw).any(axis=1) & ~np.isnan(X_raw).any(axis=1)
    if not valid_rows.all():
        X_raw = X_raw[valid_rows]
        y = df["Target"].values[valid_rows]
    else:
        y = df["Target"].values

    if len(X_raw) < LOOKBACK:
        raise ValueError(f"Insufficient data points: {len(X_raw)} < {LOOKBACK}")

    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_scaled = scaler.fit_transform(X_raw)
    else:
        X_scaled = scaler.transform(X_raw)

    # Verify no infinity values after scaling
    if np.isinf(X_scaled).any() or np.isnan(X_scaled).any():
        raise ValueError("Data contains NaN or infinity values after scaling")

    X_seq, y_seq = [], []
    for i in range(LOOKBACK, len(X_scaled)):
        X_seq.append(X_scaled[i - LOOKBACK:i])
        y_seq.append(y[i])

    return np.array(X_seq), np.array(y_seq), scaler


def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer="adam", 
        loss="binary_crossentropy", 
        metrics=["accuracy", "AUC"]
    )
    return model


def train_model(symbol: str, months: int = 10):
    """Full training pipeline. Returns (model, scaler, metrics)."""
    print(f"[train] Fetching {months} months of data for {symbol}...")
    df = fetch_historical_data(symbol, months)
    df = engineer_features(df)

    if USE_LSTM:
        X, y, scaler = prepare_sequences(df)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, y_train, epochs=50, batch_size=32,
                  validation_split=0.1, verbose=0)

        preds = (model.predict(X_test, verbose=0).flatten() > 0.5).astype(int)
        accuracy = float(np.mean(preds == y_test))

        metrics = {
            "accuracy": round(accuracy * 100, 2),
            "model_type": "LSTM",
            "data_points": len(df),
            "training_samples": len(X_train),
        }
    else:
        # Fallback: RandomForest on enhanced features
        feature_cols = [
            "Close", "MA7", "MA21", "MA50", "ROC", "RSI", "MACD", "MACD_Signal",
            "BB_Width", "Stoch_K", "Stoch_D", "ATR", "Vol_Change", "Vol_Ratio", "Close_Norm"
        ]
        X = df[feature_cols].values
        y = df["Target"].values
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        split = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:split], X_scaled[split:]
        y_train, y_test = y[:split], y[split:]

        model = RandomForestClassifier(
            n_estimators=150, 
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        metrics = {
            "accuracy": round(accuracy * 100, 2),
            "model_type": "RandomForest",
            "data_points": len(df),
            "training_samples": len(X_train),
        }

    # Save to S3
    try:
        upload_model(model, scaler, symbol)
        print(f"[train] Model for {symbol} saved locally.")
    except Exception as e:
        print(f"[train] S3 upload skipped: {e}")

    return model, scaler, metrics
