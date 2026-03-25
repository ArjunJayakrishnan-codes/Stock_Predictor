import os

class Config:
    # App
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key")
    DEBUG      = os.environ.get("DEBUG", "True") == "True"

    # Model hyperparameters
    LOOKBACK_DAYS    = 60
    TRAINING_MONTHS  = 10
    EPOCHS           = 50
    BATCH_SIZE       = 32
    LSTM_UNITS       = 64
