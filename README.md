# StockSense AI — AWS Deployment Guide

## Architecture Overview

```
Browser → Flask (EC2/Elastic Beanstalk)
              ├── yfinance  → Fetch historical OHLCV data
              ├── LSTM/RF   → Train model in-process
              ├── S3        → Store trained models (.pkl)
              └── DynamoDB  → (optional) Log predictions
```

---

## 1. Local Setup

```bash
# Clone and install
cd stock-predictor
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Copy and edit env
cp .env.example .env
# Fill in AWS credentials in .env

# Run locally
python app.py
# Visit http://localhost:5000
```

---

## 2. AWS S3 Setup

```bash
# Create bucket (Mumbai region — closest to Chennai/India)
aws s3 mb s3://stock-predictor-models --region ap-south-1

# Verify
aws s3 ls s3://stock-predictor-models
```

The app will auto-create folders:
- `models/<SYMBOL>/model_<timestamp>.pkl`
- `models/<SYMBOL>/scaler_<timestamp>.pkl`
- `models/<SYMBOL>/meta_<timestamp>.json`

---

## 3. Deploy to AWS Elastic Beanstalk (recommended)

```bash
# Install EB CLI
pip install awsebcli

# Initialize (choose Python 3.11)
eb init -p python-3.11 stock-predictor --region ap-south-1

# Create environment
eb create stock-predictor-env

# Deploy
eb deploy

# Open app
eb open
```

### Environment Variables to set in EB Console:
| Key | Value |
|-----|-------|
| `AWS_REGION` | `ap-south-1` |
| `AWS_ACCESS_KEY_ID` | your key |
| `AWS_SECRET_ACCESS_KEY` | your secret |
| `S3_BUCKET_NAME` | `stock-predictor-models` |
| `SECRET_KEY` | random string |
| `DEBUG` | `False` |

---

## 4. IAM Permissions

Your EC2/EB instance needs an IAM role with:
```json
{
  "Effect": "Allow",
  "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
  "Resource": ["arn:aws:s3:::stock-predictor-models/*",
               "arn:aws:s3:::stock-predictor-models"]
}
```

> **Tip**: Use instance profile instead of hardcoding keys — boto3 auto-picks credentials from the instance role.

---

## 5. Optional: SageMaker for heavy training

For production-scale training jobs, `aws/sagemaker_train.py` can offload LSTM training to a `ml.m5.large` SageMaker instance. This avoids blocking the Flask server during training.

---

## 6. Project Structure

```
stock-predictor/
├── app.py               # Flask routes
├── config.py            # AWS + model config
├── requirements.txt
├── ml/
│   ├── train.py         # Data fetch + model training
│   └── predict.py       # Inference + signal logic
├── aws/
│   └── s3_utils.py      # S3 upload/download
└── templates/
    ├── base.html
    ├── index.html        # Stock picker
    ├── result.html       # BUY/SELL/HOLD result
    ├── history.html      # Saved models
    └── error.html
```

---

## Signal Logic

| Probability | Signal |
|-------------|--------|
| ≥ 65% up    | **BUY** |
| ≤ 35% up    | **SELL** |
| 35–65%      | **HOLD** |

---

## Disclaimer

> This tool is for educational purposes only and does not constitute financial advice.
