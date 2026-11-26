from fastapi import FastAPI, HTTPException
import yfinance as yf
import pandas as pd
import pickle

app = FastAPI(
    title="Cryptocurrency Prediction API",
    description="REST API for cryptocurrency market analysis and Bitcoin price estimation",
    version="1.0.0"
)

# Load baseline strategy
try:
    with open("best_crypto_model.pkl", "rb") as file:
        model_info = pickle.load(file)

    if model_info["model_name"] != "Persistence Baseline":
        raise ValueError("Unexpected prediction strategy")

except Exception as e:
    raise RuntimeError(f"Unable to load prediction strategy: {e}")


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Welcome to Cryptocurrency Prediction API",
        "status": "Running"
    }


# Health check
@app.get("/health")
def health():
    return {
        "status": "Healthy",
        "service": "Crypto Prediction API"
    }


# Market data endpoint
@app.get("/market/{ticker}")
def get_market_data(ticker: str):

    allowed_tickers = ["BTC-USD", "ETH-USD", "USDT-USD", "BNB-USD"]

    ticker = ticker.upper()

    if ticker not in allowed_tickers:
        raise HTTPException(
            status_code=400,
            detail="Unsupported cryptocurrency ticker"
        )

    try:
        data = yf.download(
            ticker,
            period="1mo",
            interval="1d",
            progress=False
        )

        if data.empty:
            raise HTTPException(
                status_code=503,
                detail="Market data unavailable"
            )

        close = data["Close"]

        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        close = close.dropna()
        
        
        
        history = [
            {
                "date": date.strftime("%Y-%m-%d"),
                "price": float(price)
            }
            for date, price in close.items()
        ]


        return {
            "ticker": ticker,
            "latest_price": float(close.iloc[-1]),
            "currency": "USD",
            "historical_prices": history
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    


# Bitcoin prediction endpoint
@app.get("/predict/btc")
def predict_bitcoin():

    try:
        data = yf.download(
            "BTC-USD",
            period="5d",
            interval="1d",
            progress=False
        )

        if data.empty:
            raise HTTPException(
                status_code=503,
                detail="Bitcoin data unavailable"
            )

        close = data["Close"]

        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        close = close.dropna()

        latest_price = float(close.iloc[-1])

        return {
            "cryptocurrency": "Bitcoin",
            "latest_closing_price": latest_price,
            "predicted_next_day_price": latest_price,
            "prediction_strategy": "Persistence Baseline",
            "currency": "USD",
            "disclaimer": "Baseline estimate, not a guaranteed forecast."
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Model evaluation endpoint
@app.get("/model/metrics")
def model_metrics():

    return {
        "model_name": model_info["model_name"],
        "MAE": model_info["mae"],
        "RMSE": model_info["rmse"],
        "R2_score": model_info["r2"]
    }