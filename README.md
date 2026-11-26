
# Cryptocurrency Price Prediction 📈

## Project Overview

Cryptocurrency Price Prediction is a Python-based data analytics and machine learning project that analyzes historical cryptocurrency market data and estimates the next day's Bitcoin closing price.

The project uses Yahoo Finance data and explores machine learning regression algorithms to evaluate cryptocurrency price prediction performance.

The final Streamlit application provides cryptocurrency market visualization and a baseline Bitcoin price estimation.

## Features

- Cryptocurrency historical data collection.
- Data cleaning and preprocessing.
- Exploratory Data Analysis (EDA).
- Correlation and outlier analysis.
- Feature engineering using lag features, moving averages, returns and volatility.
- Machine learning model training and evaluation.
- Actual vs. predicted price visualization.
- Interactive Streamlit dashboard.
- Bitcoin next-day baseline price estimation.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Yahoo Finance API (yfinance)
- Streamlit
- Plotly
- Pickle

## Cryptocurrencies Analyzed

- Bitcoin (BTC)
- Ethereum (ETH)
- Tether (USDT)
- Binance Coin (BNB)

## Machine Learning Algorithms

The following regression algorithms were evaluated:

1. Linear Regression
2. Ridge Regression
3. Random Forest Regressor
4. Gradient Boosting Regressor
5. XGBoost Regressor

## Model Evaluation

Models were evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Mean Absolute Percentage Error (MAPE)

The tested machine learning models did not outperform the persistence baseline.

The baseline predicts the next day's Bitcoin closing price using the latest available closing price.

This result highlights the importance of comparing machine learning models against simple benchmarks.

## Application Pages

1. Home – Project introduction and features.
2. Live Market – Cryptocurrency market data and price charts.
3. Bitcoin Prediction – Next-day baseline price estimation.
4. About – Prediction strategy and evaluation metrics.

## Project Structure

```text
Crypto-Price-Prediction/
│
├── app.py
├── crypto_price_prediction.ipynb
├── best_crypto_model.pkl
├── requirements.txt
├── README.md
├── .gitignore
│
└── images/
    └── dashboard.png
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Crypto-Price-Prediction.git
```

### 2. Navigate to the project folder

```bash
cd Crypto-Price-Prediction
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

## Future Improvements

- Improve predictive performance through additional feature engineering.
- Evaluate additional time-series forecasting algorithms.
- Perform hyperparameter optimization.
- Add more cryptocurrency forecasting options.
- Implement automated model monitoring.
- Deploy the application publicly.

## Disclaimer

This project is developed for educational and portfolio purposes.

Cryptocurrency markets are volatile. The baseline estimate is not a guaranteed prediction or financial advice.

## Author

**Yashwanth**

Aspiring AI/ML Engineer
