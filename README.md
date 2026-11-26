# Cryptocurrency Price Prediction and Market Analysis

A Python-based cryptocurrency market analysis project that combines data science, machine learning experimentation, a Streamlit dashboard, and a FastAPI backend to analyze cryptocurrency prices and estimate the next day's Bitcoin closing price.

## Project Overview

This project analyzes historical cryptocurrency market data collected using Yahoo Finance (`yfinance`).

It focuses on four cryptocurrencies:

- Bitcoin (BTC)
- Ethereum (ETH)
- Tether (USDT)
- Binance Coin (BNB)

The project includes data preprocessing, exploratory data analysis, feature engineering, machine learning model comparison, and an interactive dashboard.

An important finding from model evaluation was that the simple Persistence Baseline performed better than the tested machine learning models on the evaluation data.

## Key Features

- Historical cryptocurrency market data collection
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Correlation and outlier analysis
- Feature engineering using lag features and moving averages
- Machine learning model comparison
- Model evaluation using regression metrics
- Interactive Streamlit dashboard
- FastAPI REST backend
- Bitcoin next-day baseline price estimation
- Technical indicators and market visualizations

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning |
| XGBoost | Gradient boosting model |
| Yahoo Finance / yfinance | Cryptocurrency market data |
| Streamlit | Interactive dashboard |
| FastAPI | REST API backend |
| Uvicorn | API server |
| Plotly | Interactive charts |
| Requests | Frontend API communication |
| Pickle | Saving baseline evaluation information |

## Machine Learning Models Evaluated

The following regression algorithms were evaluated:

1. Linear Regression
2. Ridge Regression
3. Random Forest Regressor
4. Gradient Boosting Regressor
5. XGBoost Regressor

### Evaluation Metrics

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Mean Absolute Percentage Error (MAPE)

### Important Model Evaluation Finding

The Persistence Baseline was used as a benchmark:

**Tomorrow's estimated Bitcoin price = Today's closing price**

The tested machine learning models did not outperform this baseline on the evaluated test data.

This demonstrates why machine learning models should always be compared against simple baseline approaches.

The baseline is used by the current application for its Bitcoin price estimate.

## System Architecture

The application has two main components:

**1. Streamlit Frontend**

Provides the interactive user interface, market visualizations, prediction display, and technical analysis.

**2. FastAPI Backend**

Provides REST API endpoints to retrieve cryptocurrency market data, generate the baseline Bitcoin estimate, and return evaluation metrics.

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API welcome message |
| GET | `/health` | API health check |
| GET | `/market/{ticker}` | Cryptocurrency market data |
| GET | `/predict/btc` | Bitcoin baseline estimate |
| GET | `/model/metrics` | Model evaluation metrics |

Interactive API documentation is available at:

`http://127.0.0.1:8000/docs`

## Project Structure

```text
Crypto-Price-Prediction/
│
├── app.py
├── app1.py
├── main.py
│
├── crypto_price_prediction.ipynb
│
├── best_crypto_model.pkl
├── random_forest_model.pkl
├── scaler.pkl
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/yashwanth86868/Crypto-Price-Prediction.git
```

### 2. Navigate to the project directory

```bash
cd Crypto-Price-Prediction
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Start the FastAPI backend

Open the first terminal:

```bash
python -m uvicorn main:app --reload
```

Backend address:

`http://127.0.0.1:8000`

API documentation:

`http://127.0.0.1:8000/docs`

### 6. Start the Streamlit frontend

Open a second terminal in the same project directory:

```bash
python -m streamlit run app.py
```

The dashboard will open in your browser.

Keep the FastAPI backend running while using the Streamlit application.

## Project Learning Outcomes

Through this project, I practiced:

- Working with real-world financial datasets
- Data preprocessing and exploratory analysis
- Feature engineering for time-series data
- Training and comparing regression models
- Evaluating models using appropriate metrics
- Building REST APIs using FastAPI
- Connecting a frontend application to a backend API
- Creating interactive dashboards using Streamlit
- Organizing and maintaining a GitHub project

## Future Improvements

- Experiment with additional time-series forecasting techniques
- Improve feature engineering
- Add automated testing
- Improve API error handling
- Add more cryptocurrency analysis features
- Compare forecasting models across different time periods

## Disclaimer

This project is developed for educational and portfolio purposes only.

Cryptocurrency markets are highly volatile. The baseline estimate is not a guaranteed forecast and should not be considered financial advice.

## Author

**Yashwanth**

Aspiring AI/ML Engineer

GitHub: https://github.com/yashwanth86868
