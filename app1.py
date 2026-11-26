import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Crypto Analytics Dashboard",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://127.0.0.1:8000"

CRYPTO_TICKERS = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Tether": "USDT-USD",
    "BNB": "BNB-USD"
}


# ==========================================
# 2. API HELPER FUNCTIONS
# ==========================================

def api_get(endpoint):

    try:
        response = requests.get(
            f"{API_URL}{endpoint}",
            timeout=30
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException as error:
        st.error(f"API Error: {error}")
        return None


def check_api_connection():

    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=5
        )

        return response.status_code == 200

    except requests.RequestException:
        return False


def get_market_history(ticker):

    market_data = api_get(f"/market/{ticker}")

    if market_data is None:
        return None

    historical_prices = market_data.get("historical_prices")

    if not historical_prices:
        st.warning("No historical market data available.")
        return None

    df = pd.DataFrame(historical_prices)

    if not {"date", "price"}.issubset(df.columns):
        st.error("Unexpected market data format.")
        return None

    df["date"] = pd.to_datetime(df["date"])

    df["price"] = pd.to_numeric(
        df["price"],
        errors="coerce"
    )

    df = (
        df.dropna()
        .sort_values("date")
        .reset_index(drop=True)
    )

    if df.empty:
        st.warning("No valid price records found.")
        return None

    return market_data, df


# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("₿ Crypto Analytics")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Live Market",
        "Bitcoin Prediction",
        "Model Performance",
        "Technical Analysis",
        "About"
    ]
)

st.sidebar.markdown("---")

if check_api_connection():
    st.sidebar.success("🟢 API Connected")
else:
    st.sidebar.error("🔴 API Disconnected")

st.sidebar.caption(
    "Cryptocurrency Analytics Project"
)


# ==========================================
# 4. HOME DASHBOARD
# ==========================================

if page == "Home":

    st.title("₿ Cryptocurrency Analytics Dashboard")

    st.markdown(
        """
        Welcome to the Cryptocurrency Analytics and
        Bitcoin Price Estimation Platform.

        This project combines Python, Machine Learning,
        FastAPI and Streamlit.
        """
    )

    st.markdown("---")

    st.subheader("Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Cryptocurrencies", "4")
    col2.metric("ML Algorithms Evaluated", "5")
    col3.metric("Backend", "FastAPI")
    col4.metric("Frontend", "Streamlit")

    st.markdown("---")

    st.subheader("Available Features")

    col1, col2 = st.columns(2)

    with col1:

        st.info(
            """
            **Live Market Analysis**

            - Cryptocurrency prices
            - Historical price charts
            - Market statistics
            - Price trends
            """
        )

        st.info(
            """
            **Bitcoin Price Estimation**

            - Latest BTC closing price
            - Next-day baseline estimate
            - Prediction strategy
            """
        )

    with col2:

        st.info(
            """
            **Model Performance**

            - MAE
            - RMSE
            - R² Score
            - Model evaluation
            """
        )

        st.info(
            """
            **Technical Analysis**

            - SMA
            - EMA
            - RSI
            - Daily returns
            - Volatility
            """
        )

    st.markdown("---")

    st.caption(
        "Educational project. Not financial advice."
    )


# ==========================================
# 5. LIVE MARKET DASHBOARD
# ==========================================

elif page == "Live Market":

    st.title("📊 Live Cryptocurrency Market")

    selected_crypto = st.selectbox(
        "Select Cryptocurrency",
        list(CRYPTO_TICKERS.keys())
    )

    ticker = CRYPTO_TICKERS[selected_crypto]

    if st.button("Load Market Data"):

        result = get_market_history(ticker)

        if result is not None:

            market_data, history = result

            latest_price = float(
                market_data["latest_price"]
            )

            previous_price = (
                history["price"].iloc[-2]
                if len(history) >= 2
                else latest_price
            )

            daily_change = (
                (latest_price - previous_price)
                / previous_price
            ) * 100 if previous_price else 0

            st.markdown("---")

            st.subheader(
                f"{selected_crypto} Market Overview"
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Latest Price",
                f"${latest_price:,.2f}",
                f"{daily_change:.2f}%"
            )

            col2.metric(
                "Highest Price",
                f"${history['price'].max():,.2f}"
            )

            col3.metric(
                "Lowest Price",
                f"${history['price'].min():,.2f}"
            )

            st.markdown("---")

            st.subheader("Historical Price Chart")

            fig = px.line(
                history,
                x="date",
                y="price",
                title=f"{selected_crypto} Price History",
                labels={
                    "date": "Date",
                    "price": "Price (USD)"
                }
            )

            fig.update_layout(
                hovermode="x unified"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.subheader("Market Statistics")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Average Price",
                f"${history['price'].mean():,.2f}"
            )

            col2.metric(
                "Price Range",
                f"${history['price'].max() - history['price'].min():,.2f}"
            )

            col3.metric(
                "Available Records",
                len(history)
            )

            st.subheader("Historical Data")

            st.dataframe(
                history.sort_values(
                    "date",
                    ascending=False
                ),
                use_container_width=True
            )


# ==========================================
# 6. BITCOIN PRICE PREDICTION
# ==========================================

elif page == "Bitcoin Prediction":

    st.title("₿ Bitcoin Next-Day Price Estimation")

    st.info(
        """
        Prediction strategy:

        Persistence Baseline

        Tomorrow's estimated price = Today's latest
        available closing price.
        """
    )

    if st.button("Generate Bitcoin Estimate"):

        prediction_data = api_get("/predict/btc")

        if prediction_data is not None:

            latest_price = float(
                prediction_data["latest_closing_price"]
            )

            predicted_price = float(
                prediction_data["predicted_next_day_price"]
            )

            col1, col2 = st.columns(2)

            col1.metric(
                "Latest BTC Closing Price",
                f"${latest_price:,.2f}"
            )

            col2.metric(
                "Estimated Next-Day Price",
                f"${predicted_price:,.2f}"
            )

            st.success(
                "Prediction generated successfully!"
            )

            st.write(
                "Strategy:",
                prediction_data["prediction_strategy"]
            )

            st.warning(
                "This is a persistence baseline estimate, "
                "not a trained machine-learning forecast "
                "or a guaranteed future price."
            )


# ==========================================
# 7. MODEL PERFORMANCE DASHBOARD
# ==========================================

elif page == "Model Performance":

    st.title("📊 Model Performance Dashboard")

    metrics = api_get("/model/metrics")

    if metrics is not None:

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "MAE",
            f"${metrics['MAE']:,.2f}"
        )

        col2.metric(
            "RMSE",
            f"${metrics['RMSE']:,.2f}"
        )

        col3.metric(
            "R² Score",
            f"{metrics['R2_score']:.4f}"
        )

        st.markdown("---")

        st.subheader("Selected Prediction Strategy")

        st.success(metrics["model_name"])

        st.write(
            f"Mean Absolute Error: ${metrics['MAE']:,.2f}"
        )

        st.write(
            f"Root Mean Squared Error: ${metrics['RMSE']:,.2f}"
        )

        st.write(
            f"R² Score: {metrics['R2_score']:.4f}"
        )

        st.markdown("---")

        st.subheader("Model Evaluation Explanation")

        st.write(
            """
            **MAE:** Measures average absolute prediction error.

            **RMSE:** Penalizes larger prediction errors.

            **R²:** Measures how much variation in the
            target variable is explained by the model.
            """
        )

        st.warning(
            """
            The persistence baseline performed better than
            the tested ML models in our evaluation.

            The displayed metrics belong to the baseline,
            not to a trained ML estimator.
            """
        )


# ==========================================
# 8. TECHNICAL ANALYSIS DASHBOARD
# ==========================================

elif page == "Technical Analysis":

    st.title("📈 Bitcoin Technical Analysis Dashboard")

    result = get_market_history("BTC-USD")

    if result is not None:

        market_data, df = result

        # Simple Moving Averages
        df["SMA_7"] = (
            df["price"].rolling(window=7).mean()
        )

        df["SMA_14"] = (
            df["price"].rolling(window=14).mean()
        )

        # Exponential Moving Average
        df["EMA_7"] = (
            df["price"].ewm(
                span=7,
                adjust=False
            ).mean()
        )

        # Daily Returns
        df["Daily_Return"] = (
            df["price"].pct_change() * 100
        )

        # Rolling Volatility
        df["Volatility"] = (
            df["Daily_Return"].rolling(7).std()
        )

        # RSI Calculation
        delta = df["price"].diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(
            alpha=1/14,
            min_periods=14,
            adjust=False
        ).mean()

        avg_loss = loss.ewm(
            alpha=1/14,
            min_periods=14,
            adjust=False
        ).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)

        df["RSI"] = 100 - (100 / (1 + rs))

        # Handle edge cases
        df.loc[
            (avg_loss == 0) & (avg_gain > 0),
            "RSI"
        ] = 100

        df.loc[
            (avg_gain == 0) & (avg_loss > 0),
            "RSI"
        ] = 0

        # KPI section
        st.subheader("Technical Indicators")

        col1, col2, col3 = st.columns(3)

        sma_value = df["SMA_7"].iloc[-1]
        ema_value = df["EMA_7"].iloc[-1]
        rsi_value = df["RSI"].iloc[-1]

        col1.metric(
            "SMA 7",
            f"${sma_value:,.2f}"
            if pd.notna(sma_value) else "N/A"
        )

        col2.metric(
            "EMA 7",
            f"${ema_value:,.2f}"
            if pd.notna(ema_value) else "N/A"
        )

        col3.metric(
            "RSI 14",
            f"{rsi_value:.2f}"
            if pd.notna(rsi_value) else "N/A"
        )

        st.markdown("---")

        # Price and Moving Averages
        st.subheader("Bitcoin Price and Moving Averages")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["price"],
                mode="lines",
                name="BTC Price"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["SMA_7"],
                mode="lines",
                name="SMA 7"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["SMA_14"],
                mode="lines",
                name="SMA 14"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["EMA_7"],
                mode="lines",
                name="EMA 7"
            )
        )

        fig.update_layout(
            title="BTC Price Trend Analysis",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            hovermode="x unified"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # RSI Chart
        st.subheader("Relative Strength Index")

        rsi_fig = px.line(
            df,
            x="date",
            y="RSI",
            title="Bitcoin RSI (14)"
        )

        rsi_fig.add_hline(
            y=70,
            line_dash="dash",
            annotation_text="Overbought reference"
        )

        rsi_fig.add_hline(
            y=30,
            line_dash="dash",
            annotation_text="Oversold reference"
        )

        rsi_fig.update_yaxes(
            range=[0, 100]
        )

        st.plotly_chart(
            rsi_fig,
            use_container_width=True
        )

        # Daily Returns
        st.subheader("Daily Returns")

        return_fig = px.bar(
            df,
            x="date",
            y="Daily_Return",
            title="Bitcoin Daily Percentage Returns"
        )

        st.plotly_chart(
            return_fig,
            use_container_width=True
        )

        # Volatility
        st.subheader("Rolling Volatility")

        volatility_fig = px.line(
            df,
            x="date",
            y="Volatility",
            title="7-Day Rolling Volatility"
        )

        st.plotly_chart(
            volatility_fig,
            use_container_width=True
        )

        # Data Table
        st.subheader("Technical Indicator Data")

        st.dataframe(
            df.sort_values(
                "date",
                ascending=False
            ),
            use_container_width=True
        )

        st.caption(
            "Technical indicators are historical analytical "
            "measures, not guaranteed trading signals."
        )


# ==========================================
# 9. ABOUT PROJECT
# ==========================================

elif page == "About":

    st.title("About This Project")

    st.subheader(
        "Cryptocurrency Analytics and Price Estimation"
    )

    st.write(
        """
        This project analyzes cryptocurrency historical
        market data and provides Bitcoin next-day baseline
        price estimation.
        """
    )

    st.markdown("### Technologies Used")

    technologies = [
        "Python",
        "Pandas",
        "NumPy",
        "Matplotlib",
        "Seaborn",
        "Scikit-learn",
        "XGBoost",
        "Yahoo Finance",
        "FastAPI",
        "Streamlit",
        "Plotly"
    ]

    for technology in technologies:
        st.write(f"- {technology}")

    st.markdown("---")

    st.subheader("Machine Learning Algorithms Evaluated")

    st.write(
        """
        - Linear Regression
        - Ridge Regression
        - Random Forest
        - Gradient Boosting
        - XGBoost
        """
    )

    st.markdown("---")

    st.subheader("Project Disclaimer")

    st.warning(
        """
        This application is developed for educational
        and portfolio purposes.

        Cryptocurrency markets are volatile. Historical
        performance does not guarantee future results.

        This application does not provide financial advice.
        """
    )

    st.caption(
        "Developed as an AI/ML Engineering portfolio project."
    )


# ==========================================
# 10. FOOTER
# ==========================================

st.sidebar.markdown("---")

st.sidebar.caption(
    "Cryptocurrency Analytics | Python + FastAPI + Streamlit"
)