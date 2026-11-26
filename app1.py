
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import pickle

# Page configuration
st.set_page_config(
    page_title="Crypto Price Prediction",
    page_icon="₿",
    layout="wide"
)

# Load baseline strategy
try:
    with open("best_crypto_model.pkl", "rb") as file:
        model_info = pickle.load(file)

    assert model_info["model_name"] == "Persistence Baseline"

except Exception as e:
    st.error(f"Strategy loading failed: {e}")
    st.stop()

# Sidebar
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    ["Home", "Live Market", "Bitcoin Prediction", "About"]
)

crypto_tickers = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Tether": "USDT-USD",
    "BNB": "BNB-USD"
}

# Home
if page == "Home":

    st.title("₿ Cryptocurrency Price Prediction")

    st.write(
        "Cryptocurrency market analysis and "
        "Bitcoin next-day baseline estimation."
    )

    st.success("Application is running successfully!")

    st.subheader("Project Features")

    st.write("- Live cryptocurrency market data")
    st.write("- Historical price visualization")
    st.write("- Bitcoin next-day price estimation")
    st.write("- Model evaluation metrics")

# Live Market
elif page == "Live Market":

    st.title("Live Cryptocurrency Market")

    selected_crypto = st.selectbox(
        "Select Cryptocurrency",
        list(crypto_tickers.keys())
    )

    if st.button("Get Market Data"):

        try:
            market_data = yf.download(
                crypto_tickers[selected_crypto],
                period="1mo",
                interval="1d",
                progress=False
            )

            if market_data.empty:
                st.error("Market data unavailable.")
            else:
                close = market_data["Close"]

                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]

                close = close.dropna()

                st.metric(
                    "Latest Closing Price",
                    f"${float(close.iloc[-1]):,.2f}"
                )

                chart_data = close.rename(
                    "Closing Price"
                ).reset_index()

                fig = px.line(
                    chart_data,
                    x=chart_data.columns[0],
                    y="Closing Price",
                    title=f"{selected_crypto} Price History"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"Market data error: {e}")

# Bitcoin Prediction
elif page == "Bitcoin Prediction":

    st.title("Bitcoin Next-Day Price Estimation")

    st.info(
        "Baseline strategy: Tomorrow's estimated price "
        "equals today's latest available closing price."
    )

    if st.button("Generate Prediction"):

        try:
            btc_data = yf.download(
                "BTC-USD",
                period="5d",
                interval="1d",
                progress=False
            )

            if btc_data.empty:
                st.error("Bitcoin data unavailable.")
            else:
                close = btc_data["Close"]

                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]

                close = close.dropna()

                latest_price = float(close.iloc[-1])

                st.metric(
                    "Latest BTC Closing Price",
                    f"${latest_price:,.2f}"
                )

                st.metric(
                    "Estimated Next-Day Price",
                    f"${latest_price:,.2f}"
                )

                st.caption(
                    "This is a simple baseline estimate, "
                    "not a guaranteed forecast."
                )

        except Exception as e:
            st.error(f"Prediction error: {e}")

# About
elif page == "About":

    st.title("About This Project")

    st.write("Cryptocurrency Price Prediction")

    st.write(
        "Prediction Strategy:",
        model_info["model_name"]
    )

    st.write(
        "Evaluation MAE:",
        f"${model_info['mae']:,.2f}"
    )

    st.write(
        "Evaluation RMSE:",
        f"${model_info['rmse']:,.2f}"
    )

    st.write(
        "Evaluation R²:",
        round(model_info["r2"], 4)
    )

    st.warning(
        "Historical evaluation metrics do not guarantee "
        "future performance."
    )

st.sidebar.markdown("---")
st.sidebar.caption("Cryptocurrency Analytics Project")
