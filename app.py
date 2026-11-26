# import streamlit as st
# import pandas as pd
# import pickle
# from sklearn.ensemble import RandomForestRegressor

# # Load the trained model
# model_path = 'random_forest_model.pkl'
# with open(model_path, 'rb') as file:
#     model_rf = pickle.load(file)

# # Function to predict based on user inputs
# def predict_btc_price(input_data):
#     # Make prediction using the model
#     prediction = model_rf.predict(input_data)
#     return prediction[0]  # Assuming model returns a single prediction

# def main():
#     # Title of your web app
#     st.title('Predict BTC Close Price')

#     # Sidebar for user inputs
#     st.sidebar.title('Input Features')
    
#     # Inputs for USDT, BNB closing prices and volumes
#     usdt_close = st.sidebar.number_input('USDT Close Price', min_value=0.0, format="%.2f")
#     usdt_volume = st.sidebar.number_input('USDT Volume', min_value=0.0, format="%.2f")
#     bnb_close = st.sidebar.number_input('BNB Close Price', min_value=0.0, format="%.2f")
#     bnb_volume = st.sidebar.number_input('BNB Volume', min_value=0.0, format="%.2f")

#     # Create input dataframe
#     input_data = pd.DataFrame({
#         'USDT_Close': [usdt_close],
#         'USDT_Volume': [usdt_volume],
#         'BNB_Close': [bnb_close],
#         'BNB_Volume': [bnb_volume]
#     })

#     # Button to trigger prediction
#     if st.button('Predict BTC Close Price'):
#         predicted_price = predict_btc_price(input_data)
#         st.write('Predicted BTC Close Price:', predicted_price)

# if __name__ == '__main__':
#     main()





import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import os

from datetime import datetime, timezone


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Crypto Price Prediction",
    page_icon="₿",
    layout="wide"
)

MODEL_PATH = "best_crypto_model.pkl"

SYMBOLS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "USDT": "USDT-USD",
    "BNB": "BNB-USD"
}


# =========================================================
# 2. LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "best_crypto_model.pkl not found."
        )

    if os.path.getsize(MODEL_PATH) == 0:
        raise ValueError("Model file is empty.")

    with open(MODEL_PATH, "rb") as file:
        bundle = pickle.load(file)

    required_keys = [
        "model",
        "feature_selector",
        "scaler",
        "feature_columns",
        "selected_features",
        "model_name"
    ]

    for key in required_keys:
        if key not in bundle:
            raise KeyError(f"Missing key: {key}")

    return bundle


try:

    bundle = load_model()

    model = bundle["model"]
    selector = bundle["feature_selector"]
    scaler = bundle["scaler"]

    feature_columns = bundle["feature_columns"]

except Exception as error:

    st.error(f"Model loading failed: {error}")
    st.stop()


# =========================================================
# 3. FETCH CRYPTOCURRENCY DATA
# =========================================================

@st.cache_data(ttl=900)
def get_crypto_data():

    raw_data = yf.download(
        tickers=list(SYMBOLS.values()),
        period="1mo",
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=False
    )

    if raw_data.empty:
        raise ValueError(
            "Yahoo Finance returned empty data. "
            "Please try again later."
        )

    if not isinstance(raw_data.columns, pd.MultiIndex):
        raise ValueError(
            "Unexpected Yahoo Finance data format."
        )

    crypto_frames = []

    for name, symbol in SYMBOLS.items():

        if symbol in raw_data.columns.get_level_values(0):

            crypto_df = raw_data[symbol].copy()

        elif symbol in raw_data.columns.get_level_values(1):

            crypto_df = raw_data.xs(
                symbol,
                axis=1,
                level=1
            ).copy()

        else:

            raise ValueError(
                f"Data unavailable for {symbol}"
            )

        if not {"Close", "Volume"}.issubset(
            crypto_df.columns
        ):
            raise ValueError(
                f"Close or Volume missing for {symbol}"
            )

        crypto_df = crypto_df[
            ["Close", "Volume"]
        ]

        crypto_df.columns = [
            f"Close ({name})",
            f"Volume ({name})"
        ]

        crypto_frames.append(crypto_df)

    combined_data = pd.concat(
        crypto_frames,
        axis=1,
        join="inner"
    )

    combined_data.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    combined_data.dropna(inplace=True)

    # Remove incomplete current-day candle
    today = datetime.now(timezone.utc).date()

    combined_data = combined_data[
        combined_data.index.date < today
    ]

    if combined_data.empty:
        raise ValueError(
            "No completed common market data found."
        )

    latest_date = combined_data.index[-1]

    return combined_data, latest_date


# =========================================================
# 4. PREPARE MODEL INPUT
# =========================================================

def prepare_input(market_data):

    latest_data = market_data.iloc[[-1]].copy()

    latest_data.columns = [
        column + "_Previous_Day"
        for column in latest_data.columns
    ]

    input_df = latest_data.reindex(
        columns=feature_columns
    )

    if input_df.isna().any().any():

        missing = input_df.columns[
            input_df.isna().any()
        ].tolist()

        raise ValueError(
            f"Missing features: {missing}"
        )

    return input_df


# =========================================================
# 5. MODEL PREDICTION AND VALIDATION
# =========================================================

def predict_btc_price(input_df):

    # Feature selection
    selected_input = selector.transform(input_df)

    # Check feature names and count
    if selected_input.shape[1] != scaler.n_features_in_:
        raise ValueError(
            "Selected feature count does not match scaler."
        )

    # Check whether live inputs are outside training ranges
    below_min = selected_input < scaler.data_min_
    above_max = selected_input > scaler.data_max_

    out_of_range = below_min | above_max

    # Scaling
    scaled_input = scaler.transform(selected_input)

    # Generate raw prediction for diagnostics
    prediction = model.predict(scaled_input)[0]

    if not np.isfinite(prediction):
        raise ValueError("Model produced an invalid prediction.")

    return (
        prediction,
        selected_input,
        scaled_input,
        out_of_range
    )


# =========================================================
# 6. APPLICATION HEADER
# =========================================================

st.title("₿ Cryptocurrency Price Prediction")

st.subheader(
    "Machine Learning Based Bitcoin Price Forecasting"
)

st.info(
    f"Selected ML Model: {bundle['model_name']}"
)


# =========================================================
# 7. SIDEBAR
# =========================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a page",
    [
        "Home",
        "Live Market",
        "Bitcoin Prediction"
    ]
)

st.sidebar.markdown("---")

st.sidebar.write(
    "Built with Python and Machine Learning"
)


# =========================================================
# 8. HOME PAGE
# =========================================================

if page == "Home":

    st.header("Welcome!")

    st.write("""
    This application predicts the next-day
    Bitcoin closing price using Machine Learning.

    Cryptocurrencies included:

    - Bitcoin (BTC)
    - Ethereum (ETH)
    - Tether (USDT)
    - Binance Coin (BNB)
    """)

    st.success("Model loaded successfully!")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Selected ML Model",
            bundle["model_name"]
        )

    with col2:
        st.metric(
            "Prediction Target",
            "BTC Closing Price"
        )

    st.subheader("Technologies Used")

    st.write("""
    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - Yahoo Finance
    - Streamlit
    """)


# =========================================================
# 9. LIVE MARKET PAGE
# =========================================================

elif page == "Live Market":

    st.header("Cryptocurrency Market")

    st.caption(
        "Latest available completed daily market data."
    )

    if st.button("Refresh Market Data"):
        get_crypto_data.clear()

    try:

        with st.spinner("Fetching market data..."):

            market_data, latest_date = get_crypto_data()

        latest = market_data.iloc[-1]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Bitcoin",
            f"${latest['Close (BTC)']:,.2f}"
        )

        col2.metric(
            "Ethereum",
            f"${latest['Close (ETH)']:,.2f}"
        )

        col3.metric(
            "Tether",
            f"${latest['Close (USDT)']:,.4f}"
        )

        col4.metric(
            "BNB",
            f"${latest['Close (BNB)']:,.2f}"
        )

        st.write("Market date:", latest_date)

        st.subheader("Historical Closing Prices")

        st.line_chart(
            market_data[
                [
                    "Close (BTC)",
                    "Close (ETH)",
                    "Close (USDT)",
                    "Close (BNB)"
                ]
            ]
        )

        st.subheader("Historical Data")

        st.dataframe(market_data)

    except Exception as error:

        st.error(f"Market data error: {error}")


# =========================================================
# 10. BITCOIN PREDICTION PAGE
# =========================================================


elif page == "Bitcoin Prediction":

    st.header("Bitcoin Price Prediction")

    st.write(
        "Automatically fetch cryptocurrency data "
        "and predict the next-day BTC closing price."
    )

    if st.button("Fetch Market Data and Predict", type="primary"):

        try:

            with st.spinner("Fetching market data..."):
                market_data, latest_date = get_crypto_data()

            with st.spinner("Preparing features..."):
                input_df = prepare_input(market_data)

            with st.spinner("Running ML model..."):
                (
                    prediction,
                    selected_input,
                    scaled_input,
                    out_of_range
                ) = predict_btc_price(input_df)

            st.subheader("Fetched Market Data")
            st.dataframe(input_df)

            st.write("Market Data Date:", latest_date)

            with st.expander("Model Diagnostics", expanded=True):

                st.write("Model:", bundle["model_name"])

                diagnostics = pd.DataFrame({
                    "Feature": bundle["selected_features"],
                    "Live Value": selected_input.flatten(),
                    "Training Minimum": scaler.data_min_,
                    "Training Maximum": scaler.data_max_
                })

                diagnostics["Outside Training Range"] = (
                    (diagnostics["Live Value"] < diagnostics["Training Minimum"]) |
                    (diagnostics["Live Value"] > diagnostics["Training Maximum"])
                )

                st.dataframe(
                    diagnostics,
                    use_container_width=True
                )

                if hasattr(model, "coef_"):
                    coefficients = pd.DataFrame({
                        "Feature": bundle["selected_features"],
                        "Coefficient": model.coef_
                    })

                    st.write("Model Coefficients:")
                    st.dataframe(coefficients)

                st.write("Raw Prediction:", prediction)

            if out_of_range.any():

                st.error(
                    "Live features are outside the training data range."
                )

                st.warning(
                    "Do not trust this prediction until the model "
                    "and input data are investigated."
                )

            else:

                target_min = bundle.get("target_min")
                target_max = bundle.get("target_max")

                if target_min is None or target_max is None:

                    st.warning(
                        "Training target range is not saved "
                        "in the model bundle."
                    )

                elif not target_min <= prediction <= target_max:

                    st.error(
                        "Prediction is outside the training target range."
                    )

                else:

                    st.success("Prediction completed!")

                    st.metric(
                        "Predicted Next-Day BTC Closing Price",
                        f"${prediction:,.2f}"
                    )

            st.caption(
                "Experimental ML prediction. Not financial advice."
            )

        except Exception as error:

            st.error(f"Prediction failed: {error}")



# =========================================================
# 11. FOOTER
# =========================================================

st.sidebar.markdown("---")

st.sidebar.caption(
    "Crypto Price Prediction | Python + ML + Streamlit"
)













# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import pickle
# import numpy as np
# from datetime import datetime, timezone

# # Page configuration
# st.set_page_config(
#     page_title="Crypto Price Prediction",
#     page_icon="₿",
#     layout="wide"
# )

# # Load trained model
# @st.cache_resource
# def load_model():
#     with open("best_crypto_model.pkl", "rb") as file:
#         return pickle.load(file)

# bundle = load_model()


# # Automatically fetch previous-day cryptocurrency data

# @st.cache_data(ttl=900)
# def get_latest_crypto_data():

#     symbols = {
#         "BTC": "BTC-USD",
#         "ETH": "ETH-USD",
#         "USDT": "USDT-USD",
#         "BNB": "BNB-USD"
#     }

#     raw_data = yf.download(
#         list(symbols.values()),
#         period="1mo",
#         interval="1d",
#         group_by="ticker",
#         auto_adjust=False,
#         progress=False,
#         threads=False
#     )

#     if raw_data.empty:
#         raise ValueError(
#             "Yahoo Finance returned empty data. "
#             "Please try again later."
#         )

#     crypto_frames = []

#     for name, symbol in symbols.items():

#         # Handle different Yahoo Finance column formats
#         if isinstance(raw_data.columns, pd.MultiIndex):

#             if symbol in raw_data.columns.get_level_values(0):
#                 crypto_df = raw_data[symbol].copy()

#             elif symbol in raw_data.columns.get_level_values(1):
#                 crypto_df = raw_data.xs(
#                     symbol,
#                     axis=1,
#                     level=1
#                 ).copy()

#             else:
#                 raise ValueError(
#                     f"{symbol} is missing from Yahoo Finance response."
#                 )

#         else:
#             raise ValueError(
#                 "Unexpected Yahoo Finance data format."
#             )

#         if "Close" not in crypto_df.columns or "Volume" not in crypto_df.columns:
#             raise ValueError(f"Missing Close or Volume for {symbol}")

#         crypto_df = crypto_df[["Close", "Volume"]]

#         crypto_df.columns = [
#             f"Close ({name})",
#             f"Volume ({name})"
#         ]

#         crypto_frames.append(crypto_df)

#     combined_data = pd.concat(
#         crypto_frames,
#         axis=1,
#         join="inner"
#     ).dropna()

#     if combined_data.empty:
#         raise ValueError("No common dates found for all cryptocurrencies.")

#     latest_data = combined_data.iloc[[-1]].copy()

#     latest_date = latest_data.index[-1]

#     latest_data.columns = [
#         column + "_Previous_Day"
#         for column in latest_data.columns
#     ]

#     return latest_data, latest_date

# model = bundle["model"]
# selector = bundle["feature_selector"]
# scaler = bundle["scaler"]
# feature_columns = bundle["feature_columns"]

# # Application title
# st.title("₿ Cryptocurrency Price Prediction")
# st.subheader("Predict the next day's Bitcoin closing price")

# st.info(f"Selected ML Model: {bundle['model_name']}")

# # Sidebar
# st.sidebar.title("Navigation")

# page = st.sidebar.radio(
#     "Choose a page",
#     ["Home", "Live Market", "Bitcoin Prediction"]
# )

# # Home page
# if page == "Home":

#     st.header("Welcome!")

#     st.write("""
#     This application uses Machine Learning to predict
#     the next day's Bitcoin closing price.

#     Cryptocurrencies included in the training dataset:

#     - Bitcoin (BTC)
#     - Ethereum (ETH)
#     - Tether (USDT)
#     - Binance Coin (BNB)
#     """)

#     st.success("Machine Learning model loaded successfully!")

# # Live market page

# elif page == "Bitcoin Prediction":

#     st.header("Bitcoin Price Prediction")

#     st.write(
#         "Automatically fetch cryptocurrency market data "
#         "and predict the next-day BTC closing price."
#     )

#     if st.button("Fetch Market Data and Predict"):

#         try:

#             with st.spinner("Fetching cryptocurrency data..."):

#                 latest_data, latest_date = get_latest_crypto_data()

#                 # Match saved training feature order
#                 input_df = latest_data.reindex(
#                     columns=feature_columns
#                 )

#                 if input_df.isna().any().any():
#                     raise ValueError(
#                         "Market data does not match the model's features."
#                     )

#                 # Feature selection
#                 selected_input = selector.transform(input_df)

#                 # Scaling
#                 scaled_input = scaler.transform(selected_input)

#                 # Prediction
                
#                 st.subheader("Model Debugging")

#                 st.write("Model:", bundle["model_name"])
                
#                 st.subheader("Model Debugging")

#                 st.write("Model:", bundle["model_name"])
#                 st.write("Selected features:", bundle["selected_features"])
#                 st.write("Input data:")
#                 st.dataframe(input_df)

#                 st.write("Scaled input:")
#                 st.write(scaled_input)

#                 st.write("Prediction:", prediction)

              

#                 st.write("Selected features:", selected_features)

#                 st.write("Input data:")
#                 st.dataframe(input_df)

#                 st.write("Scaled input:")
#                 st.write(scaled_input)

#                 st.write("Prediction:", prediction)
#                 prediction = model.predict(scaled_input)[0]

#                 st.success("Prediction completed successfully!")

#                 st.metric(
#                 label="Predicted Next-Day BTC Closing Price",
#                 value=f"${prediction:,.2f}"
#             )

#                 st.write(
#                 f"Latest completed market data date: {latest_date}"
#             )

#             st.subheader("Fetched Market Data")

#             st.dataframe(input_df)

#             st.caption(
#                 "Experimental machine learning prediction. "
#                 "Not financial advice."
#             )

#         except Exception as e:

#             st.error(f"Prediction failed: {e}")


# st.sidebar.markdown("---")
# st.sidebar.write("Built with Python and Machine Learning")


# import os

# st.write("Working directory:", os.getcwd())
# st.write("Model path:", os.path.abspath("best_crypto_model.pkl"))

# bundle = load_model()

# st.write("Loaded keys:", list(bundle.keys()))

# if st.button("Clear Market Data Cache"):
#     st.cache_data.clear()
#     st.success("Cache cleared! Please try fetching again.")












# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import numpy as np
# import pickle
# import os

# from datetime import datetime, timezone

# # ============================================================
# # 1. PAGE CONFIGURATION
# # ============================================================

# st.set_page_config(
#     page_title="Crypto Price Prediction",
#     page_icon="₿",
#     layout="wide"
# )

# # ============================================================
# # 2. LOAD TRAINED MODEL
# # ============================================================

# MODEL_PATH = "best_crypto_model.pkl"

# @st.cache_resource
# def load_model():

#     if not os.path.exists(MODEL_PATH):
#         raise FileNotFoundError(
#             f"Model file not found: {MODEL_PATH}"
#         )

#     if os.path.getsize(MODEL_PATH) == 0:
#         raise ValueError("Model file is empty.")

#     with open(MODEL_PATH, "rb") as file:
#         bundle = pickle.load(file)

#     required_keys = [
#         "model",
#         "feature_selector",
#         "scaler",
#         "feature_columns",
#         "selected_features",
#         "model_name"
#     ]

#     for key in required_keys:
#         if key not in bundle:
#             raise KeyError(
#                 f"Missing key in model bundle: {key}"
#             )

#     return bundle

# try:
#     bundle = load_model()

#     model = bundle["model"]
#     selector = bundle["feature_selector"]
#     scaler = bundle["scaler"]
#     feature_columns = bundle["feature_columns"]

# except Exception as e:

#     st.error(f"Model loading failed: {e}")
#     st.stop()

# # ============================================================
# # 3. CRYPTOCURRENCY DATA FETCHING
# # ============================================================

# symbols = {
#     "BTC": "BTC-USD",
#     "ETH": "ETH-USD",
#     "USDT": "USDT-USD",
#     "BNB": "BNB-USD"
# }

# @st.cache_data(ttl=900)
# def get_crypto_data():

#     raw_data = yf.download(
#         tickers=list(symbols.values()),
#         period="1mo",
#         interval="1d",
#         group_by="ticker",
#         auto_adjust=False,
#         progress=False,
#         threads=False
#     )

#     if raw_data.empty:
#         raise ValueError(
#             "Yahoo Finance returned empty data. "
#             "Please try again later."
#         )

#     crypto_frames = []

#     for name, symbol in symbols.items():

#         if not isinstance(
#             raw_data.columns,
#             pd.MultiIndex
#         ):
#             raise ValueError(
#                 "Unexpected Yahoo Finance data format."
#             )

#         if symbol in raw_data.columns.get_level_values(0):

#             crypto_df = raw_data[symbol].copy()

#         elif symbol in raw_data.columns.get_level_values(1):

#             crypto_df = raw_data.xs(
#                 symbol,
#                 axis=1,
#                 level=1
#             ).copy()

#         else:

#             raise ValueError(
#                 f"Market data missing for {symbol}"
#             )

#         required_columns = ["Close", "Volume"]

#         for column in required_columns:

#             if column not in crypto_df.columns:
#                 raise ValueError(
#                     f"{column} missing for {symbol}"
#                 )

#         crypto_df = crypto_df[required_columns]

#         crypto_df.columns = [
#             f"Close ({name})",
#             f"Volume ({name})"
#         ]

#         crypto_frames.append(crypto_df)

#     combined_data = pd.concat(
#         crypto_frames,
#         axis=1,
#         join="inner"
#     )

#     combined_data.replace(
#         [np.inf, -np.inf],
#         np.nan,
#         inplace=True
#     )

#     combined_data.dropna(inplace=True)

#     # Exclude today's potentially incomplete daily candle
#     today_utc = datetime.now(timezone.utc).date()

#     combined_data = combined_data[
#         combined_data.index.date < today_utc
#     ]

#     if combined_data.empty:
#         raise ValueError(
#             "No common completed daily market data found."
#         )

#     latest_date = combined_data.index[-1]

#     return combined_data, latest_date

# # ============================================================
# # 4. PREPARE MODEL INPUT
# # ============================================================

# def prepare_input(combined_data):

#     latest_data = combined_data.iloc[[-1]].copy()

#     latest_data.columns = [
#         column + "_Previous_Day"
#         for column in latest_data.columns
#     ]

#     # Match exact training feature order
#     input_df = latest_data.reindex(
#         columns=feature_columns
#     )

#     if input_df.isna().any().any():
#         missing = input_df.columns[
#             input_df.isna().any()
#         ].tolist()

#         raise ValueError(
#             f"Missing model features: {missing}"
#         )

#     return input_df

# # ============================================================
# # 5. PREDICTION FUNCTION
# # ============================================================

# def predict_btc_price(input_df):

#     # Feature selection
#     selected_input = selector.transform(input_df)

#     # Scaling
#     scaled_input = scaler.transform(selected_input)
    
    
#     st.subheader("Prediction Diagnostics")

#    st.write("Scaled input values:")
# st.write(scaled_input)

# st.write("Training feature minimum:")
# st.write(scaler.data_min_)

# st.write("Training feature maximum:")
# st.write(scaler.data_max_)

# st.write(
#     "Features outside training range:",
#     ((selected_input < scaler.data_min_) |
#      (selected_input > scaler.data_max_)).tolist()
# )

#     # Model prediction
#     prediction = model.predict(scaled_input)[0]

#     if not np.isfinite(prediction):
#         raise ValueError(
#             "Model generated an invalid prediction."
#         )

#     return prediction, scaled_input

# # ============================================================
# # 6. APPLICATION HEADER
# # ============================================================

# st.title("₿ Cryptocurrency Price Prediction")

# st.subheader(
#     "Machine Learning Based Bitcoin Price Forecasting"
# )

# st.info(
#     f"Selected ML Model: {bundle['model_name']}"
# )

# # ============================================================
# # 7. SIDEBAR NAVIGATION
# # ============================================================

# st.sidebar.title("Navigation")

# page = st.sidebar.radio(
#     "Choose a page",
#     [
#         "Home",
#         "Live Market",
#         "Bitcoin Prediction"
#     ]
# )

# st.sidebar.markdown("---")

# st.sidebar.write(
#     "Built with Python and Machine Learning"
# )

# # ============================================================
# # 8. HOME PAGE
# # ============================================================

# if page == "Home":

#     st.header("Welcome!")

#     st.write(
#         """
#         This project uses Machine Learning to predict
#         the next-day Bitcoin closing price.

#         Cryptocurrencies included:

#         - Bitcoin (BTC)
#         - Ethereum (ETH)
#         - Tether (USDT)
#         - Binance Coin (BNB)
#         """
#     )

#     st.success("ML model loaded successfully!")

#     st.subheader("Project Technologies")

#     st.write(
#         """
#         - Python
#         - Pandas and NumPy
#         - Scikit-learn
#         - Random Forest and Regression Models
#         - Yahoo Finance
#         - Streamlit
#         """
#     )

#     st.subheader("Selected Model")

#     st.write(bundle["model_name"])

# # ============================================================
# # 9. LIVE MARKET PAGE
# # ============================================================

# elif page == "Live Market":

#     st.header("Live Cryptocurrency Market")

#     st.caption(
#         "Displays the latest available completed daily data, "
#         "not necessarily real-time prices."
#     )

#     if st.button("Refresh Market Data"):

#         st.cache_data.clear()

#     try:

#         with st.spinner("Fetching market data..."):

#             market_data, market_date = get_crypto_data()

#             latest = market_data.iloc[-1]

#         col1, col2 = st.columns(2)

#         with col1:

#             st.metric(
#                 "Bitcoin (BTC)",
#                 f"${latest['Close (BTC)']:,.2f}"
#             )

#             st.metric(
#                 "Ethereum (ETH)",
#                 f"${latest['Close (ETH)']:,.2f}"
#             )

#         with col2:

#             st.metric(
#                 "Tether (USDT)",
#                 f"${latest['Close (USDT)']:,.4f}"
#             )

#             st.metric(
#                 "Binance Coin (BNB)",
#                 f"${latest['Close (BNB)']:,.2f}"
#             )

#         st.write("Latest completed market date:", market_date)

#         st.subheader("Historical Closing Prices")

#         price_columns = [
#             "Close (BTC)",
#             "Close (ETH)",
#             "Close (USDT)",
#             "Close (BNB)"
#         ]

#         st.line_chart(
#             market_data[price_columns]
#         )

#         st.subheader("Market Data")

#         st.dataframe(market_data)

#     except Exception as e:

#         st.error(f"Market data error: {e}")

# # ============================================================
# # 10. BITCOIN PREDICTION PAGE
# # ============================================================

# elif page == "Bitcoin Prediction":

#     st.header("Bitcoin Price Prediction")

#     st.write(
#         """
#         Automatically fetch cryptocurrency market data
#         and predict the next-day Bitcoin closing price.
#         """
#     )

#     if st.button(
#         "Fetch Market Data and Predict",
#         type="primary"
#     ):

#         try:

#             with st.spinner(
#                 "Fetching cryptocurrency data..."
#             ):

#                 market_data, latest_date = get_crypto_data()

#             with st.spinner(
#                 "Preparing prediction features..."
#             ):

#                 input_df = prepare_input(market_data)

#             with st.spinner(
#                 "Running Machine Learning model..."
#             ):

#                 prediction, scaled_input = predict_btc_price(
#                     input_df
#                 )

#             # Prediction result
#             st.success(
#                 "Prediction completed successfully!"
#             )

#             st.metric(
#                 label="Predicted Next-Day BTC Closing Price",
#                 value=f"${prediction:,.2f}"
#             )

#             st.write(
#                 "Latest completed market data date:",
#                 latest_date
#             )

#             # Input data
#             st.subheader("Fetched Market Data")

#             st.dataframe(input_df)

#             # Debugging
#             with st.expander("Model Diagnostics"):

#                 st.write(
#                     "Selected Model:",
#                     bundle["model_name"]
#                 )

#                 st.write(
#                     "Selected Features:",
#                     bundle["selected_features"]
#                 )

#                 st.write("Scaled Input:")

#                 st.write(scaled_input)

#                 st.write(
#                     "Prediction:",
#                     prediction
#                 )

#             st.caption(
#                 "Experimental machine learning prediction. "
#                 "Not financial advice."
#             )

#         except Exception as e:

#             st.error(
#                 f"Prediction failed: {e}"
#             )

# # ============================================================
# # 11. FOOTER
# # ============================================================

# st.sidebar.markdown("---")

# st.sidebar.caption(
#     "Crypto Price Prediction | Python + ML + Streamlit"
# )