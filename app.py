import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm
import altair as alt

st.set_page_config(
    page_title = "VAR Calculator",
    page_icon = "<UNK>",
    layout = "wide",
    initial_sidebar_state = "expanded",
)

alt.themes.enable('dark')

class VaR:
    def __init__(self, ticker, start_date, end_date, rolling_window, confidence_level, portfolio_value):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.rolling_window = rolling_window
        self.confidence_level = confidence_level
        self.portfolio_value = portfolio_value

        self.data()

    def data(self):

        df = yf.download(self.ticker, self.start_date, self.end_date, auto_adjust=False)
        if df.empty:
            raise Exception(f"No data found for {self.ticker}")

        self.adj_close_df = df[['Adj Close']]
        self.log_returns_df = np.log(self.adj_close_df / self.adj_close_df.shift(1))
        self.log_returns_df.dropna(inplace=True)
        self.equal_weights = np.array([1 / len(self.ticker)] * len(self.ticker))
        self.historical_returns = (self.log_returns_df * self.equal_weights).sum(axis=1)
        self.rolling_returns = self.historical_returns.rolling(window = self.rolling_window).sum()
        self.rolling_returns.dropna(inplace=True)
        self.historical_method()
        self.parametric_method()

    def historical_method(self):
        self.significance_level =  1 - self.confidence_level
        self.historical_VaR = -np.percentile(self.rolling_returns, self.significance_level) * self.portfolio_value

    def parametric_method(self):
        self.cov_matrix = self.log_returns_df.cov() * 252
        self.portfolio_std = np.sqrt(np.dot(self.equal_weights.T, np.dot(self.cov_matrix, self.equal_weights)))
        self.parametric_VaR = self.portfolio_std * norm.ppf(self.confidence_level) * np.sqrt(self.rolling_window / 252) * self.portfolio_value

    def plot_var_results(self, title, var_value, returns_in_dollar, confidence_level):

        plt.figure(figsize=(12, 10))
        plt.hist(returns_in_dollar, bins=100, density=True)
        plt.xlabel(f'\n {title} VaR = ${var_value:.2f}')
        plt.ylabel('Frequency')
        plt.title(f"Distribution of Portfolios {self.rolling_window}-Day Returns ({title} VaR)")
        plt.axvline(-var_value, color='r', linestyle='dashed', linewidth=2, label=f'VaR at {confidence_level:.0%} confidence level')
        plt.legend()
        plt.tight_layout()
        return plt

if 'recent_outputs' not in st.session_state:
    st.session_state.recent_outputs = []

with st.sidebar:
    st.title('VAR Calculator')
    st.write('Created by')
    author = 'Bakytzhan'
    form = f'<a href="{author}" target="_blank" style="text-decoration: none; color: inherit;">`Bakytzhan`</a>'

    st.markdown(form, unsafe_allow_html=True)

    tickers = st.text_input('Enter tickers separated by space').split(' ')
    start_date = st.date_input('Enter start date', value=pd.to_datetime('2024-01-01'))
    end_date = st.date_input('Enter end date', value=pd.to_datetime('today'))
    rolling_window = st.slider('Enter rolling window size', min_value=1, max_value=252, step=1)
    confidence_level = st.slider('Enter confidence level', min_value=0.80, max_value=0.99, value=0.95, step=0.01)
    portfolio_value = st.number_input('Enter portfolio value', step=100000, value=1000000)
    calculate_button = st.button('Calculate VaR')

def calculate_var(tickers, start_date, end_date, rolling_window, confidence_level, portfolio_value):

    var_instance = VaR(tickers, start_date, end_date, rolling_window, confidence_level, portfolio_value)
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.info('Historical VaR')
        historical_chart = var_instance.plot_var_results("Historical", var_instance.historical_VaR, var_instance.rolling_returns * var_instance.portfolio_value, confidence_level)
        st.pyplot(historical_chart)

    with chart_col2:
        st.info('Parametric VaR')
        parametric_chart = var_instance.plot_var_results("Parametric", var_instance.parametric_VaR, var_instance.rolling_returns * var_instance.portfolio_value, confidence_level)
        st.pyplot(parametric_chart)

    col1, col2 = st.columns([0.4, 0.6])
    with col1:
        st.info('Input summary')
        st.write(f'Tickers: {",".join(tickers)}')
        st.write(f'Start Date: {start_date}')
        st.write(f'End Date: {end_date}')
        st.write(f'Rolling Window: {rolling_window}')
        st.write(f'Confidence Level: {confidence_level}')
        st.write(f'Portfolio Value: {portfolio_value}')

    with col2:
        st.info('VaR output')
        data = {'Method': ['Historical VaR', 'Parametric VaR'], 'VaR value': [f'${var_instance.historical_VaR:,.2f}', f'${var_instance.parametric_VaR:,.2f}']}
        df = pd.DataFrame(data)
        st.table(df)

    new_row = {'Tickers': ','.join(var_instance.ticker), 'Confidence Level': var_instance.confidence_level,
               'Rolling days': int(var_instance.rolling_window), 'Portfolio value': var_instance.portfolio_value,
               'Historical VaR': var_instance.historical_VaR, 'Parametric VaR': var_instance.parametric_VaR,
               'Start date': var_instance.start_date.strftime('%Y-%m-%d'), 'End date': var_instance.end_date.strftime('%Y-%m-%d')}

    st.session_state['recent_outputs'].append(new_row)

    st.info('Previous VaR Output')
    recent_df = pd.DataFrame(st.session_state['recent_outputs'])
    st.table(recent_df)

    @st.cache_data
    def convert_for_download(df):
        return df.to_csv(index_label=False).encode("utf-8")

    csv_to_download = convert_for_download(recent_df)

    st.download_button(
        label="Download CSV",
        data=csv_to_download,
        file_name="var_data.csv",
        mime="text/csv",
        icon=":material/download:",
    )

if 'first_run' not in st.session_state or st.session_state['first_run']:
    st.session_state.first_run = False
    default_tickers = 'NVDA MSFT'.split(' ')
    default_start_date = pd.to_datetime('2020-01-01')
    default_end_date = pd.to_datetime('today')
    default_rolling_window = 20
    default_confidence_level = 0.95
    default_portfolio_value = 100000

    # Perform the default calculation
    calculate_var(default_tickers, default_start_date, default_end_date, default_rolling_window, default_confidence_level, default_portfolio_value)

if calculate_button:
    calculate_var(tickers, start_date, end_date, rolling_window, confidence_level, portfolio_value)
