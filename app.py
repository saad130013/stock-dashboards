import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
from datetime import datetime, timedelta
import numpy as np
from prophet import Prophet
import time
import os
import plotly.graph_objs as go
from plyer import notification
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import cv2
from stocknews import StockNews
import requests
from bs4 import BeautifulSoup

model_ai = ResNet50(weights='imagenet')
symbols = {
    "أرامكو": "2222.SR",
    "سابك": "2010.SR",
    "الاتصالات السعودية": "7010.SR",
    "مصرف الراجحي": "1120.SR",
    "البنك الأهلي": "1180.SR",
    "كيان": "2350.SR",
    "اللجين": "4003.SR"
}
recommendation_log_file = "recommendation_log.csv"

def add_custom_stock():
    with st.expander("إضافة سهم مخصص"):
        custom_name = st.text_input("اسم السهم المخصص")
        custom_symbol = st.text_input("الرمز (مثال: 2222.SR)")
        if st.button("إضافة"):
            symbols[custom_name] = custom_symbol
            st.success("تمت الإضافة!")

def get_market_sentiment():
    url = "https://www.saudiexchange.sa/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    sentiment = "إيجابي" if "ارتفاع" in soup.text else "سلبي"
    return f"📊 مشاعر السوق: {sentiment}"

def backtest_strategy(data):
    signals = []
    for i in range(1, len(data)):
        if data['RSI'][i] < 30 and data['MACD'][i] > data['MACD_signal'][i]:
            signals.append('شراء')
        elif data['RSI'][i] > 70 and data['MACD'][i] < data['MACD_signal'][i]:
            signals.append('بيع')
        else:
            signals.append('حياد')
    signals.insert(0, 'حياد')
    return signals

def enhanced_chart_patterns(data):
    patterns = []
    closes = data["Close"].tail(100)
    max_idx = np.argmax(closes)
    if max_idx > 30 and max_idx < 70:
        left_shoulder = np.mean(closes[max_idx-30:max_idx-10])
        right_shoulder = np.mean(closes[max_idx+10:max_idx+30])
        if left_shoulder < closes[max_idx] and right_shoulder < closes[max_idx]:
            patterns.append("🔺 نموذج رأس وكتفين")
    return patterns

def real_time_update(interval=60):
    while True:
        time.sleep(interval)
        st.experimental_rerun()

def show_portfolio_summary():
    if os.path.exists(recommendation_log_file):
        log_df = pd.read_csv(recommendation_log_file)
        portfolio = log_df.groupby('السهم')['التوصية'].last().reset_index()
        st.subheader("ملخص المحفظة")
        st.dataframe(portfolio.style.highlight_max(axis=0))

def enhanced_analysis(name, symbol):
    data = yf.download(symbol, period="1y", interval="1h")
    data.dropna(inplace=True)

    data["EMA_20"] = ta.ema(data["Close"], length=20)
    data["VWAP"] = ta.vwap(data["High"], data["Low"], data["Close"], data["Volume"])
    data["ATR"] = ta.atr(data["High"], data["Low"], data["Close"], length=14)
    data["SMA_50"] = ta.sma(data["Close"], length=50)
    data["SMA_200"] = ta.sma(data["Close"], length=200)
    data["RSI"] = ta.rsi(data["Close"], length=14)
    macd = ta.macd(data["Close"])
    data["MACD"] = macd["MACD_12_26_9"]
    data["MACD_signal"] = macd["MACDs_12_26_9"]
    data['Signal'] = backtest_strategy(data)

    stock = yf.Ticker(symbol)
    info = stock.info
    pe_ratio = info.get('trailingPE', 'N/A')
    dividend_yield = info.get('dividendYield', 'N/A')

    st.subheader(f"📈 تحليل متقدم لـ {name}")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("السعر الحالي", f"{data['Close'].iloc[-1]:.2f} ريال")
        st.metric("المعدل السعري (P/E)", pe_ratio)

    with col2:
        st.metric("العائد على التوزيعات", f"{dividend_yield*100 if dividend_yield != 'N/A' else 'N/A'}%")
        st.metric("الحجم المتداول", f"{data['Volume'].iloc[-1]:,}")

    with col3:
        st.metric("التقلب (ATR)", f"{data['ATR'].iloc[-1]:.2f}")
        st.metric("القيمة السوقية", f"{info.get('marketCap', 'N/A')/1e9:.2f} مليار")

    tab1, tab2, tab3 = st.tabs(["السعر", "المؤشرات", "الإشارات"])

    with tab1:
        fig = go.Figure(data=[
            go.Candlestick(x=data.index,
                           open=data['Open'],
                           high=data['High'],
                           low=data['Low'],
                           close=data['Close']),
            go.Scatter(x=data.index, y=data['SMA_50'], name='SMA 50'),
            go.Scatter(x=data.index, y=data['SMA_200'], name='SMA 200')
        ])
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI'))
        fig.add_hline(y=30, line_dash="dash", line_color="green")
        fig.add_hline(y=70, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

    
with tab3:
        st.write("إشارات التداول التاريخية")
        signals = data[['Close', 'Signal']].tail(30)
        st.line_chart(signals['Close'])
        st.dataframe(signals)

        # زر تحميل التوصيات PDF
        if st.button("📥 تحميل توصيات PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt=f"توصيات التداول - {name}", ln=True, align="C")
            pdf.ln(10)
            for i in range(len(signals)):
                row = signals.iloc[i]
                pdf.cell(200, 10, txt=f"{signals.index[i]} - السعر: {row['Close']:.2f} - التوصية: {row['Signal']}", ln=True)
            pdf_path = f"{name}_توصيات.pdf"
            pdf.output(pdf_path)
            with open(pdf_path, "rb") as f:
                st.download_button(label="⬇️ تحميل الملف", data=f, file_name=pdf_path, mime="application/pdf")
    
        st.write("إشارات التداول التاريخية")
        signals = data[['Close', 'Signal']].tail(30)
        st.line_chart(signals['Close'])
        st.dataframe(signals)

    st.subheader("📰 الأخبار والتحليلات")
    try:
        sn = StockNews(symbol.split('.')[0], country='Saudi Arabia')
        df_news = sn.read_news()
        for item in df_news[:5]:
            st.markdown(f"""
            <div class="news-card">
            <h4>{item['title']}</h4>
            <p>{item['summary']}</p>
            <a href="{item['link']}" target="_blank">اقرأ المزيد</a>
            </div>
            """, unsafe_allow_html=True)
    except:
        st.warning("تعذر جلب الأخبار حالياً.")

st.set_page_config(page_title="منصة تداول متكاملة", layout="wide", page_icon="💹")
st.markdown("""
<style>
.news-card {
padding: 15px;
border-radius: 10px;
box-shadow: 0 2px 5px rgba(0,0,0,0.1);
margin: 10px 0;
background: white;
}
</style>
""", unsafe_allow_html=True)

st.title("📈 منصة التداول الذكية - السوق السعودي")
st.markdown(get_market_sentiment())

with st.sidebar:
    st.header("الإعدادات")
    auto_update = st.checkbox("التحديث التلقائي كل دقيقة")
    risk_level = st.select_slider("مستوى المخاطرة", options=['منخفض', 'متوسط', 'عالي'])

add_custom_stock()
show_portfolio_summary()

tab_main, tab_news = st.tabs(["التحليل الفني", "الأخبار المالية"])

with tab_main:
    selected_stock = st.selectbox("اختر السهم", list(symbols.keys()))
    enhanced_analysis(selected_stock, symbols[selected_stock])

with tab_news:
    st.subheader("آخر الأخبار المالية")

if 'auto_update' in locals() and auto_update:
    real_time_update(60)