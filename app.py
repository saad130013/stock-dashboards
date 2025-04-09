
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
from datetime import datetime
from prophet import Prophet
import numpy as np
import os
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

model_ai = ResNet50(weights='imagenet')

@st.cache_data
def load_all_symbols():
    return {
        "أرامكو": "2222.SR",
        "سابك": "2010.SR",
        "الاتصالات السعودية": "7010.SR",
        "مصرف الراجحي": "1120.SR",
        "البنك الأهلي": "1180.SR"
    }

symbols = load_all_symbols()

def analyze_stock(name, symbol):
    data = yf.download(symbol, period="1y", interval="30m")
    data.dropna(inplace=True)
    data["RSI"] = ta.rsi(data["Close"], length=14)
    st.subheader(f"📌 تحليل سهم {name}")
    st.write(f"**📅 التاريخ:** {datetime.today().strftime('%Y-%m-%d %H:%M')}")
    st.write(f"**📉 السعر الحالي:** {data['Close'].iloc[-1]:.2f} ريال")

    st.plotly_chart(go.Figure(data=[
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='شموع'
        )
    ]))

    st.subheader("⛔️ الأخطار والخلل")
    st.info("هذا مجرد مثال مبدأي بدون تحليلات متقدمة.")

st.set_page_config(page_title="خبير الأسهم الفني", layout="wide")
st.title("📊 نظام خبير الأسهم الفني السعودي")
st.markdown("نظام تحليل الأسهم السعودية باستخدام مؤشرات فنية وذكاء اصطناعي")

option = st.radio("اختر وضع التشغيل", ("تحليل سهم محدد",))

if option == "تحليل سهم محدد":
    selected_stock = st.selectbox("اختر السهم", list(symbols.keys()))
    analyze_stock(selected_stock, symbols[selected_stock])
