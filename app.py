import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime
from prophet import Prophet
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go

# تحسين ذاكرة التخزين المؤقت
@st.cache_data(ttl=3600)  # تحديث البيانات كل ساعة
def load_all_symbols():
    return {
        "أرامكو": "2222.SR",
        "سابك": "2010.SR",
        "الاتصالات السعودية": "7010.SR",
        "مصرف الراجحي": "1120.SR",
        "البنك الأهلي": "1180.SR"
    }

symbols = load_all_symbols()

# تحسين دقة الحسابات الفنية
def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    close = data['Close']
    exp1 = close.ewm(span=fast, adjust=False).mean()
    exp2 = close.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def analyze_stock(name, symbol):
    try:
        # تحسين جلب البيانات مع معالجة الأخطاء
        data = yf.download(symbol, period="1y", interval="60m")  # تغيير إلى 60 دقيقة لدعم المزيد من الأسهم
        if data.empty:
            st.error("فشل في جلب البيانات. الرجاء التحقق من اتصال الإنترنت أو رمز السهم.")
            return
            
        data.dropna(inplace=True)
        
        # إضافة فحص البيانات قبل الحسابات
        if len(data) < 33:
            st.error("لا توجد بيانات كافية للتحليل")
            return
            
        data["RSI"] = calculate_rsi(data)
        data["MACD"], data["MACD_signal"] = calculate_macd(data)

        # واجهة مستخدم محسنة
        st.subheader(f"📊 تحليل مفصل لسهم {name}")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("آخر سعر", f"{data['Close'].iloc[-1]:.2f} ريال")
            st.write(f"**تاريخ التحليل:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            
        with col2:
            st.metric("RSI الحالي", f"{data['RSI'].iloc[-1]:.2f}")
            st.metric("الفرق MACD", f"{(data['MACD'].iloc[-1] - data['MACD_signal'].iloc[-1]):.2f}")

        # رسم بياني تفاعلي مع تحديثات
        fig = go.Figure(data=[
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='الأسعار'
            )
        ])
        fig.update_layout(
            title=f"حركة السهم خلال السنة الأخيرة - {name}",
            xaxis_title="التاريخ",
            yaxis_title="السعر (ريال سعودي)",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"حدث خطأ غير متوقع: {str(e)}")

# تحسين إعداد الصفحة
st.set_page_config(
    page_title="المحلل الفني للأسهم السعودية",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🪙 النظام الذكي لتحليل الأسهم السعودية")
st.markdown("""
    <style>
    .stRadio [role=radiogroup]{
        align-items: center;
        justify-content: center;
    }
    </style>
""", unsafe_allow_html=True)

# واجهة مستخدم أكثر تفاعلية
with st.sidebar:
    st.header("الإعدادات")
    update_freq = st.selectbox("معدل التحديث", ['60 دقيقة', '30 دقيقة', 'يومي'])
    
st.radio("وضع التحليل", ["تحليل سهم محدد"], horizontal=True)

selected_stock = st.selectbox(
    "اختر السهم للتحليل",
    list(symbols.keys()),
    index=0,
    help="اختر سهمًا من القائمة لرؤية التحليل التفصيلي"
)

if selected_stock:
    analyze_stock(selected_stock, symbols[selected_stock])
else:
    st.info("الرجاء اختيار سهم من القائمة لبدء التحليل")

# إضافة تنبيهات الأمان
st.sidebar.markdown("---")
st.sidebar.warning("""
**تنبيه مهم:**  
هذا التحليل لأغراض تعليمية فقط ولا يعتبر نصيحة استثمارية.  
الرجاء استشارة مستشار مالي قبل اتخاذ أي قرار استثماري.
""")
