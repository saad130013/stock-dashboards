import yfinance as yf
import pandas as pd
import streamlit as st
import pandas_ta as ta
import numpy as np
import plotly.graph_objs as go
from datetime import datetime, timedelta
from prophet import Prophet

from transformers import pipeline

# استخدم موديل جاهز مدمج لا يحتاج تحميل من الإنترنت
sentiment_analyzer = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    try:
        result = sentiment_analyzer(text)
        return result[0]['label'], result[0]['score']
    except:
        return "NEUTRAL", 0.5
        result = sentiment_analyzer(text)
        return result[0]['label'], result[0]['score']
    except:
        return "NEUTRAL", 0.5

# ---------------------- الواجهة الرئيسية ----------------------
def main():
    st.title("📈 منصة الذكاء المالي - السوق السعودي")
    
    with st.sidebar:
        st.header("⚙️ الإعدادات")
        auto_update = st.checkbox("التحديث التلقائي كل 5 دقائق", True)
        risk_level = st.select_slider("مستوى المخاطرة", ['منخفض', 'متوسط', 'عالي'])
        st.subheader("إدارة المحفظة")
        portfolio_file = st.file_uploader("رفع ملف المحفظة (CSV)")
        st.subheader("إضافة سهم مخصص")
        custom_symbol = st.text_input("رمز السهم (مثال: 2222.SR)")
        if st.button("إضافة السهم"):
            if custom_symbol.endswith(".SR"):
                st.success("تمت الإضافة بنجاح")
            else:
                st.error("الرمز غير صحيح")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 لوحة التحكم", "📰 الأخبار", "🧠 الذكاء الاصطناعي", "⚖️ التحليل الأساسي"])

    with tab1:
        st.subheader("📈 أداء السوق اللحظي")
        index_data = get_stock_data("^TASI.SR", period="1d", interval="5m")
        if index_data is not None:
            col1, col2, col3 = st.columns(3)
            current_price = index_data['Close'].iloc[-1]
            prev_close = index_data['Close'].iloc[0]
            change = ((current_price - prev_close)/prev_close)*100
            with col1:
                st.markdown(f"<div class='metric-box'><h3>السعر الحالي</h3><h1>{current_price:.2f}</h1></div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div class='metric-box'><h3>التغيير اليومي</h3><h1>{change:.2f}%</h1></div>", unsafe_allow_html=True)
            with col3:
                st.markdown(f"<div class='metric-box'><h3>الحجم المتداول</h3><h1>{index_data['Volume'].iloc[-1]:,}</h1></div>", unsafe_allow_html=True)
            fig = go.Figure(data=[go.Candlestick(x=index_data.index, open=index_data['Open'], high=index_data['High'], low=index_data['Low'], close=index_data['Close'])])
            fig.update_layout(title="مؤشر تاسي اليومي", height=400)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("📰 آخر الأخبار المالية")
        news_sources = {
            "أرقام": "https://www.argaam.com",
            "بلومبرغ": "https://www.bloomberg.com/markets/stocks"
        }
        for source, url in news_sources.items():
            with st.expander(f"أخبار {source}"):
                try:
                    response = requests.get(url)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    articles = soup.find_all('article', limit=3)
                    for article in articles:
                        title = article.find('h3').text
                        summary = article.find('p').text
                        sentiment, score = analyze_sentiment(title)
                        st.markdown(f"""
                        <div class="news-card">
                            <h4>{title}</h4>
                            <p>{summary}</p>
                            <div style="color: {'#00FF00' if sentiment == 'POSITIVE' else '#FF0000'}">
                                المشاعر: {sentiment} ({score:.2f})
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"خطأ في جلب الأخبار من {source}: {str(e)}")

if __name__ == "__main__":
    main()
