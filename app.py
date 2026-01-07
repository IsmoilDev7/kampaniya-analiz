import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet

st.set_page_config(page_title="Savdo va Kampaniya Analizi", layout="wide")

st.title("📊 Savdo, Zarar va Prognoz Analizi")
st.markdown("""
Bu dashboard savdo ma'lumotlari asosida:
- savdo holatini,
- foyda va zarar zonalarini,
- narx o‘zgarishining ta’sirini,
- kelajak prognozini
ko‘rsatadi.
""")

# Fayl yuklash
file = st.file_uploader("📂 Excel faylni yuklang", type=["xlsx"])
if file:
    df = pd.read_excel(file)
    df['Период'] = pd.to_datetime(df['Период'], errors='coerce')
    df['Summa'] = pd.to_numeric(df['Summa'], errors='coerce')
    df['Miqdor'] = pd.to_numeric(df['Miqdor'], errors='coerce')
    df = df.dropna(subset=['Период','Summa','Miqdor'])
    df['Narx'] = df['Summa'] / df['Miqdor']

    # Filtrlar
    st.subheader("🔍 Filtrlar")
    c1, c2 = st.columns(2)
    with c1:
        tovar = st.multiselect("Tovarni tanlang", df['Tovar'].unique(), default=df['Tovar'].unique())
    with c2:
        ombor = st.multiselect("Omborni tanlang", df['Ombor'].unique(), default=df['Ombor'].unique())
    df = df[df['Tovar'].isin(tovar) & df['Ombor'].isin(ombor)]

    # KPI
    st.subheader("📌 Asosiy ko‘rsatkichlar")
    k1, k2, k3 = st.columns(3)
    k1.metric("💰 Umumiy savdo", f"{df['Summa'].sum():,.0f}")
    k2.metric("📦 Sotilgan miqdor", f"{df['Miqdor'].sum():,.0f}")
    k3.metric("💲 O‘rtacha narx", f"{df['Narx'].mean():,.2f}")

    # Savdo trendlari
    st.subheader("📈 Savdo trendlari")
    trend = df.groupby('Период').agg({'Summa':'sum'}).reset_index()
    st.plotly_chart(px.line(trend, x='Период', y='Summa', title="Vaqt bo‘yicha savdo o‘zgarishi"), use_container_width=True)

    # What-if narx simulyatsiyasi
    st.subheader("🎚️ Narx o‘zgarishining ta’siri (What-if)")
    change = st.slider("Narxni o‘zgartirish (%)", -30, 30, -10)
    trend['Simulyatsiya'] = trend['Summa'] * (1 + change / 100)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend['Период'], y=trend['Summa'], name="Asl savdo"))
    fig.add_trace(go.Scatter(x=trend['Период'], y=trend['Simulyatsiya'], name="Simulyatsiya"))
    fig.update_layout(title="Narx o‘zgarishi savdoga ta’siri")
    st.plotly_chart(fig, use_container_width=True)

    # Foyda va zarar
    st.subheader("💸 Foyda va zarar analizi")
    df['Xarajat'] = df['Narx'] * 0.7 * df['Miqdor']
    df['Foyda'] = df['Summa'] - df['Xarajat']
    profit = df.groupby('Tovar')['Foyda'].sum().reset_index()
    st.plotly_chart(px.bar(profit, x='Tovar', y='Foyda', title="Tovarlar bo‘yicha foyda va zarar"), use_container_width=True)

    # Prognoz
    st.subheader("🔮 Savdo prognozi (6 oy)")
    ts = trend.rename(columns={'Период':'ds','Summa':'y'})
    model = Prophet()
    model.fit(ts)
    future = model.make_future_dataframe(periods=6, freq='M')
    forecast = model.predict(future)
    st.plotly_chart(px.line(forecast, x='ds', y='yhat', title="Kelgusi 6 oylik savdo prognozi"), use_container_width=True)

else:
    st.info("Excel faylni yuklang va analizni boshlang.")
