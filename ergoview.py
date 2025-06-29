import streamlit as st
import plotly.express as px
import pandas as pd

df = pd.DataFrame({
    "x": [1, 2, 3, 4],
    "y": [10, 20, 30, 40]
})

fig = px.line(df, x="x", y="y", title="Gráfico de Teste")
st.plotly_chart(fig)
