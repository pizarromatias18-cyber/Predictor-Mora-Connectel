import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Predictor de Mora - ConecTel", layout="centered")
st.title("Sistema de Alerta Temprana")

try:
    modelo = joblib.load('model.pkl')
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

st.subheader("Ingresar Datos del Cliente")
factura = st.number_input("Factura Mensual (CLP)", min_value=0, value=25000, step=1000)
antiguedad = st.number_input("Antigüedad (Meses)", min_value=0, value=12, step=1)
satisfaccion = st.number_input("Satisfacción (1-10)", min_value=1.0, max_value=10.0, value=5.0, step=0.5)

if st.button("Evaluar Riesgo"):
    
    valor_hist = factura * antiguedad
    
    input_data = pd.DataFrame({
        'factura_mensual_clp': [factura],
        'antiguedad_meses': [antiguedad],
        'indicador_satisfaccion': [satisfaccion],
        'valor_historico_cliente': [valor_hist]
    })
    
    prob = modelo.predict_proba(input_data)[0][1]
    
    st.markdown("---")
    
    if prob > 0.75:
        st.error(f"⚠️ ALERTA: Riesgo alto de morosidad detectado.")
        st.write(f"Probabilidad de riesgo: **{prob * 100:.1f}%**")
    else:
        st.success(f"✅ CLIENTE SEGURO.")
        st.write(f"Probabilidad de riesgo: **{prob * 100:.1f}%**")
