import streamlit as st
import joblib
import pandas as pd


st.set_page_config(page_title="Predictor de Mora - ConecTel", layout="centered")
st.title("📡 Sistema de Alerta Temprana de Morosidad")
st.write("Ingresa los datos financieros y de servicio del cliente para evaluar su riesgo de caer en mora (90 días).")


modelo = joblib.load('model.pkl')


st.subheader("Datos del Cliente")
gasto_relativo = st.number_input("Indicador de Gasto Relativo", min_value=0.0, value=0.5, step=0.1)
satisfaccion = st.number_input("Indicador de Satisfacción (1 al 10)", min_value=1.0, max_value=10.0, value=5.0, step=0.5)
factura = st.number_input("Factura Mensual (CLP)", min_value=0, value=15000, step=1000)

if st.button("Evaluar Riesgo"):
    # Crear la tabla invisible para el modelo
    datos_cliente = pd.DataFrame({
        'indicador_gasto_relativo': [gasto_relativo],
        'indicador_satisfaccion': [satisfaccion],
        'factura_mensual_clp': [factura]
    })

    
    prediccion = modelo.predict(datos_cliente)
    probabilidad = modelo.predict_proba(datos_cliente)[0][1]

    
    st.markdown("---")
    if prediccion[0] == 1:
        st.error(f"⚠️ ALERTA ROJA: Cliente con ALTO RIESGO de Mora.")
        st.write(f"Probabilidad de impago: **{probabilidad * 100:.1f}%**")
    else:
        st.success(f"✅ CLIENTE SEGURO: Riesgo bajo de morosidad.")
        st.write(f"Probabilidad de impago: **{probabilidad * 100:.1f}%**")
