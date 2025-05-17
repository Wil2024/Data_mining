import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# Configuración inicial
st.set_page_config(page_title="Minería de Datos BotiCura", layout="wide")
st.title("🧠 Minería de Datos: Clustering, Asociación y Sentimiento")
st.markdown("""
Este simulador permite explorar técnicas avanzadas de minería de datos aplicadas al sector salud:
- **Clustering de clientes** por frecuencia y gasto
- **Reglas de asociación** entre productos
- **Análisis de sentimiento** en reseñas

Usa estos insights para tomar decisiones estratégicas en BotiCura.
""")

# Función para descargar resultados
def get_download_link(df, filename):
    towrite = BytesIO()
    df.to_excel(towrite, index=False, engine='openpyxl')
    towrite.seek(0)
    b64 = base64.b64encode(towrite.read()).decode()
    return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">⬇️ Descargar {filename}</a>'

# Carga de archivos
st.sidebar.header("📁 Cargar Archivos")
uploaded_transacciones = st.sidebar.file_uploader("Sube el archivo de transacciones (Excel)", type=["xlsx"])
uploaded_reseñas = st.sidebar.file_uploader("Sube el archivo de reseñas (Excel)", type=["xlsx"])

if uploaded_transacciones and uploaded_reseñas:
    try:
        # Cargar datasets
        df_transacciones = pd.read_excel(uploaded_transacciones)
        df_reseñas = pd.read_excel(uploaded_reseñas)

        # Validar columnas esenciales
        required_columns = ['order_id', 'customer_id', 'product_name', 'Distrito', 'Canal_Venta', 'order_date', 'total_amount']
        for col in required_columns:
            if col not in df_transacciones.columns:
                st.error(f"El archivo debe contener la columna '{col}'.")
                st.stop()

        if 'review_text' not in df_reseñas.columns:
            st.warning("No se encontró la columna 'review_text' en las reseñas.")

        st.success("✅ Archivos cargados correctamente")

        # --- Sección 1: Segmentación de Clientes (RFM) ---
        st.header("📌 Segmentación de Clientes (Clustering)")
        if st.checkbox("Ejecutar Clustering", key="clustering_checkbox"):
            rfm = df_transacciones.groupby('customer_id').agg(
                Recency=('order_date', lambda x: (pd.to_datetime("2025-05-01") - pd.to_datetime(x.max())).days),
                Frequency=('order_id', 'count'),
                Monetary=('total_amount', 'sum')
            ).reset_index()

            # Filtrar clientes recurrentes
            rfm = rfm[(rfm['Frequency'] > 1) & (rfm['Monetary'] > 30)]

            X = rfm[['Frequency', 'Monetary']].copy()
            from sklearn.preprocessing import StandardScaler
            X_scaled = StandardScaler().fit_transform(X)

            max_clusters = min(10, len(X))
            n_clusters = st.slider("Seleccionar número de clusters:", 2, max_clusters, 4)

            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            rfm['cluster'] = kmeans.fit_predict(X_scaled)

            fig, ax = plt.subplots()
            sns.scatterplot(data=rfm, x='Frequency', y='Monetary', hue='cluster', palette='viridis', alpha=0.7, ax=ax)
            st.pyplot(fig)

            # Tabla resumen de clústeres
            cluster_summary = rfm.groupby("cluster").agg(
                Clientes=("customer_id", "count"),
                Promedio_Frecuencia=("Frequency", "mean"),
                Promedio_Monto=("Monetary", "mean")
            ).reset_index()
            st.write("📊 Resumen por Cluster:")
            st.dataframe(cluster_summary.style.highlight_max(axis=0))

        # --- Sección 2: Reglas de Asociación ---
        st.header("🔗 Reglas de Asociación")
        if st.checkbox("Ejecutar reglas de asociación", key="rules_checkbox"):
            basket = df_transacciones.groupby(["order_id", "product_name"])["quantity"].sum().unstack().fillna(0)

            # ✅ Usamos .map() en lugar de applymap()
            basket = basket.map(lambda x: True if x > 0 else False)

            st.write("Matriz de productos comprados juntos (primeras filas):")
            st.dataframe(basket.head())

            min_support = st.slider("Soporte mínimo:", 0.0001, 0.05, 0.001, step=0.0001, key="apriori_slider")
            frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)

            if not frequent_itemsets.empty:
                rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
                rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
                rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))

                st.write("Reglas de Asociación Encontradas:")
                st.dataframe(rules.sort_values(by="confidence", ascending=False)[[
                    "antecedents", "consequents", "support", "confidence", "lift"
                ]].head(10))
                st.markdown(get_download_link(rules, "reglas_asociacion.xlsx"), unsafe_allow_html=True)
            else:
                st.warning("No se encontraron conjuntos frecuentes con este soporte.")

        # --- Sección 3: Análisis de Sentimiento ---
        st.header("🗣️ Análisis de Sentimiento en Reseñas")
        if st.checkbox("Ejecutar análisis de sentimiento", key="sentiment_checkbox"):
            if 'review_text' in df_reseñas.columns:
                vectorizer = TfidfVectorizer(stop_words='english')
                X_train = vectorizer.fit_transform(df_reseñas['review_text'])
                y_train = df_reseñas['sentiment']

                model = MultinomialNB()
                model.fit(X_train, y_train)

                example_text = st.text_input("Prueba con tu propia reseña:")
                if example_text:
                    prediction = model.predict(vectorizer.transform([example_text]))[0]
                    st.success(f"Sentimiento predicho: **{prediction}**")
            else:
                st.warning("El archivo de reseñas debe contener la columna 'review_text'.")

    except Exception as e:
        st.error(f"Error al procesar los datos: {str(e)}")
else:
    st.info("Por favor, suba ambos archivos Excel: transacciones y reseñas.")
