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

# Configuración inicial de Streamlit
st.set_page_config(page_title="Simulador Minería de Datos", layout="wide")
st.title("🧠 Simulador de Data Mining")
st.markdown("### Aplicando técnicas de minería de datos a un e-commerce")

# --- Función para descargar DataFrame como Excel ---
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
    return output.getvalue()

def get_download_link(df, filename):
    excel_data = to_excel(df)
    b64 = base64.b64encode(excel_data).decode()  # Codificar a base64
    return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Descargar {filename}</a>'




# --- Carga de Datasets ---
st.sidebar.header("📁 Cargar Archivos")
uploaded_transacciones = st.sidebar.file_uploader("Sube el archivo de transacciones (Excel)", type=["xlsx"])
uploaded_reseñas = st.sidebar.file_uploader("Sube el archivo de reseñas (Excel)", type=["xlsx"])

# Función para cargar archivos
def load_data(transacciones_file, reseñas_file):
    try:
        df_transacciones = pd.read_excel(transacciones_file)
        df_reseñas = pd.read_excel(reseñas_file)
        return df_transacciones, df_reseñas
    except Exception as e:
        st.error(f"Error al cargar los archivos: {e}")
        return None, None

if uploaded_transacciones and uploaded_reseñas:
    df_transacciones, df_reseñas = load_data(uploaded_transacciones, uploaded_reseñas)

    # Mostrar vista previa de los datasets
    st.subheader("📊 Vista Previa de Datos")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Transacciones:")
        st.dataframe(df_transacciones.head())
    with col2:
        st.write("Reseñas:")
        st.dataframe(df_reseñas.head())

    # --- Sección 1: Segmentación de Clientes (K-means) ---
    st.header("📌 Segmentación de Clientes (Clustering)")
    if st.checkbox("Ejecutar segmentación", key="clustering_checkbox"):
        customer_features = df_transacciones.groupby("customer_id").agg({
            "total_amount": ["sum", "mean"],
            "order_id": "count",
            "category": lambda x: len(set(x))
        }).reset_index()
        customer_features.columns = ["customer_id", "total_spent", "avg_spent", "order_count", "unique_categories"]
        
        X = customer_features[["total_spent", "avg_spent", "order_count", "unique_categories"]]
        X = (X - X.mean()) / X.std()

        n_clusters = st.slider("Número de clústeres:", 2, 10, 3, key="kmeans_slider")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        customer_features["cluster"] = kmeans.fit_predict(X)

        # Visualización
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=customer_features, x="total_spent", y="order_count", hue="cluster", palette="deep", ax=ax)
        plt.title("Segmentación de Clientes")
        st.pyplot(fig)

        cluster_summary = customer_features.groupby("cluster").agg({
            "total_spent": "mean",
            "avg_spent": "mean",
            "order_count": "mean",
            "unique_categories": "mean"
        }).reset_index()
        st.write("Resumen por Clúster:")
        st.dataframe(cluster_summary)

        # Descargable
        st.markdown(get_download_link(customer_features, "segmentacion_clientes.xlsx"), unsafe_allow_html=True)





    # --- Sección 2: Clasificación de Reseñas (Naive Bayes) ---
    st.header("📝 Análisis de Sentimiento en Reseñas")
    if st.checkbox("Ejecutar clasificación", key="sentiment_checkbox"):
        X_text = df_reseñas["review_text"]
        y = df_reseñas["sentiment"]

        vectorizer = TfidfVectorizer(max_features=1000)
        X_vectorized = vectorizer.fit_transform(X_text)

        X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
        nb = MultinomialNB()
        nb.fit(X_train, y_train)

        y_pred = nb.predict(X_test)
        st.write("Reporte de Clasificación:")
        st.text(classification_report(y_test, y_pred))

        new_review = st.text_input("Ingresa una nueva reseña para analizar:", key="review_input")
        if new_review:
            new_review_vec = vectorizer.transform([new_review])
            prediction = nb.predict(new_review_vec)[0]
            st.success(f"Sentimiento predicho: **{prediction}**")

        df_reseñas["predicted_sentiment"] = nb.predict(vectorizer.transform(df_reseñas["review_text"]))
        st.markdown(get_download_link(df_reseñas, "clasificacion_reseñas.xlsx"), unsafe_allow_html=True)

    # --- Sección 3: Reglas de Asociación (Apriori) ---
    st.header("🔗 Análisis de Patrones de Compra")
    if st.checkbox("Ejecutar reglas de asociación", key="rules_checkbox"):
        basket = df_transacciones.groupby(["order_id", "product_name"])["quantity"].sum().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)
        st.write("Matriz de productos comprados juntos (primeras filas):")
        st.dataframe(basket.head())

        min_support = st.slider("Soporte mínimo:", 0.0001, 0.05, 0.001, step=0.0001, key="apriori_slider")
        try:
            frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)
            if not frequent_itemsets.empty:
                st.write(f"Conjuntos frecuentes encontrados ({len(frequent_itemsets)}):")
                st.dataframe(frequent_itemsets.head())

                rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
                rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
                rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))

                st.write("Reglas de Asociación:")
                st.dataframe(rules.sort_values(by="confidence", ascending=False))

                st.markdown(get_download_link(rules, "reglas_asociacion.xlsx"), unsafe_allow_html=True)
            else:
                st.warning("No se encontraron conjuntos frecuentes.")
        except Exception as e:
            st.error(f"Error al generar las reglas: {e}")

    # --- Exportar todo en ZIP (Opcional) ---
    if st.checkbox("📥 Descargar todo en carpeta comprimida"):
        def zip_files(files):
            import zipfile
            import tempfile
            temp_dir = tempfile.mkdtemp()
            zip_path = f"{temp_dir}/resultados.zip"
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for filename, df in files.items():
                    path = f"{temp_dir}/{filename}"
                    df.to_excel(path, index=False)
                    zipf.write(path, arcname=filename)
            return zip_path

        files_to_zip = {
            "segmentacion.xlsx": customer_features,
            "reseñas_clasificadas.xlsx": df_reseñas,
            "reglas.xlsx": rules
        }
        zip_path = zip_files(files_to_zip)
        with open(zip_path, "rb") as f:
            st.download_button("Descargar Todo (ZIP)", data=f, file_name="resultados_analisis.zip", mime="application/zip")

else:
    st.info("Por favor, sube ambos archivos Excel: uno de transacciones y otro de reseñas.")


# Footer
st.markdown(
    """
    <div style='text-align: center; font-size: 12px; margin-top: 50px; color: #666;'>
        ©️ 2025 Diseñado por <b>Wilton Torvisco</b> | 
        <a href='https://github.com/Wil2024' target='_blank'>GitHub</a> | 
        Todos los derechos reservados.
    </div>
    """,
    unsafe_allow_html=True
)
