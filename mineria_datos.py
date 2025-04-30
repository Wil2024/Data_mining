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

# Configuración de Streamlit
st.title("Simulador de Minería de Datos - Hirahoka E-commerce")
st.write("Análisis de transacciones y reseñas para segmentación, clasificación y reglas de asociación.")

# --- Carga de Datasets ---
st.header("Carga de Datasets")
uploaded_transacciones = st.file_uploader("Sube el archivo de transacciones (Excel)", type=["xlsx", "xls"])
uploaded_reseñas = st.file_uploader("Sube el archivo de reseñas (Excel)", type=["xlsx", "xls"])

# Función para cargar datos
def load_data(transacciones_file, reseñas_file):
    try:
        if transacciones_file is not None:
            df_transacciones = pd.read_excel(transacciones_file)
        else:
            st.error("Por favor, sube el archivo de transacciones.")
            return None, None
        
        if reseñas_file is not None:
            df_reseñas = pd.read_excel(reseñas_file)
        else:
            st.error("Por favor, sube el archivo de reseñas.")
            return None, None
        
        return df_transacciones, df_reseñas
    except Exception as e:
        st.error(f"Error al cargar los archivos: {e}")
        return None, None

# Cargar datos si se subieron los archivos
if uploaded_transacciones and uploaded_reseñas:
    df_transacciones, df_reseñas = load_data(uploaded_transacciones, uploaded_reseñas)
else:
    df_transacciones, df_reseñas = None, None
    st.warning("Sube ambos archivos para continuar.")

# Función para descargar DataFrame como Excel
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
    return output.getvalue()

def get_download_link(df, filename):
    excel_data = to_excel(df)
    b64 = base64.b64encode(excel_data).decode()
    return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Descargar {filename}</a>'

# Proceder solo si los datos están cargados
if df_transacciones is not None and df_reseñas is not None:
    # --- Sección 1: Segmentación de Clientes (K-means) ---
    st.header("Segmentación de Clientes (K-means)")

    # Preparar datos para clustering
    customer_features = df_transacciones.groupby("customer_id").agg({
        "total_amount": ["sum", "mean"],
        "order_id": "count",
        "category": lambda x: len(set(x))
    }).reset_index()

    customer_features.columns = ["customer_id", "total_spent", "avg_spent", "order_count", "unique_categories"]

    # Normalizar datos
    X = customer_features[["total_spent", "avg_spent", "order_count", "unique_categories"]]
    X = (X - X.mean()) / X.std()

    # Aplicar K-means
    n_clusters = st.slider("Número de clústeres:", 2, 10, 3, key="kmeans_slider")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    customer_features["cluster"] = kmeans.fit_predict(X)

    # Visualización
    fig, ax = plt.subplots()
    sns.scatterplot(data=customer_features, x="total_spent", y="order_count", hue="cluster", palette="deep", ax=ax)
    plt.title("Segmentación de Clientes")
    st.pyplot(fig)

    # Resumen por clúster
    cluster_summary = customer_features.groupby("cluster").agg({
        "total_spent": "mean",
        "avg_spent": "mean",
        "order_count": "mean",
        "unique_categories": "mean"
    }).reset_index()
    st.write("Resumen por Clúster:")
    st.dataframe(cluster_summary)

    # Descarga
    st.markdown(get_download_link(customer_features, "segmentacion_clientes.xlsx"), unsafe_allow_html=True)

    # --- Sección 2: Clasificación de Reseñas (Naive Bayes) ---
    st.header("Clasificación de Reseñas (Naive Bayes)")

    # Preparar datos para clasificación
    X_text = df_reseñas["review_text"]
    y = df_reseñas["sentiment"]

    # Vectorizar texto
    vectorizer = TfidfVectorizer(max_features=1000)
    X_vectorized = vectorizer.fit_transform(X_text)

    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

    # Entrenar modelo
    nb = MultinomialNB()
    nb.fit(X_train, y_train)

    # Predicciones
    y_pred = nb.predict(X_test)

    # Reporte de clasificación
    st.write("Reporte de Clasificación:")
    st.text(classification_report(y_test, y_pred))

    # Clasificar nuevas reseñas
    new_review = st.text_input("Ingresa una reseña para clasificar:", key="review_input")
    if new_review:
        new_review_vec = vectorizer.transform([new_review])
        prediction = nb.predict(new_review_vec)[0]
        st.write(f"Sentimiento predicho: **{prediction}**")

    # Descarga
    df_reseñas["predicted_sentiment"] = nb.predict(vectorizer.transform(df_reseñas["review_text"]))
    st.markdown(get_download_link(df_reseñas, "clasificacion_reseñas.xlsx"), unsafe_allow_html=True)

    # --- Sección 3: Reglas de Asociación ---
    st.header("Reglas de Asociación (Apriori)")

    # Preparar datos para reglas de asociación
    basket = df_transacciones.groupby(["order_id", "product_name"])["quantity"].sum().unstack().reset_index().fillna(0)
    basket = basket.set_index("order_id")
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)

    # Depuración: Mostrar matriz basket y su densidad
    st.write("Matriz de Transacciones (Primeras 5 filas):")
    st.dataframe(basket.head())
    density = basket.values.mean() * 100
    st.write(f"Densidad de la matriz (porcentaje de 1s): {density:.2f}%")

    # Aplicar Apriori
    min_support = st.slider("Soporte mínimo:", 0.00001, 0.01, 0.0001, step=0.00001, key="apriori_slider")
    try:
        frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)

        # Mostrar conjuntos frecuentes
        st.write(f"Se encontraron {len(frequent_itemsets)} conjuntos frecuentes.")
        if not frequent_itemsets.empty:
            st.write("Conjuntos Frecuentes (Top 5):")
            st.dataframe(frequent_itemsets.head())

        # Generar reglas
        if not frequent_itemsets.empty:
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0)
            if not rules.empty:
                rules = rules[["antecedents", "consequents", "support", "confidence", "lift"]]
                rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
                rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))

                # Mostrar reglas
                st.write("Reglas de Asociación:")
                st.dataframe(rules)

                # Descarga
                st.markdown(get_download_link(rules, "reglas_asociacion.xlsx"), unsafe_allow_html=True)
            else:
                st.warning("No se encontraron reglas de asociación. Posibles causas: combinaciones de productos poco frecuentes o datos insuficientes. Intenta reducir aún más el soporte mínimo.")
        else:
            st.warning("No se encontraron conjuntos frecuentes. Intenta reducir el soporte mínimo o verifica que el dataset tenga suficientes co-ocurrencias de productos.")
    except Exception as e:
        st.error(f"Error al generar reglas de asociación: {e}")