import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

# Configuración inicial
fake = Faker()
np.random.seed(42)
random.seed(42)

# Lista de productos ampliada
products = [
    {"id": "P001", "name": "Refrigeradora LG 400L", "category": "Refrigeradoras", "price": 2500},
    {"id": "P002", "name": "Televisor Samsung 55''", "category": "Televisores", "price": 3200},
    {"id": "P003", "name": "Lavadora Mabe 15kg", "category": "Lavadoras", "price": 1800},
    {"id": "P004", "name": "Microondas Oster 1.1", "category": "Microondas", "price": 450},
    {"id": "P005", "name": "Aspiradora Philips", "category": "Aspiradoras", "price": 600},
    {"id": "P006", "name": "Licuadora Oster 1.5L", "category": "Licuadoras", "price": 200},
    {"id": "P007", "name": "Horno Eléctrico Black+Decker", "category": "Hornos", "price": 350},
    {"id": "P008", "name": "Cafetera Nespresso", "category": "Cafeteras", "price": 500},
    {"id": "P009", "name": "Plancha a Vapor Philips", "category": "Planchas", "price": 150},
    {"id": "P010", "name": "Batidora KitchenAid", "category": "Batidoras", "price": 800},
    {"id": "P011", "name": "Parlante JBL Bluetooth", "category": "Audio", "price": 400},
    {"id": "P012", "name": "Aire Acondicionado LG 12000BTU", "category": "Aire Acondicionado", "price": 2000},
    {"id": "P013", "name": "Ventilador Taurus", "category": "Ventiladores", "price": 120},
    {"id": "P014", "name": "Tostadora Oster", "category": "Tostadoras", "price": 100},
    {"id": "P015", "name": "Calefactor Electrolux", "category": "Calefactores", "price": 250},
]

# Patrones de compra comunes (para simular co-ocurrencias realistas)
purchase_patterns = [
    ["Televisor Samsung 55''", "Parlante JBL Bluetooth"],  # TV + Parlante
    ["Refrigeradora LG 400L", "Microondas Oster 1.1"],     # Refrigeradora + Microondas
    ["Licuadora Oster 1.5L", "Batidora KitchenAid", "Tostadora Oster"],  # Electrodomésticos de cocina
    ["Lavadora Mabe 15kg", "Plancha a Vapor Philips"],    # Lavadora + Plancha
    ["Aire Acondicionado LG 12000BTU", "Ventilador Taurus"],  # Climatización
]

# Generar dataset de transacciones
n_customers = 1000
n_orders = 10000  # Más órdenes para mayor densidad

transacciones = []
for order_idx in range(n_orders):
    customer_id = f"C{random.randint(1, n_customers):04d}"
    order_id = f"O{order_idx:05d}"
    order_date = fake.date_time_between(start_date="-1y", end_date="now")
    
    # Decidir si usar un patrón común (50% de probabilidad) o selección aleatoria
    if random.random() < 0.5:
        # Usar un patrón de compra común
        selected_products = random.choice(purchase_patterns)
        selected_products = [p for p in products if p["name"] in selected_products]
    else:
        # Seleccionar entre 2 y 5 productos aleatoriamente
        n_products = random.randint(2, 5)
        selected_products = random.sample(products, n_products)
    
    for product in selected_products:
        quantity = random.randint(1, 3)
        total_amount = product["price"] * quantity
        region = random.choice(["Lima", "Arequipa", "Trujillo", "Cusco", "Piura"])
        
        transacciones.append({
            "customer_id": customer_id,
            "order_id": order_id,
            "order_date": order_date,
            "product_id": product["id"],
            "product_name": product["name"],
            "category": product["category"],
            "price": product["price"],
            "quantity": quantity,
            "total_amount": total_amount,
            "region": region
        })

df_transacciones = pd.DataFrame(transacciones)

# Generar dataset de reseñas
reseñas = []
for _, row in df_transacciones.sample(frac=0.3).iterrows():
    rating = random.randint(1, 5)
    if rating >= 4:
        sentiment = "Positivo"
        review_text = random.choice([
            "Excelente producto, superó mis expectativas.",
            "Muy buena calidad, lo recomiendo.",
            "Entrega rápida y producto en perfecto estado."
        ])
    elif rating <= 2:
        sentiment = "Negativo"
        review_text = random.choice([
            "El producto llegó defectuoso, muy decepcionado.",
            "Tardó mucho en llegar, mal servicio.",
            "No cumple con lo prometido."
        ])
    else:
        sentiment = "Neutral"
        review_text = random.choice([
            "El producto está bien, pero esperaba más.",
            "Correcto, sin mayores problemas.",
            "Aceptable, pero podría mejorar."
        ])
    
    reseñas.append({
        "customer_id": row["customer_id"],
        "order_id": row["order_id"],
        "review_text": review_text,
        "rating": rating,
        "sentiment": sentiment
    })

df_reseñas = pd.DataFrame(reseñas)

# Guardar datasets
df_transacciones.to_csv("transacciones_hirahoka.csv", index=False)
df_reseñas.to_csv("reseñas_hirahoka.csv", index=False)