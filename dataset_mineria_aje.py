import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

# Configuración inicial
fake = Faker()
np.random.seed(42)
random.seed(42)

# --- Definición de productos por categoría ---
products_by_category = {
    "Aguas y Gaseosas": [
        {"id": "P001", "name": "Inca Kola 2.5L", "price": 8.5},
        {"id": "P002", "name": "Coca Cola 2.5L", "price": 9.0},
        {"id": "P003", "name": "Agua Cielo sin Gas 2.5L(6 unds)", "price": 13.9},
        {"id": "P004", "name": "Pepsi 2L", "price": 7.8},
        {"id": "P005", "name": "Seven Up 2L", "price": 7.6},
	{"id": "P006", "name": "Sporade sabor Tropical 500ml (12 unds)", "price": 22},
	{"id": "P007", "name": "Volt Ginseng 300ml (12 unds)", "price": 22},
	{"id": "P008", "name": "BIG Cola 3.03L x1un", "price": 6.15},
	{"id": "P009", "name": "KR sabor Piña 3.03L x1un", "price": 5.60},
	{"id": "P010", "name": "gaseosa EVERNESS Ginger Ale 1.5L", "price": 5}
    ],
    "Abarrotes": [
        {"id": "P006", "name": "Arroz Costeño 1kg", "price": 5.2},
        {"id": "P007", "name": "Fideos Maruchan 80g", "price": 2.0},
        {"id": "P008", "name": "Lentejas San Jorge 500g", "price": 4.8},
        {"id": "P009", "name": "Azúcar Pureza 1kg", "price": 4.6},
        {"id": "P010", "name": "Aceite Primor 900ml", "price": 12.0}
    ],
    "Snacks y Cereales": [
        {"id": "P011", "name": "Papas Lay's Saladas 150g", "price": 3.5},
        {"id": "P012", "name": "Chocolates Bauducco Mix 200g", "price": 6.0},
        {"id": "P013", "name": "Corn Flakes Kellogg's 300g", "price": 10.0},
        {"id": "P014", "name": "Galletas Soda Integral 150g", "price": 2.8},
        {"id": "P015", "name": "Maní salado La Cubana 100g", "price": 4.0}
    ],
    "Cuidado de la Ropa y Hogar": [
        {"id": "P016", "name": "Detergente Ace 800g", "price": 14.0},
        {"id": "P017", "name": "Suavizante Downy 750ml", "price": 16.5},
        {"id": "P018", "name": "Jabón líquido Fabuloso 1L", "price": 12.0},
        {"id": "P019", "name": "Limpiavidrios Mr. Musculo 500ml", "price": 9.5},
        {"id": "P020", "name": "Desinfectante Clorox 1L", "price": 11.0}
    ],
    "Cervezas y Licores": [
        {"id": "P021", "name": "Cerveza Cusqueña Lager 355ml (6 unds)", "price": 30},
        {"id": "P022", "name": "Cerveza Pilsen Callao 355ml (6 unds)", "price": 27},
        {"id": "P023", "name": "Ron Cartavio XO 750ml", "price": 60.0},
        {"id": "P024", "name": "Vodka Smirnoff 750ml", "price": 55.0},
        {"id": "P025", "name": "Whisky J&B 750ml", "price": 95.0},
	{"id": "P026", "name": "Cerveza Tres Cruces Pum Pum Lager Lata 473ml", "price": 20.4}
    ]
}

# --- Patrones de compra realistas ---
purchase_patterns = [
    ["Inca Kola 2.5L", "Papas Lay's Saladas 150g"],  # Refresco + snack
    ["Arroz Costeño 1kg", "Aceite Primor 900ml", "Azúcar Pureza 1kg"],  # Compra básica
    ["Corn Flakes Kellogg's 300g", "Leche Evaporada Nestlé 400g"],  # Desayuno
    ["Detergente Ace 800g", "Jabón líquido Fabuloso 1L"],  # Limpieza
    ["Cerveza Cusqueña Lager 620ml", "Chocolates Bauducco Mix 200g"]  # Fiesta
]

# --- Generar datos de clientes (con emails) ---
clientes = []
for i in range(1, 1501):  # 1500 clientes
    nombre = fake.name()
    email = fake.email()
    clientes.append({
        "customer_id": f"C{i:04d}",
        "nombre": nombre,
        "email": email,
        "telefono": fake.phone_number(),
        "direccion": fake.address().replace("\n", ", "),
        "fecha_registro": fake.date_between(start_date="-2y", end_date="today")
    })

df_clientes = pd.DataFrame(clientes)

# --- Generar transacciones ---
transacciones = []
for order_idx in range(28000):  # 10,000 órdenes
    customer = random.choice(clientes)
    customer_id = customer["customer_id"]
    order_id = f"O{order_idx:05d}"
    order_date = fake.date_time_between(start_date="-6m", end_date="now")
    distrito = random.choice(["Lima cercado", "San Juan de Miraflores", "Barranco", "Chorillos", "Breña", "San Juan de Lurigancho", "Rímac", "Surco", "Miraflores", "Jesús María", "San Miguel", "Comas", "Los Olivos", "San Martín de Porres", "San Borja", "La victoria", "Surquillo"])
    
    if random.random() < 0.5:
        selected_pattern = random.choice(purchase_patterns)
        category = [cat for cat, prods in products_by_category.items() if any(prod["name"] in selected_pattern for prod in prods)][0]
        selected_products = [p for p in products_by_category[category] if p["name"] in selected_pattern]
    else:
        category = random.choice(list(products_by_category.keys()))
        selected_products = random.sample(products_by_category[category], random.randint(1, 3))
        
    for product in selected_products:
        quantity = random.randint(1, 3)
        total_amount = product["price"] * quantity
        transacciones.append({
            "customer_id": customer_id,
            "order_id": order_id,
            "order_date": order_date,
            "product_id": product["id"],
            "product_name": product["name"],
            "category": category,
            "price": product["price"],
            "quantity": quantity,
            "total_amount": total_amount,
            "place": distrito,
            "email": customer["email"]
        })

df_transacciones = pd.DataFrame(transacciones)

# --- Generar reseñas ---
reseñas = []
for _, row in df_transacciones.sample(frac=0.3).iterrows():
    rating = random.randint(1, 5)
    if rating >= 4:
        sentiment = "Positivo"
        review_text = random.choice([
            "Excelente producto, rápido y eficiente.",
            "Muy buen servicio, volveré a comprar.",
            "Entrega puntual y empaque perfecto."
        ])
    elif rating <= 2:
        sentiment = "Negativo"
        review_text = random.choice([
            "Producto llegó roto, muy mala experiencia.",
            "Demoró demasiado, no volveré a comprar aquí.",
            "No cumple con lo descrito."
        ])
    else:
        sentiment = "Neutral"
        review_text = random.choice([
            "Buen producto, pero podría mejorar el soporte técnico.",
            "Correcto, sin grandes problemas.",
            "Lo usaré por un tiempo antes de dar una opinión definitiva."
        ])
    reseñas.append({
        "customer_id": row["customer_id"],
        "order_id": row["order_id"],
        "review_text": review_text,
        "rating": rating,
        "sentiment": sentiment
    })

df_reseñas = pd.DataFrame(reseñas)

# --- Guardar datasets en Excel ---
df_transacciones.to_excel("transacciones_aje_delivery.xlsx", index=False)
df_reseñas.to_excel("reseñas_aje_delivery.xlsx", index=False)
df_clientes.to_excel("clientes_aje_delivery.xlsx", index=False)

print("✅ Datasets generados exitosamente:")
print("- transacciones_aje_delivery.xlsx")
print("- reseñas_aje_delivery.xlsx")
print("- clientes_aje_delivery.xlsx")