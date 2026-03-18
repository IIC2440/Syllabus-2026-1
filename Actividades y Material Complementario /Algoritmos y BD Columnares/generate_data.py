"""
Data generator for Clase 03 exercise.
Generates synthetic sales data: 1M sales, 1K products, 100K customers.
Outputs CSV files ready to load into PostgreSQL and DuckDB.
"""

import csv
import random
import os
from datetime import date, timedelta

random.seed(42)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Products: 1,000 rows ---
CATEGORIES = [
    "Electronics", "Clothing", "Food", "Books", "Sports",
    "Home", "Toys", "Beauty", "Auto", "Garden",
]
CATEGORY_WEIGHTS = [25, 15, 12, 8, 8, 8, 7, 7, 5, 5]  # skewed distribution
BRANDS = [
    "AlphaGear", "BetaCo", "GammaTech", "DeltaWorks", "EpsilonLab",
    "ZetaBrand", "EtaCraft", "ThetaPro", "IotaGoods", "KappaLine",
]
# Base price ranges by category (min, max)
CATEGORY_PRICE_RANGE = {
    "Electronics": (50, 500), "Clothing": (15, 150), "Food": (2, 30),
    "Books": (8, 60), "Sports": (20, 300), "Home": (10, 400),
    "Toys": (5, 80), "Beauty": (8, 120), "Auto": (15, 250), "Garden": (10, 200),
}

def generate_products(n=1000):
    path = os.path.join(OUTPUT_DIR, "products.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "product_id", "product_name", "category", "brand",
            "base_price", "weight_kg", "rating",
        ])
        for i in range(1, n + 1):
            cat = random.choices(CATEGORIES, CATEGORY_WEIGHTS)[0]
            brand = random.choice(BRANDS)
            lo, hi = CATEGORY_PRICE_RANGE[cat]
            w.writerow([
                i,
                f"Product_{i:04d}",
                cat,
                brand,
                round(random.uniform(lo, hi), 2),
                round(random.uniform(0.1, 30), 2),
                round(random.uniform(1, 5), 1),
            ])
    print(f"Generated {n} products -> {path}")

# --- Customers: 100,000 rows ---
REGIONS = ["Norte", "Centro", "Sur", "Metropolitana", "Austral"]
CITIES = {
    "Norte": ["Antofagasta", "Iquique", "Arica", "Calama"],
    "Centro": ["Valparaíso", "Viña del Mar", "Rancagua", "Talca"],
    "Sur": ["Concepción", "Temuco", "Valdivia", "Osorno"],
    "Metropolitana": ["Santiago", "Puente Alto", "Maipú", "Las Condes"],
    "Austral": ["Puerto Montt", "Coyhaique", "Punta Arenas", "Castro"],
}
SEGMENTS = ["Individual", "Corporate", "Government"]

def generate_customers(n=100_000):
    path = os.path.join(OUTPUT_DIR, "customers.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "customer_id", "customer_name", "region", "city",
            "segment", "registration_date",
        ])
        base_date = date(2018, 1, 1)
        for i in range(1, n + 1):
            region = random.choice(REGIONS)
            city = random.choice(CITIES[region])
            reg_date = base_date + timedelta(days=random.randint(0, 2000))
            w.writerow([
                i,
                f"Customer_{i:06d}",
                region,
                city,
                random.choice(SEGMENTS),
                reg_date.isoformat(),
            ])
    print(f"Generated {n} customers -> {path}")

# --- Sales: 1,000,000 rows (wide table, 20+ columns) ---
PAYMENT_METHODS = ["Credit Card", "Debit Card", "Cash", "Transfer", "Crypto"]
CHANNELS = ["Online", "Store", "Phone", "App"]
STATUSES = ["Completed", "Returned", "Cancelled"]
STATUS_WEIGHTS = [0.92, 0.05, 0.03]

NOTES_OPTIONS = [
    "", "", "", "", "", "", "", "",  # ~80% empty
    "rush order", "gift wrap", "fragile", "bulk order",
]

def generate_sales(n=1_000_000, n_products=1000, n_customers=100_000):
    # Pre-load product base prices for realistic unit_price
    prices_path = os.path.join(OUTPUT_DIR, "products.csv")
    product_prices = {}
    with open(prices_path, "r") as pf:
        reader = csv.DictReader(pf)
        for row in reader:
            product_prices[int(row["product_id"])] = float(row["base_price"])

    path = os.path.join(OUTPUT_DIR, "sales.csv")
    start_date = date(2023, 1, 1)
    date_range = (date(2025, 12, 31) - start_date).days

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "sale_id", "sale_date", "sale_time", "product_id", "customer_id",
            "quantity", "unit_price", "discount_pct", "tax_pct", "amount",
            "payment_method", "channel", "status",
            "shipping_cost", "shipping_days", "warehouse_id",
            "salesperson_id", "promotion_id", "notes", "is_gift",
        ])
        for i in range(1, n + 1):
            sale_date = start_date + timedelta(days=random.randint(0, date_range))
            hour = random.randint(8, 22)
            minute = random.randint(0, 59)
            product_id = random.randint(1, n_products)
            customer_id = random.randint(1, n_customers)
            quantity = random.randint(1, 10)
            base = product_prices.get(product_id, 50.0)
            unit_price = round(base * random.uniform(0.8, 1.2), 2)
            discount = round(random.uniform(0, 0.3), 2)
            tax = 0.19
            subtotal = quantity * unit_price * (1 - discount)
            amount = round(subtotal * (1 + tax), 2)
            status = random.choices(STATUSES, STATUS_WEIGHTS)[0]
            promo = random.randint(1, 20) if random.random() > 0.5 else ""

            w.writerow([
                i,
                sale_date.isoformat(),
                f"{hour:02d}:{minute:02d}:00",
                product_id,
                customer_id,
                quantity,
                unit_price,
                discount,
                tax,
                amount,
                random.choice(PAYMENT_METHODS),
                random.choice(CHANNELS),
                status,
                round(random.uniform(0, 25), 2),
                random.randint(1, 15),
                random.randint(1, 10),
                random.randint(1, 50),
                promo,
                random.choice(NOTES_OPTIONS),
                random.choice([True, False]),
            ])
            if i % 200_000 == 0:
                print(f"  Sales: {i:,}/{n:,}")
    print(f"Generated {n:,} sales -> {path}")


if __name__ == "__main__":
    print("Generating synthetic dataset...")
    generate_products()
    generate_customers()
    generate_sales()
    print("Done!")
