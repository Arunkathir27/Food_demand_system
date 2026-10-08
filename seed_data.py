"""Generates a sample dataset of 104 weeks of meal orders for demonstration."""
import math
import random
import app

random.seed(52)
MEALS = {"Veg Biryani": (220, 420), "Chicken Biryani": (280, 520), "Paneer Pizza": (320, 260),
         "Masala Dosa": (120, 380), "Veg Noodles": (180, 300), "Chocolate Shake": (150, 220)}


def records():
    for week in range(1, 105):
        season = 1 + 0.18 * math.sin(2 * math.pi * week / 52)
        trend = 1 + week * 0.0005
        for meal, (price, base) in MEALS.items():
            discount = random.choice([0, 0, 0, 0.1, 0.15, 0.2])
            promo = 1 if random.random() < 0.15 else 0
            featured = 1 if random.random() < 0.12 else 0
            shake = 1 + (0.25 if meal == "Chocolate Shake" else 0) * math.sin(2 * math.pi * (week - 8) / 52)
            demand = base * season * trend * shake * (1 + 2.2 * discount) * (1 + 0.35 * promo) * (1 + 0.25 * featured)
            yield (week, meal, price, round(price * (1 - discount)), promo, featured,
                   int(demand * random.uniform(0.92, 1.08)))


if __name__ == "__main__":
    app.init_db()
    conn = app.get_db()
    conn.executemany("INSERT OR IGNORE INTO orders (week, meal, base_price, checkout_price, "
                     "email_promo, featured, num_orders) VALUES (?, ?, ?, ?, ?, ?, ?)", list(records()))
    conn.commit()
    conn.close()
    print("Inserted sample weekly order data")
