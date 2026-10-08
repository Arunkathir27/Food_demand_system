from flask import Flask, render_template, request, redirect, url_for, session, flash
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import sqlite3

app = Flask(__name__)
app.secret_key = "devops-mini-project-secret"
DB = "food_demand.db"
MEALS = ["Veg Biryani", "Chicken Biryani", "Paneer Pizza",
         "Masala Dosa", "Veg Noodles", "Chocolate Shake"]
FEATURES = ["week_of_year", "meal_id", "base_price", "checkout_price",
            "discount_pct", "email_promo", "featured"]
TEST_WEEKS = 16          # last 16 weeks are kept for testing
MODEL = {}


def get_db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        week INTEGER NOT NULL,
        meal TEXT NOT NULL,
        base_price REAL NOT NULL,
        checkout_price REAL NOT NULL,
        email_promo INTEGER NOT NULL DEFAULT 0,
        featured INTEGER NOT NULL DEFAULT 0,
        num_orders INTEGER NOT NULL,
        UNIQUE(week, meal)
    )""")
    conn.commit()
    conn.close()


def make_features(r):
    """Converts one order record into the model's feature vector."""
    discount = (r["base_price"] - r["checkout_price"]) / r["base_price"] * 100
    return [(r["week"] - 1) % 52 + 1, MEALS.index(r["meal"]), r["base_price"],
            r["checkout_price"], round(discount, 2), r["email_promo"], r["featured"]]


def train_model(rows):
    """Trains on older weeks, evaluates on the latest weeks, then refits on all data."""
    X = np.array([make_features(r) for r in rows], dtype=float)
    y = np.array([r["num_orders"] for r in rows], dtype=float)
    weeks = np.array([r["week"] for r in rows])
    split = weeks.max() - TEST_WEEKS
    train, test = weeks <= split, weeks > split
    rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
    rf.fit(X[train], y[train])
    pred = rf.predict(X[test])
    metrics = {"r2": round(r2_score(y[test], pred), 3),
               "mae": round(mean_absolute_error(y[test], pred), 1),
               "rmse": round(float(np.sqrt(mean_squared_error(y[test], pred))), 1),
               "train_rows": int(train.sum()), "test_rows": int(test.sum())}
    weekly = {}
    for w, a, p in zip(weeks[test], y[test], pred):
        weekly.setdefault(int(w), [0, 0])
        weekly[int(w)][0] += a
        weekly[int(w)][1] += p
    importance = sorted(zip(FEATURES, rf.feature_importances_), key=lambda t: -t[1])
    final = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42).fit(X, y)
    MODEL.update(model=final, metrics=metrics, last_week=int(weeks.max()),
                 weekly=[(w, round(a), round(p)) for w, (a, p) in sorted(weekly.items())],
                 importance=[(f, round(float(v) * 100, 1)) for f, v in importance])
    return metrics


def predict_orders(record):
    return int(round(MODEL["model"].predict([make_features(record)])[0]))


def logged_in():
    return session.get("logged_in")


def ensure_model(conn):
    if not MODEL:
        rows = conn.execute("SELECT * FROM orders").fetchall()
        if len(rows) < 50:
            return False
        train_model(rows)
    return True


@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.form["username"] == "admin" and request.form["password"] == "admin123":
            session["logged_in"] = True
            return redirect(url_for("dashboard"))
        flash("Invalid username or password")
    return render_template("login.html")


@app.route("/dashboard")
def dashboard():
    if not logged_in():
        return redirect(url_for("login"))
    conn = get_db()
    total = conn.execute("SELECT COALESCE(SUM(num_orders),0) FROM orders").fetchone()[0]
    weeks = conn.execute("SELECT COUNT(DISTINCT week) FROM orders").fetchone()[0]
    trend = conn.execute("SELECT week, SUM(num_orders) AS orders FROM orders "
                         "GROUP BY week ORDER BY week").fetchall()
    by_meal = conn.execute("SELECT meal, SUM(num_orders) AS orders FROM orders "
                           "GROUP BY meal ORDER BY orders DESC").fetchall()
    ready = ensure_model(conn)
    conn.close()
    return render_template("dashboard.html", total=total, weeks=weeks, trend=trend,
                           by_meal=by_meal, metrics=MODEL.get("metrics") if ready else None)


@app.route("/orders", methods=["GET", "POST"])
def orders():
    if not logged_in():
        return redirect(url_for("login"))
    conn = get_db()
    if request.method == "POST":
        try:
            conn.execute(
                "INSERT INTO orders (week, meal, base_price, checkout_price, email_promo, "
                "featured, num_orders) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (int(request.form["week"]), request.form["meal"],
                 float(request.form["base_price"]), float(request.form["checkout_price"]),
                 1 if request.form.get("email_promo") else 0,
                 1 if request.form.get("featured") else 0, int(request.form["num_orders"])))
            conn.commit()
            MODEL.clear()
            flash("Order record added - retrain the model to include it")
        except sqlite3.IntegrityError:
            flash("Record for this week and meal already exists")
        except ValueError:
            flash("Enter valid numeric values")
    rows = conn.execute("SELECT * FROM orders ORDER BY week DESC, meal LIMIT 18").fetchall()
    conn.close()
    return render_template("orders.html", rows=rows, meals=MEALS)


@app.route("/train")
def train():
    if not logged_in():
        return redirect(url_for("login"))
    conn = get_db()
    rows = conn.execute("SELECT * FROM orders").fetchall()
    conn.close()
    if len(rows) < 50:
        flash("At least 50 order records are needed to train the model")
        return redirect(url_for("orders"))
    train_model(rows)
    return render_template("train.html", m=MODEL["metrics"], weekly=MODEL["weekly"],
                           importance=MODEL["importance"])


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if not logged_in():
        return redirect(url_for("login"))
    conn = get_db()
    ready = ensure_model(conn)
    conn.close()
    result = None
    if request.method == "POST" and ready:
        record = {"week": int(request.form["week"]), "meal": request.form["meal"],
                  "base_price": float(request.form["base_price"]),
                  "checkout_price": float(request.form["checkout_price"]),
                  "email_promo": 1 if request.form.get("email_promo") else 0,
                  "featured": 1 if request.form.get("featured") else 0}
        result = predict_orders(record)
    elif request.method == "POST":
        flash("Train the model first")
    return render_template("predict.html", meals=MEALS, result=result, form=request.form)


@app.route("/forecast")
def forecast():
    if not logged_in():
        return redirect(url_for("login"))
    conn = get_db()
    if not ensure_model(conn):
        conn.close()
        flash("Train the model first")
        return redirect(url_for("orders"))
    prices = {r["meal"]: r["base_price"] for r in conn.execute(
        "SELECT meal, base_price FROM orders GROUP BY meal")}
    conn.close()
    start = MODEL["last_week"] + 1
    weeks = list(range(start, start + 8))
    table = []
    for meal in MEALS:
        values = [predict_orders({"week": w, "meal": meal, "base_price": prices[meal],
                                  "checkout_price": prices[meal], "email_promo": 0,
                                  "featured": 0}) for w in weeks]
        table.append({"meal": meal, "weekly": values, "total": sum(values)})
    totals = [sum(t["weekly"][i] for t in table) for i in range(len(weeks))]
    return render_template("forecast.html", weeks=weeks, table=table, totals=totals)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=True)
