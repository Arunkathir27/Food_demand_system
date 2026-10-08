import pytest
import app as food_app
import seed_data


@pytest.fixture
def client(tmp_path):
    food_app.DB = str(tmp_path / "test.db")
    food_app.MODEL.clear()
    food_app.app.config["TESTING"] = True
    food_app.init_db()
    conn = food_app.get_db()
    conn.executemany("INSERT INTO orders (week, meal, base_price, checkout_price, email_promo, "
                     "featured, num_orders) VALUES (?, ?, ?, ?, ?, ?, ?)", list(seed_data.records()))
    conn.commit()
    conn.close()
    with food_app.app.test_client() as c:
        yield c


def login(c):
    return c.post("/", data={"username": "admin", "password": "admin123"}, follow_redirects=True)


def test_feature_discount():
    f = food_app.make_features({"week": 53, "meal": "Masala Dosa", "base_price": 100,
                                "checkout_price": 80, "email_promo": 1, "featured": 0})
    assert f[0] == 1 and f[1] == 3 and f[4] == 20.0


def test_login_success_and_failure(client):
    assert b"Dashboard" in login(client).data
    bad = client.post("/", data={"username": "x", "password": "y"}, follow_redirects=True)
    assert b"Invalid username or password" in bad.data


def test_protected_page_redirects(client):
    assert client.get("/train").status_code == 302


def test_training_accuracy(client):
    login(client)
    assert client.get("/train").status_code == 200
    assert food_app.MODEL["metrics"]["r2"] > 0.8


def test_discount_increases_prediction(client):
    login(client)
    client.get("/train")
    base = {"week": 105, "meal": "Veg Biryani", "base_price": 220, "email_promo": 0, "featured": 0}
    full = food_app.predict_orders(dict(base, checkout_price=220))
    disc = food_app.predict_orders(dict(base, checkout_price=176))
    assert disc > full
    assert b"8-Week Demand Forecast" in client.get("/forecast").data
