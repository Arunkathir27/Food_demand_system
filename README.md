# Food Demand Prediction System

DevOps & Agile Methodologies (23AD52C) mini project - ARUN KATHIRESAN M (24243052), AI & DS, National Engineering College.

A Flask web app that stores weekly meal order data, trains a Random Forest regression model on price,
discount, promotion and seasonality features, predicts the number of orders for a meal in a given week,
and produces an 8-week demand forecast so the kitchen can plan raw material and staffing.

## Run
    pip install -r requirements.txt
    python seed_data.py      # loads 104 weeks of sample order data
    python app.py            # http://localhost:5000  (admin / admin123)

## Test
    pytest -v

## Docker
    docker compose up --build
