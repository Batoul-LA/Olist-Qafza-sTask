import numpy as np

from src.predict import predict

def test_predict_valid_order():
    sample_order = {
        "order_purchase_timestamp": "2024-05-15 14:30:00",
        "num_items": 2,
        "total_price": 150.0,
        "total_freight": 20.0,
        "total_payment_value": 170.0,
        "num_payment_methods": 1,
        "customer_state": "SP"
    }
    class DummyModel:
        def predict(self, X):
            return [1]

        def predict_proba(self, X):
            return np.array([[0.2, 0.8]])

    feature_columns = [
        "num_items",
        "total_price",
        "total_freight",
        "total_payment_value",
        "num_payment_methods",
        "purchase_month",
        "purchase_weekday",
        "purchase_hour",
        "state_SP",
    ]

    result = predict(sample_order, model=DummyModel(), feature_columns=feature_columns)
    assert "prediction" in result[0]
    assert "probability" in result[0]
    assert result[0]["prediction"] in [0, 1]

def test_predict_missing_field_raises():
    bad_order = {"num_items": 2, "customer_state": "SP"}
    try:
        predict(bad_order)
        assert False, "Expected ValueError"
    except ValueError:
        pass