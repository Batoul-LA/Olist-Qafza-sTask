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
    result = predict(sample_order)
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