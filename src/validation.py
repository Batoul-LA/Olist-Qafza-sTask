import pandas as pd


def validate_order(data):
    """
    Lightweight validation on raw incoming order data.
    Returns (is_valid: bool, errors: list[str])
    """
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data.copy()

    errors = []

    required_cols = [
        "order_purchase_timestamp", "num_items", "total_price",
        "total_freight", "total_payment_value", "num_payment_methods",
        "customer_state"
    ]
    for col in required_cols:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")

    if "num_items" in df.columns and (df["num_items"] < 0).any():
        errors.append("num_items must be >= 0")

    if "total_price" in df.columns and (df["total_price"] < 0).any():
        errors.append("total_price must be >= 0")

    valid_states = [
        "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG",
        "MS", "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR",
        "RS", "SC", "SE", "SP", "TO"
    ]
    if "customer_state" in df.columns:
        invalid_states = df[~df["customer_state"].isin(valid_states)]
        if not invalid_states.empty:
            errors.append(f"Invalid customer_state values found: {invalid_states['customer_state'].unique().tolist()}")

    return len(errors) == 0, errors