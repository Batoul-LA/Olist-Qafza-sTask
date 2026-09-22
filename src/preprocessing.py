import pandas as pd


def preprocess(data, feature_columns):
    """
    Applies the same transformations as notebook 05 to raw order data.

    data: dict (single order) or DataFrame (batch of orders)
    feature_columns: list of column names the model was trained on
                      (loaded from models/feature_columns.json)

    Returns a DataFrame with exactly `feature_columns`, in the same order.
    """
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data.copy()

    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_weekday"] = df["order_purchase_timestamp"].dt.weekday
    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour

    numeric_cols = ["num_items", "total_price", "total_freight",
                    "total_payment_value", "num_payment_methods"]
    for col in numeric_cols:
        df[col] = df[col].fillna(0)

    state_encoded = pd.get_dummies(df["customer_state"], prefix="state")
    state_cols = [c for c in feature_columns if c.startswith("state_")]
    state_encoded = state_encoded.reindex(columns=state_cols, fill_value=0)

    df_final = pd.concat([df, state_encoded], axis=1)
    return df_final.reindex(columns=feature_columns, fill_value=0)