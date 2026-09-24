import mlflow
import mlflow.sklearn
import joblib

model = joblib.load("models/final_model.pkl")

mlflow.set_experiment("olist_late_delivery")

with mlflow.start_run(run_name="logistic_regression_baseline"):
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("threshold", 0.5)

    mlflow.log_metric("recall_class1", 0.51)
    mlflow.log_metric("precision_class1", 0.08)
    mlflow.log_metric("f1_class1", 0.14)

    mlflow.sklearn.log_model(model, "model", registered_model_name="olist_late_delivery_model")

    print("Run logged successfully!")