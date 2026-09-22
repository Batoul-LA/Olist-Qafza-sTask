import time
import joblib
import json
from src.config_loader import load_config
from src.preprocessing import preprocess
from src.logger import get_logger

config = load_config()
logger = get_logger(
    "predict",
    log_file=config["logging"]["log_file"],
    level=config["logging"]["level"]
)


def load_artifacts(cfg=None):
    cfg = cfg or config
    model = joblib.load(cfg["paths"]["model_file"])
    with open(cfg["paths"]["feature_columns_file"], "r") as f:
        feature_columns = json.load(f)
    logger.info(f"Loaded model from {cfg['paths']['model_file']}")
    return model, feature_columns, cfg


def predict(data, model=None, feature_columns=None, cfg=None):
    start_time = time.time()

    try:
        if model is None or feature_columns is None:
            model, feature_columns, cfg = load_artifacts(cfg)

        X = preprocess(data, feature_columns)
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]

        results = [
            {"prediction": int(p), "probability": float(prob)}
            for p, prob in zip(predictions, probabilities)
        ]

        latency_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(
            f"input={data} | output={results} | latency_ms={latency_ms} "
            f"| model_version=final_model_v1"
        )
        return results

    except Exception as e:
        latency_ms = round((time.time() - start_time) * 1000, 2)
        logger.error(f"Prediction failed | input={data} | error={e} | latency_ms={latency_ms}")
        raise