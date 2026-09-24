# Olist Late Delivery Prediction — Inference Service

Predicts whether a new order will arrive late, based on a model trained in notebooks 01-06.

## Run it locally

```bash
python -m venv Qenv
Qenv\Scripts\activate
python -m pip install -r requirements/base.txt
uvicorn app.main:app --reload
```

Then open http://127.0.0.1:8000/docs to try the API interactively.

## Run tests

```bash
python -m pip install -r requirements/dev.txt
python -m pytest tests/ -v
```

## Run with Docker

```bash
docker compose up --build
```

## Project structure

| Folder | Purpose |
|---|---|
| `app/` | FastAPI service — health, model-info, predict, predict-batch routes |
| `config/` | `config.yaml` — all paths and settings, no hardcoded values in code |
| `data/raw/`, `data/processed/` | Raw and processed data, versioned with DVC (see `*.dvc` files) |
| `models/` | Trained model (`final_model.pkl`) + `feature_columns.json`, both loaded — never re-fit at inference |
| `src/` | `preprocessing.py`, `predict.py`, `validation.py`, `logger.py`, `config_loader.py` |
| `notebooks/` | Original training notebooks (01-06) — training stays here, not in the inference pipeline |
| `tests/` | pytest unit and integration tests |
| `requirements/` | `base.txt` (runtime) and `dev.txt` (development, includes base.txt) |

## Model

- LogisticRegression (`class_weight="balanced"`), classification threshold 0.5 (set in `config.yaml`)
- Baseline recall 0%, model recall 51% / precision 8% / F1 0.14 on the late-delivery class (see `data/processed/results_summary.md`)

## Notes

- Every prediction is logged (input, output, latency, model version) to `logs/app.log`
- Incoming orders are validated (required columns, value ranges, valid states) before reaching the model
- Data and model artifacts are tracked with DVC (local cache in this version — no remote storage configured yet)