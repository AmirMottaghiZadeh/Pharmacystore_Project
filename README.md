# PharmacyStore

Forecast weekly pharmacy sales (UPW = unique packets per week) for an Iranian pharmacy using XGBoost on tabular data.

## Highlights
- Limited 10-month dataset and tabular features
- SQL Server extraction pipeline
- Feature engineering: weekly aggregates, rolling stats, ATC classification, holiday effects
- XGBoost training with time-based validation and walk-forward evaluation

## Forecast Target & Evaluation
| Component | Standard |
| --- | --- |
| Target (UPW) | Weekly unique prescriptions per drug: `UPW = nunique(PrID)` aggregated by `DrugId` and `week_start` (week buckets start on Monday), filtered to `SalePrice > 0`. |
| Time split | Train = oldest weeks, validation = middle weeks, test = newest weeks (70/15/15 by week, no leakage across weeks). |
| Baselines | Naive (last week), Moving average (last 4 weeks), Seasonal naive (t-52 weeks, if available). |
| Metrics | MAE, RMSE, WAPE, sMAPE. |

## Repository Layout
- `src/pharmacystore/`: core package
- `scripts/`: runnable entry points
- `data/external/`: reference data (WHO ATC, holidays)
- `data/processed/`: generated outputs (gitignored)
- `artifacts/`: trained models and artifacts (gitignored)

## Setup
1. Install dependencies:
   `pip install -e .`
2. Set SQL connection values in `.env` or environment variables (auto-loaded).
3. Place external data files:
   - `data/external/who_atc_ddd.csv`
   - `data/external/holidays/weekly_official_holidays.csv`
   - Optional (for building holiday summaries): `data/external/holidays/persian_holidays.csv`

## Config
The app loads `.env` automatically. See `.env.example` for a template. Key variables:
- `PHARMACYSTORE_SQL_SERVER`, `PHARMACYSTORE_SQL_DATABASE`, `PHARMACYSTORE_SQL_USERNAME`, `PHARMACYSTORE_SQL_PASSWORD`
- `PHARMACYSTORE_SQL_DRIVER` (default: `ODBC Driver 18 for SQL Server`)
- `PHARMACYSTORE_SQL_ENCRYPT` (default: `false`)
- `PHARMACYSTORE_SQL_TRUST_CERT` (default: `true`)
- `PHARMACYSTORE_RANDOM_SEED` (default: `42`)
- `PHARMACYSTORE_RUN_WALK_FORWARD` (default: `true`)

## Run
- Full pipeline (features + model training):
  `python -m pharmacystore.pipeline full`
- Build features and export weekly dataset:
  `python -m pharmacystore.pipeline run`
- Train the main XGBoost model:
  `python -m pharmacystore.pipeline train`
- Train the baseline XGBoost model:
  `python -m pharmacystore.pipeline train-baseline`
- Optional run tag to label artifacts:
  `python -m pharmacystore.pipeline train --run-tag exp1`

Legacy script entry points still work:
- `python scripts/run_pipeline.py`
- `python scripts/train_model.py`
- `python scripts/train_baseline.py`
- Scale sales categories:
  `python scripts/scale_sales.py`
- Preview ATC classification:
  `python scripts/classify_atc.py`

## Reproducibility
- Fixed seeds for numpy/random/xgboost via `PHARMACYSTORE_RANDOM_SEED`.
- Each run writes artifacts to `artifacts/runs/<run_id>/` with:
  `config.json`, `train_params.json`, `metrics.json`, `data_manifest.json`,
  plus per-run predictions and model files.
- `data_manifest.json` records dataset hash and train/test week ranges.

## Data Privacy
`data/processed/` and `artifacts/` are ignored by Git to keep sensitive outputs and model files out of the repository.
